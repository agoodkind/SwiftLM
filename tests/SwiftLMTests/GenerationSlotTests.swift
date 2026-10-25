import XCTest
import Foundation
@testable import SwiftLM

/// GenerationSlot guarantees the AsyncSemaphore slot acquired after `wait()`
/// is returned exactly once — on explicit `release()` or `deinit` (whichever
/// comes first). Without it, any throw between `wait()` and the success-path
/// `signal()` calls permanently leaked the slot; with the default
/// `--parallel 1` a single failed request wedged the server until restart.
final class GenerationSlotTests: XCTestCase {

    /// Race `wait()` against a timeout: returns true if the wait completed.
    private func waitCompletes(_ sem: AsyncSemaphore, timeoutNs: UInt64 = 1_000_000_000) async -> Bool {
        await withTaskGroup(of: Bool.self) { group in
            group.addTask {
                await sem.wait()
                await sem.signal()  // immediately give it back; only completion matters
                return true
            }
            group.addTask {
                try? await Task.sleep(nanoseconds: timeoutNs)
                return false
            }
            let first = await group.next() ?? false
            group.cancelAll()
            return first
        }
    }

    /// Simulates the error path: slot created after `wait()`, then dropped
    /// without an explicit release (scope unwinds on throw). The deinit must
    /// reclaim the slot so the next request can proceed.
    func testDeinitReclaimsSlotWithoutExplicitRelease() async {
        let sem = AsyncSemaphore(limit: 1)
        await sem.wait()
        do {
            let slot = GenerationSlot(semaphore: sem)
            _ = slot
        }  // deinit → release

        let ok = await waitCompletes(sem)
        XCTAssertTrue(ok, "slot was not reclaimed by deinit — server would wedge after one error at --parallel 1")
    }

    /// Explicit release on the success path, then deinit when the frame unwinds:
    /// the once-flag must prevent a second signal (which would over-admit and
    /// break the parallel limit when other requests are queued).
    func testExplicitReleaseThenDeinitDoesNotDoubleSignal() async {
        let sem = AsyncSemaphore(limit: 1)
        await sem.wait()

        let slot = GenerationSlot(semaphore: sem)
        slot.release()          // success-path release
        // slot deinits at end of scope → second release attempt, must no-op

        var bAdmitted = false
        let bTask = Task {
            await sem.wait()
            bAdmitted = true
            // hold the slot until the test finishes asserting
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            await sem.signal()
        }
        // Let B queue and be admitted by the single legitimate release.
        try? await Task.sleep(nanoseconds: 200_000_000)
        XCTAssertTrue(bAdmitted, "legitimate release should admit exactly one waiter")

        // If double-signalling had happened, the semaphore would have counted
        // an extra free slot and this second wait would complete while B still
        // holds the only slot.
        var cAdmitted = false
        let cTask = Task {
            await sem.wait()
            cAdmitted = true
            await sem.signal()
        }
        try? await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertFalse(cAdmitted, "double-release over-admitted a second concurrent request (parallel limit broken)")

        await bTask.value
        await cTask.value
        _ = slot
    }

    /// Multiple explicit releases (e.g. a success path plus a deferred cleanup
    /// that both run) must not signal twice either: exactly one waiter is
    /// admitted while the holder still holds the slot.
    func testRepeatedReleaseIsIdempotent() async {
        let sem = AsyncSemaphore(limit: 1)
        await sem.wait()
        let slot = GenerationSlot(semaphore: sem)
        slot.release()
        slot.release()
        slot.release()

        var bAdmitted = false
        let bTask = Task {
            await sem.wait()
            bAdmitted = true
            // hold the slot so any over-admission would surface as C below
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            await sem.signal()
        }
        try? await Task.sleep(nanoseconds: 200_000_000)
        XCTAssertTrue(bAdmitted, "single slot must be reclaimed exactly once")

        var cAdmitted = false
        let cTask = Task {
            await sem.wait()
            cAdmitted = true
            await sem.signal()
        }
        try? await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertFalse(cAdmitted, "extra signals leaked additional slots beyond the limit")

        await bTask.value
        await cTask.value
        withExtendedLifetime(slot) {}  // deinit runs here; flag must make it a no-op
    }
}
