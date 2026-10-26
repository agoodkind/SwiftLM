import XCTest
import MLX
import MLXLMCommon
@testable import SwiftLM

/// The prompt cache keeps several entries so interleaved sessions (a side request between
/// two agent turns) don't evict each other's prefix.
final class PromptCacheLRUTests: XCTestCase {

    private func makeCache(seqLen T: Int, fill: Float) -> [any KVCache] {
        let attn = KVCacheSimple()
        _ = attn.update(keys: MLXArray.ones([1, 2, T, 4], dtype: .float16) * fill,
                        values: MLXArray.ones([1, 2, T, 4], dtype: .float16) * fill)
        return [attn]
    }

    private func makeHybrid(seqLen T: Int, fill: Float) -> [any KVCache] {
        let mamba = MambaCache()
        mamba.state = [MLXArray.ones([1, 3, 8]) * fill, MLXArray.ones([1, 2, 4, 4]) * fill]
        return makeCache(seqLen: T, fill: fill) + [mamba]
    }

    private func hit(_ pc: PromptCache, _ tokens: [Int]) async -> Int? {
        await pc.restore(newTokens: tokens, into: [KVCacheSimple()])
    }

    /// `count` consecutive token ids from `start`.
    private func seq(_ start: Int, _ count: Int) -> [Int] { Array(start ..< start + count) }

    /// Save `tokens` with a KV state whose every element is `fill`.
    private func put(_ pc: PromptCache, _ tokens: [Int], fill: Float = 1) async {
        await pc.save(tokens: tokens, cache: makeCache(seqLen: tokens.count, fill: fill))
    }

    /// Entries still held (destructive: empties the cache).
    private func drain(_ pc: PromptCache) async -> Int {
        await pc.evict(keepMostRecent: false)
    }

    /// Sum of every key element, i.e. `fill * 8 * T` for a `makeCache` layer.
    private func keySum(_ layer: any KVCache) -> Float {
        layer.state[0].asType(.float32).sum().item(Float.self)
    }

    /// A ring fed `n` single tokens; values encode the token index.
    private func makeRing(maxSize: Int, tokens n: Int) -> RotatingKVCache {
        let ring = RotatingKVCache(maxSize: maxSize, keep: 0, step: 4)
        for t in 0 ..< n {
            let k = MLXArray([Float(t)]).reshaped([1, 1, 1, 1])
            _ = ring.update(keys: k, values: k)
        }
        return ring
    }

    func testTwoUnrelatedPromptsBothHitWithTwoEntries() async {
        let pc = PromptCache(maxEntries: 2)
        await pc.save(tokens: [1, 2, 3, 4], cache: makeCache(seqLen: 4, fill: 1))
        await pc.save(tokens: [9, 8, 7, 6], cache: makeCache(seqLen: 4, fill: 2))
        let a = await hit(pc, [1, 2, 3, 4, 5])
        let b = await hit(pc, [9, 8, 7, 6, 5])
        XCTAssertEqual(a, 4)
        XCTAssertEqual(b, 4)
    }

    func testSingleEntryKeepsTodaysBehaviour() async {
        let pc = PromptCache()
        await pc.save(tokens: [1, 2, 3, 4], cache: makeCache(seqLen: 4, fill: 1))
        await pc.save(tokens: [9, 8, 7, 6], cache: makeCache(seqLen: 4, fill: 2))
        let first = await hit(pc, [1, 2, 3, 4, 5])
        XCTAssertNil(first)
    }

    func testHitRefreshesRecencyAndLeastRecentIsDropped() async {
        let pc = PromptCache(maxEntries: 2)
        await pc.save(tokens: [1, 2, 3, 4], cache: makeCache(seqLen: 4, fill: 1))
        await pc.save(tokens: [9, 8, 7, 6], cache: makeCache(seqLen: 4, fill: 2))
        _ = await hit(pc, [1, 2, 3, 4, 5])  // [1..] is now most recent
        await pc.save(tokens: [5, 5, 5, 5], cache: makeCache(seqLen: 4, fill: 3))  // evicts [9..]
        let kept = await hit(pc, [1, 2, 3, 4, 5])
        let dropped = await hit(pc, [9, 8, 7, 6, 5])
        XCTAssertEqual(kept, 4)
        XCTAssertNil(dropped)
    }

    func testExtensionReplacesItsPrefixEntry() async {
        let pc = PromptCache(maxEntries: 2)
        await pc.save(tokens: [1, 2, 3], cache: makeCache(seqLen: 3, fill: 1))
        await pc.save(tokens: [9, 9, 9], cache: makeCache(seqLen: 3, fill: 2))
        await pc.save(tokens: [1, 2, 3, 4, 5], cache: makeCache(seqLen: 5, fill: 1))
        // The other session must survive: the linear conversation replaced its own entry.
        let other = await hit(pc, [9, 9, 9, 1])
        let conv = await hit(pc, [1, 2, 3, 4, 5, 6])
        XCTAssertEqual(other, 3)
        XCTAssertEqual(conv, 5)
    }

    func testRestorePicksLongestMatch() async {
        let pc = PromptCache(maxEntries: 3)
        await pc.save(tokens: [1, 2, 3, 4, 5, 6], cache: makeCache(seqLen: 6, fill: 1))
        await pc.save(tokens: [1, 2, 9, 9, 9, 9], cache: makeCache(seqLen: 6, fill: 2))
        let n = await hit(pc, [1, 2, 3, 4, 7, 8])
        XCTAssertEqual(n, 4)
    }

    func testHybridExactPrefixPicksLongestEntry() async {
        let pc = PromptCache(maxEntries: 3)
        await pc.save(tokens: [1, 2, 3], cache: makeHybrid(seqLen: 3, fill: 1), allowRecurrent: true)
        await pc.save(tokens: [7, 7, 7, 7], cache: makeHybrid(seqLen: 4, fill: 3), allowRecurrent: true)
        await pc.save(tokens: [1, 2, 3, 4, 5], cache: makeHybrid(seqLen: 5, fill: 2), allowRecurrent: true)
        // The 5-token entry replaced the 3-token one, so [1,2,3,4,5,...] reuses all 5.
        let fresh: [any KVCache] = [KVCacheSimple(), MambaCache()]
        let n = await pc.restoreExactPrefix(newTokens: [1, 2, 3, 4, 5, 6], limit: 5, into: fresh)
        XCTAssertEqual(n, 5)
        let other = await pc.restoreExactPrefix(newTokens: [7, 7, 7, 7, 1], limit: 4,
                                                into: [KVCacheSimple(), MambaCache()])
        XCTAssertEqual(other, 4, "the other session's snapshot survives")
    }

    func testEvictKeepMostRecentAndEvictAll() async {
        let pc = PromptCache(maxEntries: 3)
        await pc.save(tokens: [1, 2, 3, 4], cache: makeCache(seqLen: 4, fill: 1))
        await pc.save(tokens: [9, 8, 7, 6], cache: makeCache(seqLen: 4, fill: 2))
        await pc.evict(keepMostRecent: true)
        let old = await hit(pc, [1, 2, 3, 4, 5])
        let recent = await hit(pc, [9, 8, 7, 6, 5])
        XCTAssertNil(old)
        XCTAssertEqual(recent, 4)
        await pc.evict(keepMostRecent: false)
        let none = await hit(pc, [9, 8, 7, 6, 5])
        XCTAssertNil(none)
    }

    // MARK: - Superseding a stale entry of the same conversation

    /// The prompt saved for turn N is not always an exact prefix of turn N+1: templates
    /// that re-render history drop or rewrite the generation-prompt tokens it ended with
    /// (Qwen3 `<think>` handling), and edits or regenerations change the last message. That
    /// entry must be replaced, not left to push another conversation out of the cache.
    func testNearPrefixReplacesItsStaleEntryInsteadOfEvictingTheOtherSession() async {
        let pc = PromptCache(maxEntries: 2)
        await put(pc, [9, 9, 9, 9], fill: 1)
        await put(pc, [1, 2, 3, 4, 5, 6], fill: 2)
        let partial = await hit(pc, [1, 2, 3, 4, 7, 8, 9])
        XCTAssertEqual(partial, 4)
        await put(pc, [1, 2, 3, 4, 7, 8, 9], fill: 3)

        let other = await hit(pc, [9, 9, 9, 9, 0])
        XCTAssertEqual(other, 4, "the other session survives")
        let stale = await hit(pc, [1, 2, 3, 4, 5, 6, 0])
        XCTAssertEqual(stale, 4, "the stale entry is gone: only the replacement's 4-token prefix matches")
        let held = await drain(pc)
        XCTAssertEqual(held, 2)
    }

    func testPingPongWithReRenderedHistoryKeepsTheOtherConversation() async {
        let b1 = seq(5000, 40)
        let history = seq(0, 60)
        // `<|im_start|>assistant\n<think>\n\n</think>\n\n`: 7 tokens that the next turn's
        // re-rendered history does not repeat. This pins the slack above that real tail.
        let a1 = history + seq(900, 7)
        let a2 = history + seq(700, 25)
        let pc = PromptCache(maxEntries: 2)
        await put(pc, b1)
        await put(pc, a1)
        let partial = await hit(pc, a2)
        XCTAssertEqual(partial, history.count)
        await put(pc, a2)

        let b = await hit(pc, b1 + [1])
        let a = await hit(pc, a2 + [1])
        let staleA1 = await hit(pc, a1 + [1])
        XCTAssertEqual(b, b1.count, "B1 survives the A1 -> A2 turn")
        XCTAssertEqual(a, a2.count)
        XCTAssertEqual(staleA1, history.count, "A1's tail is gone")
        let held = await drain(pc)
        XCTAssertEqual(held, 2)
    }

    func testSupersedesWhenTheStaleTailIsExactlyTheSlack() async {
        let slack = PromptCache.supersedeSlack
        let old = seq(0, 3 * slack)
        let other = seq(1000, 3 * slack)
        let new = seq(0, 2 * slack) + seq(2000, 2 * slack)  // drops the last `slack` tokens of old
        let pc = PromptCache(maxEntries: 2)
        await put(pc, other)  // least recent: it is what a leftover `old` would push out
        await put(pc, old)
        await put(pc, new)

        let oldTail = await hit(pc, old + [5])
        let otherHit = await hit(pc, other + [5])
        let newHit = await hit(pc, new + [5])
        XCTAssertEqual(oldTail, 2 * slack, "old was replaced; only the shared prefix remains reusable")
        XCTAssertEqual(otherHit, other.count)
        XCTAssertEqual(newHit, new.count)
        let held = await drain(pc)
        XCTAssertEqual(held, 2)
    }

    func testKeepsBothWhenTheStaleTailExceedsTheSlack() async {
        let slack = PromptCache.supersedeSlack
        let old = seq(0, 3 * slack)
        let other = seq(1000, 3 * slack)
        let new = seq(0, 2 * slack - 1) + seq(2000, 2 * slack)  // old's tail is slack + 1
        let pc = PromptCache(maxEntries: 3)
        await put(pc, old)
        await put(pc, other)
        await put(pc, new)

        let oldHit = await hit(pc, old + [5])
        let otherHit = await hit(pc, other + [5])
        let newHit = await hit(pc, new + [5])
        XCTAssertEqual(oldHit, old.count)
        XCTAssertEqual(otherHit, other.count)
        XCTAssertEqual(newHit, new.count)
        let held = await drain(pc)
        XCTAssertEqual(held, 3)
    }

    /// Two conversations that share a long system prompt and then diverge are different
    /// conversations. A request that only restored the shared system prompt from one must
    /// not make saving the other evict it.
    func testConversationsSharingOnlyASystemPromptBothStay() async {
        // Fixed sizes on purpose: 20-token tails against a 32-token shared system prompt.
        // The shared part outweighs each tail, so only the slack keeps them apart.
        XCTAssertLessThan(PromptCache.supersedeSlack, 20, "the tails must exceed the slack for this test to mean anything")
        let system = seq(0, 32)
        let a = system + seq(1000, 20)
        let b = system + seq(2000, 20)
        let pc = PromptCache(maxEntries: 2)
        await put(pc, a)
        let shared = await hit(pc, b)
        XCTAssertEqual(shared, system.count, "B's request reuses only the shared system prompt")
        await put(pc, b)

        let aHit = await hit(pc, a + [7])
        let bHit = await hit(pc, b + [7])
        XCTAssertEqual(aHit, a.count)
        XCTAssertEqual(bHit, b.count)
        let held = await drain(pc)
        XCTAssertEqual(held, 2)
    }

    /// Entries that only start alike (a shared BOS or template header) and are shorter than
    /// the slack must not be replaced by an unrelated prompt.
    func testShortEntriesSharingOnlyAHeaderAreNotSuperseded() async {
        let pc = PromptCache(maxEntries: 2)
        await put(pc, [0, 5, 5, 5, 5])
        await put(pc, [0, 6, 6, 6, 6])
        let first = await hit(pc, [0, 5, 5, 5, 5, 1])
        let second = await hit(pc, [0, 6, 6, 6, 6, 1])
        XCTAssertEqual(first, 5)
        XCTAssertEqual(second, 5)
        let held = await drain(pc)
        XCTAssertEqual(held, 2)
    }

    func testSavingTheSamePromptAgainReplacesIt() async {
        let pc = PromptCache(maxEntries: 2)
        await put(pc, [1, 2, 3, 4])
        await put(pc, [1, 2, 3, 4])
        let held = await drain(pc)
        XCTAssertEqual(held, 1)
    }

    func testEmptyPromptIsNeverSaved() async {
        let pc = PromptCache(maxEntries: 1)
        await put(pc, [1, 2, 3])
        await pc.save(tokens: [], cache: makeCache(seqLen: 2, fill: 2))
        let kept = await hit(pc, [1, 2, 3, 4])
        XCTAssertEqual(kept, 3, "an unrestorable empty entry must not push a real one out")
        let held = await drain(pc)
        XCTAssertEqual(held, 1)
    }

    /// Recurrent snapshots are valid only at exactly `tokens.count`, so only an exact prefix
    /// replaces one. The generic rule would collapse this pair.
    func testHybridSavesKeepTheStrictPrefixRule() async {
        let older = seq(0, 6)
        let newer = seq(0, 4) + [7, 8, 9]  // diverges from `older` in its last 2 tokens
        let hybrid = PromptCache(maxEntries: 2)
        await hybrid.save(tokens: older, cache: makeHybrid(seqLen: 6, fill: 1), allowRecurrent: true)
        await hybrid.save(tokens: newer, cache: makeHybrid(seqLen: 7, fill: 2), allowRecurrent: true)
        let olderHit = await hybrid.restoreExactPrefix(
            newTokens: older + [10], limit: 6, into: [KVCacheSimple(), MambaCache()])
        let newerHit = await hybrid.restoreExactPrefix(
            newTokens: newer + [10], limit: 7, into: [KVCacheSimple(), MambaCache()])
        XCTAssertEqual(olderHit, 6)
        XCTAssertEqual(newerHit, 7)
        let hybridHeld = await drain(hybrid)
        XCTAssertEqual(hybridHeld, 2)

        let generic = PromptCache(maxEntries: 2)
        await put(generic, older)
        await put(generic, newer)
        let genericHeld = await drain(generic)
        XCTAssertEqual(genericHeld, 1, "the same pair collapses on the generic path")
    }

    /// The pure rule behind the tests above.
    func testIsSupersededRule() {
        let slack = PromptCache.supersedeSlack
        let old = seq(0, 3 * slack)
        func superseded(_ new: [Int], exactOnly: Bool = false, saved: [Int]? = nil) -> Bool {
            PromptCache.isSuperseded(saved ?? old, by: new, exactPrefixOnly: exactOnly)
        }
        XCTAssertTrue(superseded(old), "identical")
        XCTAssertTrue(superseded(old + [1]), "extension")
        XCTAssertTrue(superseded(seq(0, 2 * slack) + seq(900, 5)), "stale tail == slack")
        XCTAssertFalse(superseded(seq(0, 2 * slack - 1) + seq(900, 5)), "stale tail == slack + 1")
        XCTAssertFalse(superseded(seq(900, 40)), "no overlap is never superseded")
        XCTAssertFalse(superseded([]), "nothing shared")
        // Shared part must outweigh the stale part, whatever the slack.
        XCTAssertTrue(superseded([1, 2, 3, 9], saved: [1, 2, 3, 4]), "3 shared, 1 stale")
        XCTAssertFalse(superseded([1, 2, 9], saved: [1, 2, 3, 4]), "2 shared, 2 stale")
        XCTAssertFalse(superseded([1, 2, 9, 9, 9, 9], saved: [1, 2, 3, 4, 5, 6]), "short entry, minority shared")
        XCTAssertTrue(superseded([1, 2, 3, 4, 7, 8, 9], saved: [1, 2, 3, 4, 5, 6]))
        // Recurrent snapshots: exact prefix only.
        XCTAssertTrue(superseded(old + [1], exactOnly: true))
        XCTAssertTrue(superseded(old, exactOnly: true))
        XCTAssertFalse(superseded(seq(0, 2 * slack) + seq(900, 5), exactOnly: true))
        XCTAssertFalse(superseded(Array(old.dropLast()), exactOnly: true), "shorter prompt does not replace a longer snapshot")
    }

    /// One save replaces every entry it dominates, including ones that do not dominate each
    /// other.
    func testOneSaveSupersedesEveryDominatedEntry() async {
        let prefix = seq(0, 30)
        let old1 = prefix + seq(1000, 20)
        let old2 = prefix + seq(2000, 8)
        let new = Array(old1.prefix(45)) + seq(3000, 10)

        let pair = PromptCache(maxEntries: 3)
        await put(pair, old1)
        await put(pair, old2)
        let pairHeld = await drain(pair)
        XCTAssertEqual(pairHeld, 2, "setup: old1 and old2 differ in more than the slack, so they coexist")

        let pc = PromptCache(maxEntries: 3)
        await put(pc, old1)
        await put(pc, old2)
        await put(pc, new)  // shares 45 of old1 and 30 of old2: both are within the slack
        let held = await drain(pc)
        XCTAssertEqual(held, 1)
    }

    // MARK: - Sliding-window (ring buffer) caches

    /// A ring that has dropped tokens (offset beyond its window) can be rewound by one slot
    /// at most, so it cannot stand in for the entry a near-prefix save would replace: that
    /// would turn a full hit into a miss, not into a few tokens of lost reuse. Wrapped rings
    /// keep the exact-prefix rule. Two conversations that share a system prompt and have
    /// short tails (a short first user message) must both stay.
    func testWrappedRingsSharingASystemPromptBothStay() async {
        let tail = 10
        XCTAssertLessThanOrEqual(tail, PromptCache.supersedeSlack, "the tails must be within the slack for this test to mean anything")
        let system = seq(0, 200)
        let a = system + seq(1000, tail)
        let b = system + seq(2000, tail)
        let pc = PromptCache(maxEntries: 2)
        await pc.save(tokens: a, cache: [makeRing(maxSize: 32, tokens: a.count)])
        await pc.save(tokens: b, cache: [makeRing(maxSize: 32, tokens: b.count)])

        // A's next turn extends A exactly: a full hit on A, not a miss.
        let fresh = RotatingKVCache(maxSize: 32, keep: 0, step: 4)
        let n = await pc.restore(newTokens: a + seq(3000, 30), into: [fresh])
        XCTAssertEqual(n, a.count)
        XCTAssertEqual(fresh.offset, a.count)
        let held = await drain(pc)
        XCTAssertEqual(held, 2)
    }

    /// Gemma-style cache: full-attention layers beside sliding-window ones. One wrapped ring
    /// is enough, whichever layer it is.
    func testMixedCacheWithAWrappedRingKeepsTheExactPrefixRule() async {
        func mixed(_ n: Int) -> [any KVCache] {
            makeCache(seqLen: n, fill: 1) + [makeRing(maxSize: 32, tokens: n)]
        }
        let system = seq(0, 200)
        let a = system + seq(1000, 10)
        let b = system + seq(2000, 10)
        let pc = PromptCache(maxEntries: 2)
        await pc.save(tokens: a, cache: mixed(a.count))
        await pc.save(tokens: b, cache: mixed(b.count))

        let n = await pc.restore(newTokens: a + seq(3000, 30),
                                 into: [KVCacheSimple(), RotatingKVCache(maxSize: 32, keep: 0, step: 4)])
        XCTAssertEqual(n, a.count)
        let held = await drain(pc)
        XCTAssertEqual(held, 2)
    }

    /// The exact-prefix rule still applies to a wrapped ring: a turn that extends it
    /// replaces it.
    func testWrappedRingIsStillReplacedByAnExactExtension() async {
        let turn1 = seq(0, 80)
        let pc = PromptCache(maxEntries: 2)
        await pc.save(tokens: turn1, cache: [makeRing(maxSize: 32, tokens: 80)])
        await pc.save(tokens: turn1 + seq(500, 20), cache: [makeRing(maxSize: 32, tokens: 100)])
        let held = await drain(pc)
        XCTAssertEqual(held, 1)
    }

    /// A ring that has not dropped tokens rewinds like any attention cache, so the
    /// near-prefix rule applies to it (the guard is the wrapped state, not the cache type).
    func testUnwrappedRingsReplaceANearPrefix() async {
        let old = seq(0, 30)
        let new = seq(0, 25) + seq(900, 5)
        let pc = PromptCache(maxEntries: 2)
        await pc.save(tokens: old, cache: [makeRing(maxSize: 64, tokens: old.count)])
        await pc.save(tokens: new, cache: [makeRing(maxSize: 64, tokens: new.count)])

        let fresh = RotatingKVCache(maxSize: 64, keep: 0, step: 4)
        let n = await pc.restore(newTokens: old + [7], into: [fresh])
        XCTAssertEqual(n, 25, "old was replaced; the replacement still restores the shared prefix")
        XCTAssertEqual(fresh.offset, 25)
        let held = await drain(pc)
        XCTAssertEqual(held, 1)
    }

    /// How far a wrapped ring may be rewound on restore: the one slot a full-match replay
    /// overwrites, nothing deeper.
    func testWrappedRingRewindBoundaries() async {
        let saved = seq(0, 80)
        func restoreInto(_ newTokens: [Int]) async -> Int? {
            let pc = PromptCache(maxEntries: 1)
            await pc.save(tokens: saved, cache: [makeRing(maxSize: 32, tokens: 80)])
            return await pc.restore(newTokens: newTokens,
                                    into: [RotatingKVCache(maxSize: 32, keep: 0, step: 4)])
        }
        let extended = await restoreInto(saved + [1])
        let replayed = await restoreInto(saved)                 // excess 0, replays one token
        let oneBack = await restoreInto(seq(0, 79) + [500])     // excess 1
        let oneBackReplayed = await restoreInto(seq(0, 79))     // excess 1 plus the replay
        let twoBack = await restoreInto(seq(0, 78) + [500])     // excess 2
        XCTAssertEqual(extended, 80)
        XCTAssertEqual(replayed, 80)
        XCTAssertEqual(oneBack, 79)
        XCTAssertNil(oneBackReplayed)
        XCTAssertNil(twoBack)
    }

    /// A ring that is exactly full has evicted nothing, so it may be rewound as deep as a
    /// plain cache.
    func testFullButNotWrappedRingRewindsDeeply() async {
        let pc = PromptCache(maxEntries: 1)
        await pc.save(tokens: seq(0, 32), cache: [makeRing(maxSize: 32, tokens: 32)])
        let fresh = RotatingKVCache(maxSize: 32, keep: 0, step: 4)
        let n = await pc.restore(newTokens: seq(0, 20) + [500], into: [fresh])
        XCTAssertEqual(n, 20)
        XCTAssertEqual(fresh.offset, 20)
    }

    /// The cached state must hold more tokens than are trimmed off it: an excess equal to the
    /// length is rejected, one less is not. The 4-token state under a 10-token prompt stands
    /// in for a sliding-window layer that kept fewer tokens than the prompt.
    func testExcessEqualToTheCachedLengthIsAMissOneLessIsAHit() async {
        func restoreInto(_ newTokens: [Int]) async -> Int? {
            let pc = PromptCache(maxEntries: 1)
            await pc.save(tokens: seq(0, 10), cache: makeCache(seqLen: 4, fill: 1))
            return await pc.restore(newTokens: newTokens, into: [KVCacheSimple()])
        }
        let equal = await restoreInto(seq(0, 6) + [500])     // excess 4 == 4 cached
        let oneLess = await restoreInto(seq(0, 7) + [500])   // excess 3 < 4
        XCTAssertNil(equal)
        XCTAssertEqual(oneLess, 7)
    }

    // MARK: - Choosing between several entries

    /// The longest usable match wins, not the most recent entry.
    func testRestorePicksTheLongestMatchRegardlessOfRecency() async {
        let slack = PromptCache.supersedeSlack
        let long = seq(0, 3 * slack)
        let short = seq(0, slack)  // a prefix of `long`, saved after it, and the most recent
        let pc = PromptCache(maxEntries: 2)
        await put(pc, long, fill: 1)
        await put(pc, short, fill: 2)

        let fresh = KVCacheSimple()
        let n = await pc.restore(newTokens: long + [999], into: [fresh])
        XCTAssertEqual(n, long.count)
        XCTAssertEqual(fresh.offset, long.count)
        XCTAssertEqual(keySum(fresh), 8 * Float(long.count) * 1, "restored from the long entry's KV")
    }

    func testEqualMatchesPreferTheMostRecentEntry() async {
        let slack = PromptCache.supersedeSlack
        let common = seq(0, 2 * slack)
        let a = common + seq(1000, slack + 4)
        let b = common + seq(2000, slack + 4)
        let pc = PromptCache(maxEntries: 2)
        await put(pc, a, fill: 1)
        await put(pc, b, fill: 2)

        let fresh = KVCacheSimple()
        let n = await pc.restore(newTokens: common + seq(3000, 5), into: [fresh])
        XCTAssertEqual(n, common.count)
        XCTAssertEqual(keySum(fresh), 8 * Float(common.count) * 2, "tie goes to the most recent entry (b)")
    }

    /// A diverging tail restores the shared prefix and trims the rest.
    func testTailDivergenceRestoresTheSharedPrefixAndTrimsTheRest() async {
        let pc = PromptCache(maxEntries: 2)
        await put(pc, seq(0, 30))
        let fresh = KVCacheSimple()
        let n = await pc.restore(newTokens: seq(0, 27) + [900, 901, 902], into: [fresh])
        XCTAssertEqual(n, 27)
        XCTAssertEqual(fresh.offset, 27)
    }

    /// A candidate whose restore would corrupt a layer is skipped, and the next one is used,
    /// even when the rejected one matched longer and is the most recent.
    func testWrappedRingCandidateIsSkippedForAUsableEntry() async {
        let pc = PromptCache(maxEntries: 2)
        // Unwrapped ring (offset 30 <= maxSize 32): any trim is exact.
        await pc.save(tokens: seq(0, 10) + seq(700, 20), cache: [makeRing(maxSize: 32, tokens: 30)])
        // Wrapped ring (offset 80 > maxSize 32): rewinding it by 4 would point the window at
        // evicted keys. Saved last, so it is the first candidate, and the longer match.
        await pc.save(tokens: seq(0, 80), cache: [makeRing(maxSize: 32, tokens: 80)])

        let fresh = RotatingKVCache(maxSize: 32, keep: 0, step: 4)
        let n = await pc.restore(newTokens: seq(0, 76) + [500, 501], into: [fresh])
        XCTAssertEqual(n, 10)
        XCTAssertEqual(fresh.offset, 10)
        let stats = await pc.stats()
        XCTAssertEqual(stats.hits, 1)
        XCTAssertEqual(stats.misses, 0)
    }

    func testEveryCandidateRejectedIsAMiss() async {
        let pc = PromptCache(maxEntries: 2)
        await pc.save(tokens: seq(0, 80), cache: [makeRing(maxSize: 32, tokens: 80)])
        await pc.save(tokens: seq(1000, 80), cache: [makeRing(maxSize: 32, tokens: 80)])
        let fresh = RotatingKVCache(maxSize: 32, keep: 0, step: 4)
        let n = await pc.restore(newTokens: seq(0, 76) + [500, 501], into: [fresh])
        XCTAssertNil(n)
        XCTAssertEqual(fresh.offset, 0, "the live cache is untouched")
        let stats = await pc.stats()
        XCTAssertEqual(stats.misses, 1)
    }

    /// An entry whose trim would empty a layer is skipped. The 4-token state under a
    /// 10-token prompt stands in for a sliding-window layer that kept fewer tokens than
    /// the prompt it was saved with.
    func testEntryThatWouldBeTrimmedToNothingIsSkippedForAUsableEntry() async {
        let pc = PromptCache(maxEntries: 2)
        await put(pc, seq(0, 2) + seq(100, 20), fill: 2)                       // usable
        await pc.save(tokens: seq(0, 10), cache: makeCache(seqLen: 4, fill: 1))  // excess 8 >= 4

        let fresh = KVCacheSimple()
        let n = await pc.restore(newTokens: seq(0, 2) + seq(300, 5), into: [fresh])
        XCTAssertEqual(n, 2)
        XCTAssertEqual(fresh.offset, 2)
        XCTAssertEqual(keySum(fresh), 8 * 2 * 2, "restored from the usable entry, not the rejected one")
    }

    /// Hybrid restore takes the longest exact prefix within the limit, not the most recent.
    func testHybridExactPrefixPicksTheLongestEntryNotTheMostRecent() async {
        let long = seq(0, 50)
        let short = seq(0, 20)
        let pc = PromptCache(maxEntries: 2)
        await pc.save(tokens: long, cache: makeHybrid(seqLen: 50, fill: 1), allowRecurrent: true)
        await pc.save(tokens: short, cache: makeHybrid(seqLen: 20, fill: 2), allowRecurrent: true)

        let fresh: [any KVCache] = [KVCacheSimple(), MambaCache()]
        let n = await pc.restoreExactPrefix(newTokens: long + [99], limit: 50, into: fresh)
        XCTAssertEqual(n, 50)
        XCTAssertEqual(fresh[1].state[1].sum().item(Float.self), 1 * 32, "the long snapshot's recurrent state")

        let capped: [any KVCache] = [KVCacheSimple(), MambaCache()]
        let m = await pc.restoreExactPrefix(newTokens: long + [99], limit: 30, into: capped)
        XCTAssertEqual(m, 20, "the long snapshot is past the limit, so the short one is used")
        XCTAssertEqual(capped[1].state[1].sum().item(Float.self), 2 * 32)
    }

    /// A hybrid hit refreshes recency too, so the next save evicts the other snapshot.
    func testHybridExactPrefixHitRefreshesRecency() async {
        let h1 = seq(0, 10)
        let h2 = seq(100, 10)
        let h3 = seq(200, 10)
        let pc = PromptCache(maxEntries: 2)
        await pc.save(tokens: h1, cache: makeHybrid(seqLen: 10, fill: 1), allowRecurrent: true)
        await pc.save(tokens: h2, cache: makeHybrid(seqLen: 10, fill: 2), allowRecurrent: true)
        let touched = await pc.restoreExactPrefix(newTokens: h1 + [9], limit: 10,
                                                  into: [KVCacheSimple(), MambaCache()])
        XCTAssertEqual(touched, 10)
        await pc.save(tokens: h3, cache: makeHybrid(seqLen: 10, fill: 3), allowRecurrent: true)  // evicts h2

        let kept = await pc.restoreExactPrefix(newTokens: h1 + [9], limit: 10,
                                               into: [KVCacheSimple(), MambaCache()])
        let dropped = await pc.restoreExactPrefix(newTokens: h2 + [9], limit: 10,
                                                  into: [KVCacheSimple(), MambaCache()])
        XCTAssertEqual(kept, 10)
        XCTAssertNil(dropped)
    }

    // MARK: - Eviction

    func testEvictReportsHowManyEntriesItDropped() async {
        let pc = PromptCache(maxEntries: 3)
        let emptyKeep = await pc.evict(keepMostRecent: true)
        let emptyAll = await pc.evict(keepMostRecent: false)
        XCTAssertEqual(emptyKeep, 0)
        XCTAssertEqual(emptyAll, 0)

        await put(pc, [1, 1, 1, 1])
        await put(pc, [2, 2, 2, 2])
        await put(pc, [3, 3, 3, 3])
        let keepRecent = await pc.evict(keepMostRecent: true)
        let again = await pc.evict(keepMostRecent: true)
        let last = await pc.evict(keepMostRecent: false)
        let none = await pc.evict(keepMostRecent: false)
        XCTAssertEqual(keepRecent, 2)
        XCTAssertEqual(again, 0, "nothing left to drop but the most recent")
        XCTAssertEqual(last, 1)
        XCTAssertEqual(none, 0)
    }

    func testCapacityBelowOneIsRaisedToOne() async {
        for requested in [0, -1] {
            let pc = PromptCache(maxEntries: requested)
            await put(pc, [1, 2, 3, 4])
            await put(pc, [9, 8, 7, 6])
            let old = await hit(pc, [1, 2, 3, 4, 5])
            let recent = await hit(pc, [9, 8, 7, 6, 5])
            XCTAssertNil(old, "maxEntries \(requested) still holds one entry, not none or many")
            XCTAssertEqual(recent, 4)
        }
    }

    func testFlagParses() throws {
        XCTAssertEqual(try MLXServer.parse(["--model", "m"]).promptCacheEntries, 1)
        XCTAssertEqual(try MLXServer.parse(["--model", "m", "--prompt-cache-entries", "4"]).promptCacheEntries, 4)
    }
}
