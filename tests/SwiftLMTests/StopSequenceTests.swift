import XCTest
import Foundation
@testable import SwiftLM

// MARK: - Contract tests for the `stop` parameter (issue #126)
//
// The stop sequence must never reach the client. These pin the parts of that contract
// that are testable without a model; the streaming-boundary half is covered by
// tests/test-contract.sh against a live server.
final class StopSequenceTests: XCTestCase {

    func testTrimsAtTheStopSequence() {
        let result = checkStopSequences("answer<|end|>trailing", stopSequences: ["<|end|>"])
        XCTAssertEqual(result?.0, "answer")
        XCTAssertEqual(result?.1, "<|end|>")
    }

    /// Earliest *in the text*, not first in the caller's list. Returning whichever entry
    /// was listed first kept everything between the real stop and that one.
    func testPicksEarliestMatchNotFirstListed() {
        let result = checkStopSequences("abXc\nUser:", stopSequences: ["\nUser:", "X"])
        XCTAssertEqual(result?.0, "ab", "must stop at X, the earliest match in the text")
        XCTAssertEqual(result?.1, "X")
    }

    func testOrderOfStopListDoesNotMatter() {
        let a = checkStopSequences("one TWO three", stopSequences: ["TWO", "three"])
        let b = checkStopSequences("one TWO three", stopSequences: ["three", "TWO"])
        XCTAssertEqual(a?.0, b?.0)
        XCTAssertEqual(a?.0, "one ")
    }

    func testNoMatchReturnsNil() {
        XCTAssertNil(checkStopSequences("nothing here", stopSequences: ["<|end|>"]))
        XCTAssertNil(checkStopSequences("anything", stopSequences: []))
    }

    /// An empty stop string matches at index 0 and would truncate every response.
    func testEmptyStopSequenceIsIgnored() {
        XCTAssertNil(checkStopSequences("real content", stopSequences: [""]))
        let result = checkStopSequences("real<|end|>", stopSequences: ["", "<|end|>"])
        XCTAssertEqual(result?.0, "real", "an empty entry must not shadow a real one")
    }

    func testStopAtStartYieldsEmptyContent() {
        let result = checkStopSequences("<|end|>everything", stopSequences: ["<|end|>"])
        XCTAssertEqual(result?.0, "")
    }

    /// Multi-byte content must not be split mid-character.
    func testUnicodeContentIsTrimmedCleanly() {
        let result = checkStopSequences("日本の首都は東京です<|end|>", stopSequences: ["<|end|>"])
        XCTAssertEqual(result?.0, "日本の首都は東京です")
    }

    // MARK: Bounded streaming scan

    /// Feeds `chunks` the way the streaming handlers do, scanning every `scanEvery` chunks.
    /// `bounded: false` is the old full rescan, the oracle the bounded scan must match.
    private func streamScan(
        _ chunks: [String], stops: [String], bounded: Bool, scanEvery: Int = 1
    ) -> (String, String)? {
        var full = ""
        var unscanned = 0
        for (i, chunk) in chunks.enumerated() {
            full += chunk
            unscanned += chunk.count
            guard (i + 1) % scanEvery == 0 || i == chunks.count - 1 else { continue }
            let lookback = bounded ? stopScanWindow(stops) + unscanned : nil
            unscanned = 0
            if let hit = checkStopSequences(full, stopSequences: stops, lookback: lookback) {
                return hit
            }
        }
        return nil
    }

    /// Splits `text` into chunks whose sizes cycle through `sizes`.
    private func chunked(_ text: String, sizes: [Int]) -> [String] {
        var chunks: [String] = []
        var rest = Substring(text)
        var i = 0
        while !rest.isEmpty {
            let n = sizes[i % sizes.count]
            chunks.append(String(rest.prefix(n)))
            rest = rest.dropFirst(n)
            i += 1
        }
        return chunks
    }

    func testBoundedScanFindsStopStraddlingChunks() {
        let result = streamScan(["answer\nUs", "er: next"], stops: ["\nUser:"], bounded: true)
        XCTAssertEqual(result?.0, "answer")
        XCTAssertEqual(result?.1, "\nUser:")
    }

    /// JSON mode skips the scan for its buffered chunks; one later scan must still see them.
    func testBoundedScanCoversChunksSkippedSinceLastScan() {
        let result = streamScan(
            ["{\"a\": 1}", "<|end|>", "tail", "more"], stops: ["<|end|>"], bounded: true, scanEvery: 4)
        XCTAssertEqual(result?.0, "{\"a\": 1}")
    }

    func testBoundedScanMatchesFullRescanForEveryChunking() {
        let cases: [(String, [String])] = [
            ("plain text then <|im_end|> and after", ["<|im_end|>", "<end_of_turn>"]),
            ("abXc\nUser: trailing", ["\nUser:", "X"]),
            ("ab<|eot_id|>cd<turn|>", ["<turn|>", "<|eot_id|>"]),
            ("日本の首都は東京です🎌<|end|>🎌", ["<|end|>"]),
            ("cafe\u{301} STOP", ["STOP"]),
            ("👨‍👩‍👧 family STOP", ["STOP", "family"]),
            ("no stop anywhere in this response", ["<|end|>", "\n\n"]),
            ("overlapping abcd", ["abcd", "bc"]),
        ]
        let chunkings: [[Int]] = [[1], [2], [3], [5], [1, 4, 2], [7, 1], [100]]
        for (text, stops) in cases {
            for sizes in chunkings {
                let chunks = chunked(text, sizes: sizes)
                let bounded = streamScan(chunks, stops: stops, bounded: true)
                let full = streamScan(chunks, stops: stops, bounded: false)
                XCTAssertEqual(bounded?.0, full?.0, "\(text) chunked \(sizes)")
                XCTAssertEqual(bounded?.1, full?.1, "\(text) chunked \(sizes)")
            }
        }
    }

    /// A combining mark opening a chunk merges into the previous chunk's last character.
    func testBoundedScanSurvivesGraphemeMergeAcrossChunks() {
        let chunks = ["cafe", "\u{301}ST", "OP"]
        let bounded = streamScan(chunks, stops: ["STOP"], bounded: true)
        XCTAssertEqual(bounded?.0, "caf\u{E9}")
        XCTAssertEqual(bounded?.0, streamScan(chunks, stops: ["STOP"], bounded: false)?.0)
    }

    func testLookbackLongerThanTextScansAll() {
        let result = checkStopSequences("x<|end|>", stopSequences: ["<|end|>"], lookback: 1000)
        XCTAssertEqual(result?.0, "x")
    }
}
