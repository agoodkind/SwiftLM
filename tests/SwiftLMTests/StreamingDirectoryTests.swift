import Foundation
import XCTest

@testable import SwiftLM

/// `localWeightState` must judge a copy the way the loader reads it, so a partial
/// download is never served and a complete copy with a stale index is not rejected.
final class StreamingDirectoryTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("streaming-dir-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try write("config.json")
        try write("tokenizer.json")
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private func write(_ name: String, _ text: String = "{}") throws {
        try Data(text.utf8).write(to: dir.appendingPathComponent(name))
    }

    private func shards(_ indices: [Int], of count: Int) throws {
        for i in indices {
            try write(String(format: "model-%05d-of-%05d.safetensors", i, count), "weights")
        }
    }

    /// An index naming shards the repository doesn't ship (carried over from the source).
    private func staleIndex(naming count: Int) throws {
        let map = Dictionary(uniqueKeysWithValues: (1 ... count).map {
            ("layer\($0).weight", String(format: "model-%05d-of-%05d.safetensors", $0, count))
        })
        let json = try JSONSerialization.data(withJSONObject: ["weight_map": map])
        try json.write(to: dir.appendingPathComponent("model.safetensors.index.json"))
    }

    func testPartialCopyWithoutIndexIsIncomplete() throws {
        try shards([1, 3], of: 3)
        XCTAssertEqual(localWeightState(in: dir), .incomplete)
    }

    func testAllShardsWithoutIndexAreComplete() throws {
        try shards([1, 2, 3], of: 3)
        XCTAssertEqual(localWeightState(in: dir), .complete)
    }

    func testStaleIndexWithEveryShippedShardIsComplete() throws {
        try staleIndex(naming: 13)
        try shards([1, 2, 3, 4], of: 4)
        XCTAssertEqual(localWeightState(in: dir), .complete)
    }

    func testStaleIndexWithAMissingShardIsIncomplete() throws {
        try staleIndex(naming: 13)
        try shards([1, 2, 4], of: 4)
        XCTAssertEqual(localWeightState(in: dir), .incomplete)
    }

    /// OptiQ VLMs index their vision tower in optiq/; a download of only model*.safetensors
    /// has every top-level shard but not that file.
    private func indexWithNestedVision() throws {
        var map = Dictionary(uniqueKeysWithValues: (1 ... 2).map {
            ("layer\($0).weight", String(format: "model-%05d-of-%05d.safetensors", $0, 2))
        })
        map["vision_tower.weight"] = "optiq/optiq_vision.safetensors"
        let json = try JSONSerialization.data(withJSONObject: ["weight_map": map])
        try json.write(to: dir.appendingPathComponent("model.safetensors.index.json"))
    }

    func testIndexedFileMissingFromSubdirectoryIsIncomplete() throws {
        try indexWithNestedVision()
        try shards([1, 2], of: 2)
        XCTAssertEqual(localWeightState(in: dir), .incomplete)
    }

    func testIndexedSubdirectoryFilePresentIsComplete() throws {
        try indexWithNestedVision()
        try shards([1, 2], of: 2)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("optiq"), withIntermediateDirectories: true)
        try write("optiq/optiq_vision.safetensors", "weights")
        XCTAssertEqual(localWeightState(in: dir), .complete)
    }

    func testRequiredListedFilesIncludeIndexedNestedFiles() throws {
        try indexWithNestedVision()
        let listed = ["config.json", "model-00001-of-00002.safetensors",
                      "optiq/optiq_vision.safetensors", "optiq/mtp.safetensors"]
        XCTAssertEqual(
            Set(requiredListedFiles(listed, in: dir)),
            ["config.json", "model-00001-of-00002.safetensors", "optiq/optiq_vision.safetensors"])
    }

    func testSingleModelFileIsComplete() throws {
        try write("model.safetensors", "weights")
        XCTAssertEqual(localWeightState(in: dir), .complete)
    }

    func testUnconventionalLayoutIsUnverified() throws {
        try write("weights.00.safetensors", "weights")
        XCTAssertEqual(localWeightState(in: dir), .unverified)
    }

    func testDanglingSymlinkIsMissing() throws {
        try FileManager.default.createSymbolicLink(
            at: dir.appendingPathComponent("model.safetensors"),
            withDestinationURL: dir.appendingPathComponent("gone.bin"))
        XCTAssertFalse(isPresentFile("model.safetensors", in: dir))
        XCTAssertEqual(localWeightState(in: dir), .incomplete)
    }

    func testTokenizerJSONIsRequired() throws {
        XCTAssertTrue(hasModelConfigAndTokenizer(in: dir))
        try FileManager.default.removeItem(at: dir.appendingPathComponent("tokenizer.json"))
        try write("tokenizer_config.json")
        XCTAssertFalse(hasModelConfigAndTokenizer(in: dir))
    }
}
