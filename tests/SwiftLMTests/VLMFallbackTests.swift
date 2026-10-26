import XCTest
import Foundation
import MLXLMCommon
import MLXNN
@testable import SwiftLM

/// An auto-detected VLM that fails to load falls back to text-only only for
/// checkpoint mismatches. Other failures must surface as themselves.
final class VLMFallbackTests: XCTestCase {

    private struct Probe: Decodable { let image_mean: [Double] }

    func testConfigDecodingErrorFallsBack() {
        // The real case: unsloth/Qwen3.6-35B-A3B's preprocessor_config.json has no image_mean.
        do {
            _ = try JSONDecoder().decode(Probe.self, from: Data("{}".utf8))
            XCTFail("expected a DecodingError")
        } catch {
            XCTAssertTrue(isVLMCheckpointMismatch(error))
        }
    }

    func testWeightMismatchFallsBack() {
        let error = UpdateError.unhandledKeys(path: [], modules: ["Vision"], keys: ["pre_projection"])
        XCTAssertTrue(isVLMCheckpointMismatch(error))
    }

    func testUnsupportedModelTypeFallsBack() {
        XCTAssertTrue(isVLMCheckpointMismatch(ModelFactoryError.unsupportedModelType("qwen4_exp")))
    }

    func testCancellationDoesNotFallBack() {
        XCTAssertFalse(isVLMCheckpointMismatch(CancellationError()))
    }

    func testNetworkErrorDoesNotFallBack() {
        XCTAssertFalse(isVLMCheckpointMismatch(URLError(.notConnectedToInternet)))
    }

    func testMissingConfigFileDoesNotFallBack() {
        let io = CocoaError(.fileReadNoSuchFile)
        XCTAssertFalse(isVLMCheckpointMismatch(ModelFactoryError.configurationFileError("config.json", "m", io)))
    }
}
