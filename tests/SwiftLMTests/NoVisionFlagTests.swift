import XCTest
@testable import SwiftLM

/// `--no-vision` loads a vision-capable checkpoint as a text-only LLM, so text-only
/// workloads get the prompt cache (skipped for every VLM-loaded model).
final class NoVisionFlagTests: XCTestCase {

    func testNoVisionParses() throws {
        XCTAssertTrue(try MLXServer.parse(["--model", "m", "--no-vision"]).noVision)
        XCTAssertFalse(try MLXServer.parse(["--model", "m"]).noVision)
    }

    func testVisionAndNoVisionAreMutuallyExclusive() {
        XCTAssertThrowsError(try MLXServer.parse(["--model", "m", "--vision", "--no-vision"]))
    }
}
