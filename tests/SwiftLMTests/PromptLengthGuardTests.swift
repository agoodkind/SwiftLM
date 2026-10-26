import XCTest
import Foundation
@testable import SwiftLM

/// `--max-prompt-tokens` rejects over-long prompts before prefill, so a prompt the
/// machine can't hold fails fast with a 400 instead of driving the server into swap.
final class PromptLengthGuardTests: XCTestCase {

    func testNoLimitAllowsAnything() {
        XCTAssertNil(promptTooLongBody(promptTokens: 1_000_000, limit: nil))
    }

    func testPromptAtTheLimitIsAllowed() {
        XCTAssertNil(promptTooLongBody(promptTokens: 32_768, limit: 32_768))
    }

    func testPromptOverTheLimitIsRejected() {
        XCTAssertNotNil(promptTooLongBody(promptTokens: 32_769, limit: 32_768))
    }

    /// Agent clients key on OpenAI's code and wording to decide to compact.
    func testRejectionIsOpenAIShaped() throws {
        let body = try XCTUnwrap(promptTooLongBody(promptTokens: 34_412, limit: 32_768))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any])
        let error = try XCTUnwrap(json["error"] as? [String: Any])
        XCTAssertEqual(error["code"] as? String, "context_length_exceeded")
        XCTAssertEqual(error["type"] as? String, "invalid_request_error")
        let message = try XCTUnwrap(error["message"] as? String)
        XCTAssertTrue(message.contains("maximum context length is 32768 tokens"), message)
        XCTAssertTrue(message.contains("34412 tokens"), message)
    }

    func testFlagParses() throws {
        let server = try MLXServer.parse(["--model", "m", "--max-prompt-tokens", "20000"])
        XCTAssertEqual(server.maxPromptTokens, 20_000)
        XCTAssertNil(try MLXServer.parse(["--model", "m"]).maxPromptTokens)
    }
}
