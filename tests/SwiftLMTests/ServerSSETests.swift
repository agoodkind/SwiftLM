import XCTest
import Foundation
@testable import SwiftLM

final class ServerSSETests: XCTestCase {

    // MARK: - Truthy header parser

    func testParseTruthyHeaderValue() {
        XCTAssertTrue(parseTruthyHeaderValue("true"))
        XCTAssertTrue(parseTruthyHeaderValue("TRUE"))
        XCTAssertTrue(parseTruthyHeaderValue(" yes "))
        XCTAssertTrue(parseTruthyHeaderValue("1"))
        XCTAssertFalse(parseTruthyHeaderValue(nil))
        XCTAssertFalse(parseTruthyHeaderValue("false"))
        XCTAssertFalse(parseTruthyHeaderValue("0"))
    }

    // MARK: - 1a: "on" is a documented truthy alias (HTML-form / reverse-proxy parity)

    func testParseTruthyHeaderValue_OnAlias() {
        // "on" is intentionally accepted for parity with common reverse-proxy conventions.
        // See ssePrefillChunk doc comment for the rationale.
        XCTAssertTrue(parseTruthyHeaderValue("on"))
        XCTAssertTrue(parseTruthyHeaderValue("ON"))
    }

    // MARK: - Named event + lean payload (existing test, Fix #4 applied)

    func testPrefillChunkUsesNamedEventAndLeanPayload() throws {
        let chunk = ssePrefillChunk(nPast: 32, promptTokens: 128, elapsedSeconds: 4)

        let prefix = "event: prefill_progress\r\ndata: "
        let suffix = "\r\n\r\n"
        XCTAssertTrue(chunk.hasPrefix(prefix))
        XCTAssertTrue(chunk.hasSuffix(suffix))

        // Fix #4: use suffix.count not the literal 4, so multi-byte chars at boundary
        // don't silently corrupt the JSON slice.
        let payload = String(chunk.dropFirst(prefix.count).dropLast(suffix.count))
        let data = try XCTUnwrap(payload.data(using: .utf8))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(json["status"] as? String, "processing")
        XCTAssertEqual(json["n_past"] as? Int, 32)
        XCTAssertEqual(json["n_prompt_tokens"] as? Int, 128)
        XCTAssertEqual(json["elapsed_seconds"] as? Int, 4)
        XCTAssertNil(json["object"])
        // Issue #168: empty choices keeps strict OpenAI chunk validators (opencode) happy
        // even when they validate every data: line regardless of event name.
        XCTAssertEqual((json["choices"] as? [Any])?.count, 0)
    }

    // MARK: - 1b: Zero-token boundary (no divide-by-zero crash)

    func testPrefillChunk_ZeroTokenBoundary() throws {
        let chunk = ssePrefillChunk(nPast: 0, promptTokens: 0, elapsedSeconds: 0)
        let prefix = "event: prefill_progress\r\ndata: "
        let suffix = "\r\n\r\n"
        let payload = String(chunk.dropFirst(prefix.count).dropLast(suffix.count))
        let data = try XCTUnwrap(payload.data(using: .utf8))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        let fraction = try XCTUnwrap(json["fraction"] as? Double)
        XCTAssertEqual(fraction, 0.0, accuracy: 1e-9, "Division by zero must yield 0.0")
        XCTAssertFalse(fraction.isNaN, "fraction must not be NaN")
        XCTAssertFalse(fraction.isInfinite, "fraction must not be infinite")
    }

    // MARK: - 1c: dropLast correctness regression guard

    func testPrefillChunk_DropLastSafe() throws {
        // Confirms the suffix-count trim extracts parseable JSON for any content length.
        let chunk = ssePrefillChunk(nPast: 100, promptTokens: 400, elapsedSeconds: 6)
        let prefix = "event: prefill_progress\r\ndata: "
        let suffix = "\r\n\r\n"
        XCTAssertTrue(chunk.hasSuffix(suffix), "SSE terminator must be \\r\\n\\r\\n")
        let trimmed = String(chunk.dropFirst(prefix.count).dropLast(suffix.count))
        let data = try XCTUnwrap(trimmed.data(using: .utf8))
        // Must parse — would crash if dropLast sliced inside a multi-byte char
        XCTAssertNoThrow(try JSONSerialization.jsonObject(with: data))
    }

    // MARK: - 1d: No OpenAI-specific fields bleed into prefill payload

    func testPrefillChunk_NoOpenAIFields() throws {
        let chunk = ssePrefillChunk(nPast: 1, promptTokens: 4, elapsedSeconds: 1)
        let prefix = "event: prefill_progress\r\ndata: "
        let suffix = "\r\n\r\n"
        let payload = String(chunk.dropFirst(prefix.count).dropLast(suffix.count))
        let data = try XCTUnwrap(payload.data(using: .utf8))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        // Fields that would confuse strict OpenAI-SDK clients (e.g. OpenCode) must be absent
        XCTAssertNil(json["id"],      "prefill chunk must not carry an id field")
        XCTAssertNil(json["object"],  "prefill chunk must not carry an object field")
        XCTAssertNil(json["model"],   "prefill chunk must not carry a model field")
        // Issue #168: choices must be present but empty — opencode's chunk union
        // requires `choices` (or `error`); omitting it fails type validation.
        XCTAssertEqual((json["choices"] as? [Any])?.count, 0,
                       "prefill chunk must carry an empty choices array")
    }

    // MARK: - Issue #168: empty choices is what strict validators require

    func testPrefillChunk_ChoicesIsEmptyArrayForStrictValidators() throws {
        let chunk = ssePrefillChunk(nPast: 1, promptTokens: 4, elapsedSeconds: 1)
        let prefix = "event: prefill_progress\r\ndata: "
        let suffix = "\r\n\r\n"
        let payload = String(chunk.dropFirst(prefix.count).dropLast(suffix.count))
        let data = try XCTUnwrap(payload.data(using: .utf8))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        let choices = try XCTUnwrap(json["choices"] as? [Any], "choices must be present")
        XCTAssertTrue(choices.isEmpty)
    }

    // MARK: - 1e: PrefillState.finish() is idempotent (Issue #2 guard)

    func testPrefillState_FinishIsIdempotent() async {
        let state = PrefillState()
        await state.finish()
        await state.finish()  // second call must not throw or reset done
        let done = await state.done
        XCTAssertTrue(done, "PrefillState.done must remain true after double finish()")
    }

    // MARK: - 1f: PrefillState contract: update after finish (Issue #2 guard)

    func testPrefillState_UpdateAfterFinishContract() async {
        let state = PrefillState()
        await state.update(nPast: 50)
        await state.finish()
        await state.update(nPast: 999)  // post-done update
        let done = await state.done
        // Invariant: done must stay true — the heartbeat loop guards on this
        XCTAssertTrue(done, "PrefillState.done must remain true after post-finish update")
        // The heartbeat loop reads nPast only when !done, so its value after finish
        // is irrelevant to correctness. We capture the current contract here.
        // If a post-done guard is added later, add XCTAssertNotEqual(await state.nPast, 999).
    }

    // MARK: - Error event stays valid JSON for any message text

    private struct MessyError: Error, CustomStringConvertible {
        let description = "bad \"quote\", back\\slash\nnew line"
    }

    func testErrorChunkEncodesMessageAsValidJSON() throws {
        let chunk = sseErrorChunk(MessyError())

        let prefix = "data: "
        let suffix = "\r\n\r\n"
        XCTAssertTrue(chunk.hasPrefix(prefix))
        XCTAssertTrue(chunk.hasSuffix(suffix))

        let payload = String(chunk.dropFirst(prefix.count).dropLast(suffix.count))
        let obj = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any])
        let err = try XCTUnwrap(obj["error"] as? [String: Any])
        XCTAssertEqual(err["message"] as? String, MessyError().description)
        XCTAssertEqual(err["type"] as? String, "server_error")
        XCTAssertEqual(err["code"] as? String, "internal_error")
    }

    func testErrorJSONEncodesMessageAsValidJSON() throws {
        let body = errorJSON(MessyError())

        let obj = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any])
        let err = try XCTUnwrap(obj["error"] as? [String: Any])
        XCTAssertEqual(err["message"] as? String, MessyError().description)
        XCTAssertEqual(err["type"] as? String, "server_error")
        XCTAssertEqual(err["code"] as? String, "internal_error")
    }

    // MARK: - Keepalive

    /// A silent stream (long prefill, buffered tool call) must still carry bytes, as
    /// SSE comments that parsers ignore, or Node/Bun clients abort it after ~300 s.
    func testKeepaliveEmitsSSECommentsWhileStreamIsSilent() async {
        let (stream, cont) = AsyncStream<String>.makeStream()
        let task = startSSEKeepalive(cont, interval: .milliseconds(20))
        var received: [String] = []
        for await event in stream {
            received.append(event)
            if received.count == 2 { break }
        }
        task.cancel()
        XCTAssertEqual(received, [": keepalive\r\n\r\n", ": keepalive\r\n\r\n"])
    }

    func testKeepaliveStopsWhenStreamFinishes() async {
        let (_, cont) = AsyncStream<String>.makeStream()
        let task = startSSEKeepalive(cont, interval: .milliseconds(10))
        cont.finish()
        // Returns on its own (yield reports .terminated), without being cancelled.
        await task.value
    }
}
