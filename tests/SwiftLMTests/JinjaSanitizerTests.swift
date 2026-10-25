import XCTest
import Foundation
@testable import SwiftLM

/// Issue #168: JSON `null` (NSNull) in tool schemas used to abort Jinja.Value conversion
/// with a misleading "Optional<Any>" error, mislabeled as a broken chat template.
final class JinjaSanitizerTests: XCTestCase {

    func testDropsTopLevelNSNull() {
        XCTAssertNil(sanitizeForJinja(NSNull()))
    }

    func testDropsNestedObjectNulls() throws {
        let tool: [String: any Sendable] = [
            "type": "function",
            "function": [
                "name": "probe",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "query": [
                            "type": "string",
                            "default": NSNull() as any Sendable,
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]

        let cleaned = try XCTUnwrap(sanitizeForJinja(tool) as? [String: any Sendable])
        let fn = try XCTUnwrap(cleaned["function"] as? [String: any Sendable])
        let params = try XCTUnwrap(fn["parameters"] as? [String: any Sendable])
        let props = try XCTUnwrap(params["properties"] as? [String: any Sendable])
        let query = try XCTUnwrap(props["query"] as? [String: any Sendable])

        XCTAssertNil(query["default"], "null default must be stripped")
        XCTAssertEqual(query["type"] as? String, "string")
        XCTAssertEqual(params["type"] as? String, "object")
    }

    func testDropsNullArrayElements() throws {
        let value: [String: any Sendable] = [
            "enum": [1, NSNull(), 2] as [any Sendable]
        ]
        let cleaned = try XCTUnwrap(sanitizeForJinja(value) as? [String: any Sendable])
        let enumValues = try XCTUnwrap(cleaned["enum"] as? [Any])
        XCTAssertEqual(enumValues.count, 2)
        XCTAssertFalse(enumValues.contains { $0 is NSNull })
    }

    func testPreservesScalarsAndStructure() throws {
        let value: [String: any Sendable] = [
            "type": "string",
            "minimum": 0,
            "required": ["command"] as [any Sendable],
            "flag": true,
        ]
        let cleaned = try XCTUnwrap(sanitizeForJinja(value) as? [String: any Sendable])
        XCTAssertEqual(cleaned["type"] as? String, "string")
        XCTAssertEqual(cleaned["minimum"] as? Int, 0)
        XCTAssertEqual(cleaned["flag"] as? Bool, true)
        XCTAssertEqual((cleaned["required"] as? [Any])?.count, 1)
    }

    func testUnwrapsNestedOptional() throws {
        let wrapped: any Sendable = Optional<String>.some("hi")
        let cleaned = sanitizeForJinja(wrapped)
        XCTAssertEqual(cleaned as? String, "hi")

        let empty: any Sendable = Optional<String>.none
        XCTAssertNil(sanitizeForJinja(empty))
    }

    func testMapValuesDeepOnToolDict() throws {
        let dict: [String: any Sendable] = [
            "keep": "x",
            "drop": NSNull(),
        ]
        let cleaned = dict.mapValuesDeep(sanitizeForJinja)
        XCTAssertEqual(cleaned["keep"] as? String, "x")
        XCTAssertNil(cleaned["drop"])
    }
}
