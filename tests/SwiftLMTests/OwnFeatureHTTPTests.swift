import Darwin
import Foundation
import MLX
import MLXLMCommon
import Tokenizers
import XCTest

final class OwnFeatureHTTPTests: XCTestCase {
  private var previousMemoryLimit = 0
  private var previousCacheLimit = 0

  override func setUp() {
    super.setUp()
    previousMemoryLimit = Memory.memoryLimit
    previousCacheLimit = Memory.cacheLimit
    Memory.memoryLimit = 64 * 1024 * 1024
    Memory.cacheLimit = 0
  }

  override func tearDown() {
    Stream.defaultStream.synchronize()
    Memory.clearCache()
    Memory.cacheLimit = previousCacheLimit
    Memory.memoryLimit = previousMemoryLimit
    super.tearDown()
  }

  func testPartitionedLayerCallSelectsDevicesAndRestoresGPUContext() {
    XCTAssertFalse(ExpertStreamingConfig.shared.isEnabled)
    let originalDevice = Device.defaultDevice().deviceType
    Device.withDefaultDevice(.gpu) {
      XCTAssertEqual(Device.defaultDevice().deviceType, .gpu)
      let cpuResult = partitionedLayerCall(index: 1, gpuLayerCount: 1) {
        XCTAssertEqual(Device.defaultDevice().deviceType, .cpu)
        let result = MLXArray([Float(2), 3]) * 2
        eval(result)
        return result
      }
      XCTAssertEqual(cpuResult.asArray(Float.self), [4, 6])
      XCTAssertEqual(Device.defaultDevice().deviceType, .gpu)

      let gpuResult = partitionedLayerCall(index: 0, gpuLayerCount: 1) {
        XCTAssertEqual(Device.defaultDevice().deviceType, .gpu)
        let result = MLXArray([Float(4), 5]) + 1
        eval(result)
        return result
      }
      XCTAssertEqual(gpuResult.asArray(Float.self), [5, 6])
      XCTAssertEqual(Device.defaultDevice().deviceType, .gpu)
    }
    XCTAssertEqual(Device.defaultDevice().deviceType, originalDevice)
  }

  func testDefaultMemoryLimitUsesEightyPercentOfPhysicalRAM() async throws {
    let checkpoint = try await makeCheckpoint()
    let server = try await startServer(checkpoint: checkpoint.directory, memoryLimitMB: nil)
    defer { server.stop() }

    let health = try await server.health()
    let expected = Int(Double(ProcessInfo.processInfo.physicalMemory) * 0.8) / (1024 * 1024)
    XCTAssertEqual(health.memory.limitMB, expected)
    XCTAssertLessThan(health.memory.peakMB, 64)
  }

  func testCompletedNonStreamingRequestsReusePromptCache() async throws {
    try await assertCompletedRequestsReusePromptCache(streaming: false)
  }

  func testCompletedStreamingRequestsReusePromptCache() async throws {
    try await assertCompletedRequestsReusePromptCache(streaming: true)
  }

  private func assertCompletedRequestsReusePromptCache(streaming: Bool) async throws {
    let checkpoint = try await makeCheckpoint()
    let logURL: URL
    do {
      let server = try await startServer(checkpoint: checkpoint.directory, memoryLimitMB: 64)
      defer { server.stop() }
      let first = try await server.chat(streaming: streaming, logprobs: false)
      let second = try await server.chat(streaming: streaming, logprobs: false)
      XCTAssertEqual(try responseText(first, streaming: streaming), "AAA")
      XCTAssertEqual(try responseText(second, streaming: streaming), "AAA")
      let health = try await server.health()
      XCTAssertLessThan(health.memory.peakMB, 64)
      logURL = server.logURL
    }
    let output = try String(contentsOf: logURL, encoding: .utf8)
    let pattern = try NSRegularExpression(
      pattern: "Prompt cache HIT: ([0-9]+)/([0-9]+) tokens reused")
    let matches = pattern.matches(in: output, range: NSRange(output.startIndex..., in: output))
    XCTAssertEqual(matches.count, 1, "The second request must reuse its completed prompt: \(output)")
    let match = try XCTUnwrap(matches.first)
    let reusedRange = try XCTUnwrap(Range(match.range(at: 1), in: output))
    let promptRange = try XCTUnwrap(Range(match.range(at: 2), in: output))
    let reused = try XCTUnwrap(Int(output[reusedRange]))
    let prompt = try XCTUnwrap(Int(output[promptRange]))
    XCTAssertGreaterThan(reused, 0)
    XCTAssertEqual(reused, prompt)
  }

  private func responseText(_ response: Data, streaming: Bool) throws -> String {
    if !streaming {
      let reply = try JSONDecoder().decode(ChatReply.self, from: response)
      return try XCTUnwrap(reply.choices.first?.message?.content)
    }
    let body = String(decoding: response, as: UTF8.self)
    var text = ""
    var completed = false
    for raw in body.components(separatedBy: "\n") {
      let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
      guard line.hasPrefix("data: ") else { continue }
      let payload = String(line.dropFirst(6))
      if payload == "[DONE]" {
        completed = true
        continue
      }
      let event = try JSONDecoder().decode(ChatReply.self, from: Data(payload.utf8))
      text += event.choices.compactMap { $0.delta?.content }.joined()
    }
    XCTAssertTrue(completed, "The streaming response must finish: \(body)")
    return text
  }

  func testExplicitMemoryLimitAndHTTPLogprobsUseActualCheckpoint() async throws {
    let checkpoint = try await makeCheckpoint()
    let server = try await startServer(checkpoint: checkpoint.directory, memoryLimitMB: 64)
    defer { server.stop() }
    let initialHealth = try await server.health()
    XCTAssertEqual(initialHealth.memory.limitMB, 64)

    let withoutLogprobs = try await server.chat(streaming: false, logprobs: false)
    let omitted = try JSONDecoder().decode(ChatReply.self, from: withoutLogprobs)
    XCTAssertNil(try XCTUnwrap(omitted.choices.first).logprobs)

    let response = try await server.chat(streaming: false, logprobs: true)
    let reply = try JSONDecoder().decode(ChatReply.self, from: response)
    let choice = try XCTUnwrap(reply.choices.first)
    let entries = try XCTUnwrap(choice.logprobs).content
    XCTAssertEqual(choice.message?.content, "AAA")
    try assertLogprobs(entries, expected: checkpoint.logProbability)

    let streamed = try await server.chat(streaming: true, logprobs: true)
    let body = String(decoding: streamed, as: UTF8.self)
    var streamedEntries = [ProbabilityEntry]()
    var text = ""
    var completed = false
    for raw in body.components(separatedBy: "\n") {
      let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
      guard line.hasPrefix("data: ") else { continue }
      let payload = String(line.dropFirst(6))
      if payload == "[DONE]" {
        completed = true
        continue
      }
      let event = try JSONDecoder().decode(ChatReply.self, from: Data(payload.utf8))
      for eventChoice in event.choices {
        text += eventChoice.delta?.content ?? ""
        streamedEntries += eventChoice.logprobs?.content ?? []
      }
    }
    XCTAssertTrue(completed, "The SSE response did not include [DONE]: \(body)")
    XCTAssertEqual(text, "AAA")
    try assertLogprobs(streamedEntries, expected: checkpoint.logProbability)
    for (entry, streamedEntry) in zip(entries, streamedEntries) {
      XCTAssertEqual(entry.logprob, streamedEntry.logprob, accuracy: 0.0001)
    }
    let health = try await server.health()
    XCTAssertEqual(health.memory.limitMB, 64)
    XCTAssertLessThan(health.memory.peakMB, 64)
  }

  private func assertLogprobs(_ entries: [ProbabilityEntry], expected: Double) throws {
    XCTAssertEqual(entries.count, 3)
    for entry in entries {
      XCTAssertEqual(entry.token, "A")
      XCTAssertEqual(entry.bytes, [65])
      XCTAssertTrue(entry.topLogprobs.isEmpty)
      XCTAssertTrue(entry.logprob.isFinite)
      XCTAssertEqual(entry.logprob, expected, accuracy: 0.0001)
    }
  }

  private var repository: URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
  }

  private func makeCheckpoint() async throws -> (directory: URL, logProbability: Double) {
    let source = repository.appendingPathComponent("tests/fixtures/dense")
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("swiftlm-http-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    addTeardownBlock { try FileManager.default.removeItem(at: directory) }
    for file in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
      try FileManager.default.copyItem(
        at: source.appendingPathComponent(file), to: directory.appendingPathComponent(file))
    }
    let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
    let tokenA = try XCTUnwrap(tokenizer.convertTokenToId("A"))
    let tokenB = try XCTUnwrap(tokenizer.convertTokenToId("B"))
    let original = try MLX.loadArrays(url: source.appendingPathComponent("model.safetensors"))
    var weights = [String: MLXArray]()
    for (name, tensor) in original {
      if name == "model.embed_tokens.weight" || name.hasSuffix("norm.weight") {
        weights[name] = MLXArray.ones(tensor.shape)
      } else {
        weights[name] = MLXArray.zeros(tensor.shape)
      }
    }
    let head = try XCTUnwrap(weights["lm_head.weight"])
    head[tokenA] = MLXArray.ones([head.dim(1)]) / Float(head.dim(1))
    head[tokenB] = MLXArray.ones([head.dim(1)]) / Float(head.dim(1) * 2)
    try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
    let expected = 1 - log(Double(head.dim(0) - 2) + exp(1) + exp(0.5))
    return (directory, expected)
  }

  private func startServer(checkpoint: URL, memoryLimitMB: Int?) async throws -> TinyServer {
    let binaryDirectory: URL
    if let configured = ProcessInfo.processInfo.environment["SWIFTLM_BIN_DIR"] {
      binaryDirectory = URL(
        fileURLWithPath: configured, isDirectory: true, relativeTo: repository)
        .standardizedFileURL
    } else {
      binaryDirectory = Bundle(for: OwnFeatureHTTPTests.self).bundleURL
        .deletingLastPathComponent()
    }
    let executable = binaryDirectory.appendingPathComponent("SwiftLM")
    guard FileManager.default.isExecutableFile(atPath: executable.path) else {
      throw ServerTestError.missingExecutable(
        "Build SwiftLM beside the test bundle or set SWIFTLM_BIN_DIR. Missing: \(executable.path)")
    }
    let logURL = checkpoint.appendingPathComponent("server.log")
    guard FileManager.default.createFile(atPath: logURL.path, contents: nil) else {
      throw CocoaError(.fileWriteUnknown)
    }
    let output = try FileHandle(forWritingTo: logURL)
    defer { try? output.close() }
    let process = Process()
    process.executableURL = executable
    process.currentDirectoryURL = repository
    let port = try reservePort()
    process.arguments = [
      "--model", checkpoint.path, "--host", "127.0.0.1", "--port", String(port),
      "--max-tokens", "3", "--temp", "0", "--no-token-echo",
    ]
    if let memoryLimitMB {
      process.arguments! += ["--mem-limit", String(memoryLimitMB)]
    }
    process.standardOutput = output
    process.standardError = output
    try process.run()
    let server = TinyServer(
      process: process, url: URL(string: "http://127.0.0.1:\(port)")!, logURL: logURL)
    do {
      let deadline = Date().addingTimeInterval(30)
      var lastError: (any Error)?
      while process.isRunning, Date() < deadline {
        do {
          _ = try await server.health()
          return server
        } catch {
          lastError = error
        }
        try await Task.sleep(for: .milliseconds(20))
      }
      throw ServerTestError.startup(
        "\(String(describing: lastError))\n\(try String(contentsOf: logURL, encoding: .utf8))")
    } catch {
      server.stop()
      throw error
    }
  }

  private func reservePort() throws -> UInt16 {
    let descriptor = socket(AF_INET, SOCK_STREAM, 0)
    guard descriptor >= 0 else { throw ServerTestError.socket(errno) }
    defer { close(descriptor) }
    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_addr.s_addr = inet_addr("127.0.0.1")
    let bound = withUnsafePointer(to: &address) { pointer in
      pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
        Darwin.bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
      }
    }
    guard bound == 0 else { throw ServerTestError.socket(errno) }
    var length = socklen_t(MemoryLayout<sockaddr_in>.size)
    let named = withUnsafeMutablePointer(to: &address) { pointer in
      pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
        getsockname(descriptor, $0, &length)
      }
    }
    guard named == 0 else { throw ServerTestError.socket(errno) }
    return UInt16(bigEndian: address.sin_port)
  }
}

private enum ServerTestError: Error {
  case missingExecutable(String)
  case socket(Int32)
  case startup(String)
  case status(Int)
  case request(String)
}

private struct TinyServer {
  let process: Process
  let url: URL
  let logURL: URL

  func stop() {
    print("SwiftLM child before cleanup: \(processState)")
    if process.isRunning {
      process.terminate()
      let deadline = Date().addingTimeInterval(2)
      while process.isRunning, Date() < deadline {
        Thread.sleep(forTimeInterval: 0.01)
      }
      if process.isRunning { kill(process.processIdentifier, SIGKILL) }
      process.waitUntilExit()
    }
    print("SwiftLM child after cleanup: \(processState)")
    do {
      let output = try String(contentsOf: logURL, encoding: .utf8)
      print("Final SwiftLM child output (\(logURL.path)):\n\(output)")
    } catch {
      print("Cannot read final SwiftLM child output at \(logURL.path): \(error)")
    }
  }

  private var processState: String {
    if process.isRunning {
      return "pid=\(process.processIdentifier), running=true"
    }
    let reason: String
    switch process.terminationReason {
    case .exit: reason = "exit"
    case .uncaughtSignal: reason = "signal"
    @unknown default: reason = "unknown"
    }
    return "pid=\(process.processIdentifier), running=false, reason=\(reason), status=\(process.terminationStatus)"
  }

  func health() async throws -> HealthReply {
    let data = try await request(path: "health", body: nil)
    return try JSONDecoder().decode(HealthReply.self, from: data)
  }

  func chat(streaming: Bool, logprobs: Bool) async throws -> Data {
    let body = Data(
      """
      {"messages":[{"role":"user","content":"hello"}],"max_tokens":3,
       "temperature":0,"stream":\(streaming),"logprobs":\(logprobs)}
      """.utf8)
    return try await request(path: "v1/chat/completions", body: body)
  }

  private func request(path: String, body: Data?) async throws -> Data {
    var request = URLRequest(
      url: url.appendingPathComponent(path), timeoutInterval: body == nil ? 3 : 20)
    if let body {
      request.httpMethod = "POST"
      request.httpBody = body
      request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    }
    do {
      let (data, response) = try await URLSession.shared.data(for: request)
      let status = (response as? HTTPURLResponse)?.statusCode ?? 0
      guard status == 200 else { throw ServerTestError.status(status) }
      return data
    } catch {
      guard body != nil else { throw error }
      throw ServerTestError.request(
        "\(error)\nSwiftLM child at request failure: \(processState)\n\(try String(contentsOf: logURL, encoding: .utf8))")
    }
  }
}

private struct HealthReply: Decodable {
  let memory: MemoryValues
  struct MemoryValues: Decodable {
    let limitMB: Int
    let peakMB: Int
    enum CodingKeys: String, CodingKey {
      case limitMB = "limit_mb"
      case peakMB = "peak_mb"
    }
  }
}

private struct ChatReply: Decodable {
  let choices: [Choice]
  struct Choice: Decodable {
    let message: Message?
    let delta: Message?
    let logprobs: Probabilities?
  }
  struct Message: Decodable {
    let content: String?
  }
  struct Probabilities: Decodable {
    let content: [ProbabilityEntry]
  }
}

private struct ProbabilityEntry: Decodable {
  let token: String
  let logprob: Double
  let bytes: [Int]
  let topLogprobs: [Alternative]
  enum CodingKeys: String, CodingKey {
    case token, logprob, bytes
    case topLogprobs = "top_logprobs"
  }
  struct Alternative: Decodable {
    let token: String
    let logprob: Double
  }
}
