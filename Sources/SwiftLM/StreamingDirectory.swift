// StreamingDirectory.swift — the directory `--stream-experts` loads a Hub model from.
//
// Streaming targets one directory, and a directory load doesn't fill in missing
// files, so that copy must be complete. A local copy whose files show it is complete
// is used as is; otherwise the Hub listing decides.

import Foundation
import Hub
import MLXInferenceCore

/// Files SwiftLM downloads for a model: weights, configs, tokenizer and templates.
let modelDownloadPatterns = ["*.safetensors", "*.json", "*.jinja"]

/// A `--stream-experts` download finished without the files the loader needs.
struct ModelDownloadIncomplete: Error, CustomStringConvertible {
    let modelId: String
    let directory: URL
    var description: String {
        "Download of \(modelId) is incomplete (\(directory.path)). Check the network, or delete the directory and retry."
    }
}

/// The repository has no `tokenizer.json`, which the tokenizer loader requires.
struct ModelMissingTokenizer: Error, CustomStringConvertible {
    let modelId: String
    var description: String {
        "\(modelId) has no tokenizer.json, which SwiftLM needs to load it."
    }
}

/// The Hub answered 401/404 and there is no usable local copy.
struct ModelNotOnHub: Error, CustomStringConvertible {
    let modelId: String
    let underlying: Error
    var description: String {
        "The Hub has no accessible repository \(modelId) (\(underlying)), and no usable local copy was found. Check the id, or set HF_TOKEN for a private repository."
    }
}

/// The Hub couldn't be used and there is no usable local copy.
struct ModelUnavailableOffline: Error, CustomStringConvertible {
    let modelId: String
    let underlying: Error
    var description: String {
        "Could not get \(modelId) from the Hub (\(underlying)), and no usable local copy was found."
    }
}

/// Whether a local copy's weights are known to be complete.
enum LocalWeights: Equatable {
    case complete
    case incomplete
    /// A layout the file names can't vouch for (`weights.00.safetensors`, say);
    /// the Hub listing decides.
    case unverified
}

/// Whether `name` in `directory` exists and is non-empty, following symlinks
/// (Hub snapshots link to blobs, and a dangling link counts as missing).
func isPresentFile(_ name: String, in directory: URL) -> Bool {
    let url = directory.appendingPathComponent(name)
    let fm = FileManager.default
    guard fm.fileExists(atPath: url.path) else { return false }
    let attributes = try? fm.attributesOfItem(atPath: url.resolvingSymlinksInPath().path)
    return ((attributes?[.size] as? Int) ?? 0) > 0
}

/// `stem-00002-of-00004.safetensors` as (stem, 2, 4); nil for other names.
func shardNumber(_ name: String) -> (stem: String, index: Int, count: Int)? {
    guard name.hasSuffix(".safetensors") else { return nil }
    let base = name.dropLast(".safetensors".count)
    guard let of = base.range(of: "-of-", options: .backwards) else { return nil }
    let countText = base[of.upperBound...]
    let head = base[..<of.lowerBound]
    guard let dash = head.lastIndex(of: "-") else { return nil }
    let indexText = head[head.index(after: dash)...]
    guard countText.count == 5, indexText.count == 5,
        countText.allSatisfy(\.isNumber), indexText.allSatisfy(\.isNumber),
        let index = Int(indexText), let count = Int(countText)
    else { return nil }
    return (String(head[..<dash]), index, count)
}

/// The weight files the loader would read, judged for completeness. Mirrors
/// `safetensorWeightURLs`: the index when every file it names exists, otherwise
/// `model*`, then `weight*`, then every top-level `*.safetensors`.
func localWeightState(in directory: URL) -> LocalWeights {
    let indexed = indexedWeightFiles(in: directory)
    if !indexed.isEmpty {
        let missing = indexed.filter { !isPresentFile($0, in: directory) }
        if missing.isEmpty { return .complete }
        // A stale index names top-level shards the repo doesn't ship; a missing file
        // in a subdirectory (optiq/optiq_vision.safetensors) is really missing.
        if missing.contains(where: { $0.contains("/") }) { return .incomplete }
    }
    let top = ((try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? [])
        .filter { $0.hasSuffix(".safetensors") }
    let chosen = ["model", "weight"]
        .map { prefix in top.filter { $0.hasPrefix(prefix) } }
        .first { !$0.isEmpty } ?? top
    guard !chosen.isEmpty else { return .incomplete }

    // `stem-00002-of-00004.safetensors`: every shard 1...4 must be there.
    var groups = [String: (count: Int, have: Set<Int>)]()
    for name in chosen {
        guard let (stem, i, n) = shardNumber(name) else { continue }
        let key = "\(stem)/\(n)"
        var group = groups[key] ?? (n, [])
        if isPresentFile(name, in: directory) { group.have.insert(i) }
        groups[key] = group
    }
    if !groups.isEmpty {
        let whole = groups.values.allSatisfy { $0.count > 0 && $0.have == Set(1 ... $0.count) }
        return whole ? .complete : .incomplete
    }
    if chosen == ["model.safetensors"] {
        return isPresentFile("model.safetensors", in: directory) ? .complete : .incomplete
    }
    return .unverified
}

/// The files `model.safetensors.index.json` in `directory` names, if any.
func indexedWeightFiles(in directory: URL) -> Set<String> {
    let index = directory.appendingPathComponent("model.safetensors.index.json")
    guard let data = try? Data(contentsOf: index),
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
        let weightMap = json["weight_map"] as? [String: String]
    else { return [] }
    return Set(weightMap.values)
}

/// Listed files `directory` must have: every top-level one, plus any nested one its
/// index names (weights in a subdirectory, such as OptiQ's vision tower).
func requiredListedFiles(_ files: [String], in directory: URL) -> [String] {
    let indexed = indexedWeightFiles(in: directory)
    return files.filter { !$0.contains("/") || indexed.contains($0) }
}

/// Whether `directory` has the non-weight files the loader reads. swift-transformers
/// requires `tokenizer.json`; `tokenizer_config.json` is optional.
func hasModelConfigAndTokenizer(in directory: URL) -> Bool {
    isPresentFile("config.json", in: directory) && isPresentFile("tokenizer.json", in: directory)
}

/// Whether a Hub error means the repository isn't available to this user.
private func isNotOnHub(_ error: Error) -> Bool {
    guard let hubError = error as? Hub.HubClientError else { return false }
    switch hubError {
    case .authorizationRequired, .resourceNotFound, .fileNotFound: return true
    case .httpStatusCode(let code): return code == 401 || code == 404
    default: return false
    }
}

/// The directory `--stream-experts` loads and streams `modelId` from, and whether that
/// copy is complete (a dense model's partial copy is returned for the loader to finish).
///
/// A local copy (`candidate`, then the loader's Application Support copy) whose files
/// show it is complete is used without contacting the Hub. Otherwise the Hub listing
/// decides: a copy with every top-level listed file is used, or the model is downloaded
/// into the Application Support copy. If the Hub can't be used, a copy whose layout
/// its names can't verify is used with a warning.
func resolveStreamingDirectory(
    modelId: String, candidate: URL?, hub: HubApi
) async throws -> (directory: URL, complete: Bool) {
    let fm = FileManager.default
    let localRepo = hub.localRepoLocation(Hub.Repo(id: modelId))
    // Present while a download into localRepo is unfinished; that copy isn't trusted.
    let inProgress = localRepo.appendingPathComponent(".swiftlm-download-in-progress")
    let copies = [candidate, localRepo].compactMap { $0 }.filter {
        fm.fileExists(atPath: $0.path)
            && !($0 == localRepo && fm.fileExists(atPath: inProgress.path))
    }
    let usable = copies.filter(hasModelConfigAndTokenizer)
    if let dir = usable.first(where: { localWeightState(in: $0) == .complete }) {
        return (dir, true)
    }
    let unverified = usable.filter { localWeightState(in: $0) == .unverified }

    let files: [String]
    do {
        files = try await hub.getFilenames(from: modelId, matching: modelDownloadPatterns)
    } catch {
        if let dir = unverified.first {
            print("[SwiftLM] ⚠️  Could not check \(modelId) with the Hub (\(error)); using \(dir.path) as is.")
            return (dir, true)
        }
        if isNotOnHub(error) { throw ModelNotOnHub(modelId: modelId, underlying: error) }
        throw ModelUnavailableOffline(modelId: modelId, underlying: error)
    }
    guard files.contains("tokenizer.json") else { throw ModelMissingTokenizer(modelId: modelId) }
    if let dir = usable.first(where: { dir in
        requiredListedFiles(files, in: dir).allSatisfy { isPresentFile($0, in: dir) }
    }) {
        return (dir, true)
    }

    // A dense model won't stream: leave completing it to the loader instead of
    // downloading everything here first.
    let allCopies = [candidate, localRepo].compactMap { $0 }
    if let dir = allCopies.first(where: { isPresentFile("config.json", in: $0) }),
        let profile = ModelProfiler.profile(modelDirectory: dir, modelId: modelId), !profile.isMoE
    {
        return (dir, false)
    }
    for dir in allCopies where fm.fileExists(atPath: dir.path) {
        let reason = dir == localRepo && fm.fileExists(atPath: inProgress.path)
            ? "its download didn't finish" : "it is missing files"
        print("[SwiftLM] \(dir.path) can't be used as is (\(reason)).")
    }

    print("[SwiftLM] --stream-experts: downloading \(modelId) before loading...")
    try fm.createDirectory(at: localRepo, withIntermediateDirectories: true)
    fm.createFile(atPath: inProgress.path, contents: nil)
    let tracker = ProgressTracker(modelId: modelId)
    defer { tracker.finish() }
    let snapshot = try await hub.snapshot(from: modelId, matching: modelDownloadPatterns) {
        tracker.printProgress($0)
    }
    tracker.finish()
    guard requiredListedFiles(files, in: snapshot).allSatisfy({ isPresentFile($0, in: snapshot) }),
        hasModelConfigAndTokenizer(in: snapshot)
    else {
        throw ModelDownloadIncomplete(modelId: modelId, directory: snapshot)
    }
    try? fm.removeItem(at: inProgress)
    return (snapshot, true)
}
