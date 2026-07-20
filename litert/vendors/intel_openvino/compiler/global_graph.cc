// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/intel_openvino/compiler/global_graph.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>

#include "litert/cc/litert_expected.h"

namespace litert::openvino {
namespace {

constexpr char kMagic[8] = {'O', 'V', 'G', 'L', 'O', 'B', 'A', 'L'};

// Bytes of fixed header preceding the variable-length body:
//   magic(8) + pool_data_offset(8) + pool_size(8)
constexpr size_t kFixedHeaderBytes = sizeof(kMagic) + sizeof(uint64_t) * 2;

void PutU32(std::string& s, uint32_t v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof(v));
}
void PutU64(std::string& s, uint64_t v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof(v));
}

// Bounds-checked little-endian readers over a [begin, end) cursor. Tracks the
// absolute position so callers can record offsets of pool/payload spans.
struct Reader {
  const uint8_t* begin;
  const uint8_t* p;
  const uint8_t* end;
  bool ok = true;
  size_t Pos() const { return static_cast<size_t>(p - begin); }
  bool Bytes(void* out, size_t n) {
    if (!ok || static_cast<size_t>(end - p) < n) {
      ok = false;
      return false;
    }
    std::memcpy(out, p, n);
    p += n;
    return true;
  }
  // Advance over n bytes without copying; returns the position where they start.
  bool Skip(size_t n, size_t* start_pos) {
    if (!ok || static_cast<size_t>(end - p) < n) {
      ok = false;
      return false;
    }
    if (start_pos) *start_pos = Pos();
    p += n;
    return true;
  }
  uint32_t U32() {
    uint32_t v = 0;
    Bytes(&v, sizeof(v));
    return v;
  }
  uint64_t U64() {
    uint64_t v = 0;
    Bytes(&v, sizeof(v));
    return v;
  }
  bool Str(std::string& out, size_t n) {
    if (!ok || static_cast<size_t>(end - p) < n) {
      ok = false;
      return false;
    }
    out.assign(reinterpret_cast<const char*>(p), n);
    p += n;
    return true;
  }
};

}  // namespace

bool OpenVinoGlobalGraph::HasMagic(const uint8_t* data, size_t size) {
  return data != nullptr && size >= sizeof(kMagic) &&
         std::memcmp(data, kMagic, sizeof(kMagic)) == 0;
}

size_t OpenVinoGlobalGraph::BankBytes() const {
  size_t total = 0;
  for (const auto& [id, bytes] : buffers) total += bytes.size();
  return total;
}

std::unordered_map<int32_t, std::size_t>
OpenVinoGlobalGraph::ComputePoolOffsets() const {
  std::unordered_map<int32_t, std::size_t> offsets;
  offsets.reserve(buffers.size());
  size_t running = 0;
  // std::map iterates in ascending key order -- the same order Serialize()
  // writes the pool -- so these offsets match the on-disk layout.
  for (const auto& [id, bytes] : buffers) {
    offsets.emplace(static_cast<int32_t>(id), running);
    running += bytes.size();
  }
  return offsets;
}

std::string OpenVinoGlobalGraph::Serialize() const {
  // Build the variable-length body (buffer index + subgraphs) first so we can
  // compute pool_data_offset before writing the fixed header.
  std::string body;
  // Buffer index: buffer_id -> (pool_offset, size), ascending buffer_id.
  PutU32(body, static_cast<uint32_t>(buffers.size()));
  size_t running = 0;
  for (const auto& [id, bytes] : buffers) {
    PutU32(body, id);
    PutU64(body, static_cast<uint64_t>(running));
    PutU64(body, bytes.size());
    running += bytes.size();
  }
  // Subgraphs.
  PutU32(body, static_cast<uint32_t>(subgraphs.size()));
  for (const auto& [name, subgraph] : subgraphs) {
    PutU32(body, static_cast<uint32_t>(subgraph.name.size()));
    body.append(subgraph.name);
    body.push_back(static_cast<char>(subgraph.device));
    PutU32(body, static_cast<uint32_t>(subgraph.const_map.size()));
    for (const auto& [input_index, buffer_id] : subgraph.const_map) {
      PutU32(body, input_index);
      PutU32(body, buffer_id);
    }
    PutU64(body, subgraph.payload.size());
    body.append(subgraph.payload);
  }

  const uint64_t pool_data_offset = kFixedHeaderBytes + body.size();
  const uint64_t pool_size = BankBytes();

  std::string out;
  out.reserve(pool_data_offset + pool_size);
  out.append(kMagic, sizeof(kMagic));
  PutU64(out, pool_data_offset);
  PutU64(out, pool_size);
  out.append(body);
  // Contiguous pool, ascending buffer_id (matches the index offsets above).
  for (const auto& [id, bytes] : buffers) {
    out.append(bytes);
  }
  return out;
}

litert::Expected<OpenVinoGlobalGraph> OpenVinoGlobalGraph::Parse(
    const uint8_t* data, size_t size) {
  auto header = ParseHeader(data, size);
  if (!header.HasValue()) {
    return litert::Error(header.Error().Status(), header.Error().Message());
  }
  const HeaderView& view = header.Value();

  OpenVinoGlobalGraph graph;
  // Copy each pool buffer out of the contiguous pool region using the index.
  for (const auto& [id, loc] : view.buffer_index) {
    const size_t abs = view.pool_data_offset + loc.pool_offset;
    // ParseHeader already validated abs + size <= size.
    graph.buffers.emplace(
        id, std::string(reinterpret_cast<const char*>(data + abs), loc.size));
  }
  for (const auto& [name, sv] : view.subgraphs) {
    Subgraph subgraph;
    subgraph.name = sv.name;
    subgraph.device = sv.device;
    subgraph.const_map = sv.const_map;
    subgraph.payload.assign(
        reinterpret_cast<const char*>(data + sv.payload_offset),
        sv.payload_size);
    graph.subgraphs.emplace(subgraph.name, std::move(subgraph));
  }
  return graph;
}

litert::Expected<OpenVinoGlobalGraph::HeaderView>
OpenVinoGlobalGraph::ParseHeader(const uint8_t* data, size_t size) {
  if (!HasMagic(data, size)) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: bad magic");
  }
  Reader reader{data, data + sizeof(kMagic), data + size};
  HeaderView view;
  view.pool_data_offset = static_cast<size_t>(reader.U64());
  view.pool_size = static_cast<size_t>(reader.U64());

  // Buffer index.
  const uint32_t num_buffers = reader.U32();
  for (uint32_t i = 0; i < num_buffers && reader.ok; ++i) {
    const uint32_t id = reader.U32();
    const uint64_t pool_offset = reader.U64();
    const uint64_t sz = reader.U64();
    if (!reader.ok) break;
    // Every weight must lie fully within the declared pool.
    if (pool_offset > view.pool_size || sz > view.pool_size - pool_offset) {
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "OpenVinoGlobalGraph: buffer index out of pool range");
    }
    view.buffer_index.emplace(
        id, HeaderView::BufferLoc{static_cast<size_t>(pool_offset),
                                  static_cast<size_t>(sz)});
  }

  // Subgraphs (payloads recorded as spans, not copied).
  const uint32_t num_subgraphs = reader.U32();
  for (uint32_t i = 0; i < num_subgraphs && reader.ok; ++i) {
    HeaderView::SubgraphView sv;
    const uint32_t name_len = reader.U32();
    reader.Str(sv.name, name_len);
    uint8_t dev = 0;
    reader.Bytes(&dev, 1);
    sv.device = dev;
    const uint32_t cm_len = reader.U32();
    for (uint32_t j = 0; j < cm_len && reader.ok; ++j) {
      const uint32_t idx = reader.U32();
      const uint32_t bid = reader.U32();
      sv.const_map.emplace(idx, bid);
    }
    const uint64_t payload_len = reader.U64();
    size_t payload_pos = 0;
    reader.Skip(static_cast<size_t>(payload_len), &payload_pos);
    sv.payload_offset = payload_pos;
    sv.payload_size = static_cast<size_t>(payload_len);
    if (reader.ok) view.subgraphs.emplace(sv.name, std::move(sv));
  }

  if (!reader.ok) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: truncated/corrupt container");
  }
  // The body must end exactly where the pool begins, and the pool must fit.
  if (reader.Pos() != view.pool_data_offset ||
      view.pool_data_offset > size ||
      view.pool_size > size - view.pool_data_offset) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: pool region inconsistent");
  }
  return view;
}

}  // namespace litert::openvino
