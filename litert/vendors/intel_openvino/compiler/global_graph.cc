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
#include <vector>

#include "litert/cc/litert_expected.h"

namespace litert::openvino {
namespace {

constexpr char kMagic[8] = {'O', 'V', 'G', 'L', 'O', 'B', 'A', 'L'};

void PutU32(std::string& s, uint32_t v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof(v));
}
void PutU64(std::string& s, uint64_t v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof(v));
}

// Bounds-checked little-endian readers over a [data, data+size) cursor. Tracks
// the base so callers can recover the absolute byte offset of the cursor (used
// to locate the contiguous pool without copying it).
struct Reader {
  const uint8_t* base;
  const uint8_t* p;
  const uint8_t* end;
  bool ok = true;
  size_t Offset() const { return static_cast<size_t>(p - base); }
  bool Bytes(void* out, size_t n) {
    if (!ok || static_cast<size_t>(end - p) < n) {
      ok = false;
      return false;
    }
    std::memcpy(out, p, n);
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
  // Returns a view of the next |n| bytes and advances past them (no copy).
  const uint8_t* View(size_t n) {
    if (!ok || static_cast<size_t>(end - p) < n) {
      ok = false;
      return nullptr;
    }
    const uint8_t* v = p;
    p += n;
    return v;
  }
  // Skips |n| bytes without reading them.
  bool Skip(size_t n) {
    if (!ok || static_cast<size_t>(end - p) < n) {
      ok = false;
      return false;
    }
    p += n;
    return true;
  }
};

// One entry of the on-disk buffer directory.
struct DirEntry {
  uint32_t id;
  uint64_t pool_offset;
  uint64_t size;
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

std::string OpenVinoGlobalGraph::Serialize() const {
  std::string out;
  out.append(kMagic, sizeof(kMagic));
  PutU32(out, kVersion);

  // Buffer directory: (id, pool_offset, size) for every buffer, in ascending
  // buffer_id order (std::map iterates sorted). pool_offset is the buffer's
  // byte position in the contiguous pool written below.
  PutU32(out, static_cast<uint32_t>(buffers.size()));
  uint64_t running_offset = 0;
  for (const auto& [id, bytes] : buffers) {
    PutU32(out, id);
    PutU64(out, running_offset);
    PutU64(out, static_cast<uint64_t>(bytes.size()));
    running_offset += bytes.size();
  }

  // Contiguous shared pool: all buffer bytes back-to-back, same ascending
  // order, so a consumer can map/write exactly this run and resolve each
  // buffer as pool + pool_offset.
  PutU64(out, running_offset);  // pool_size
  for (const auto& [id, bytes] : buffers) {
    out.append(bytes);
  }

  // Subgraphs.
  PutU32(out, static_cast<uint32_t>(subgraphs.size()));
  for (const auto& [name, subgraph] : subgraphs) {
    PutU32(out, static_cast<uint32_t>(subgraph.name.size()));
    out.append(subgraph.name);
    out.push_back(static_cast<char>(subgraph.device));
    PutU32(out, static_cast<uint32_t>(subgraph.const_map.size()));
    for (const auto& [input_index, buffer_id] : subgraph.const_map) {
      PutU32(out, input_index);
      PutU32(out, buffer_id);
    }
    PutU64(out, subgraph.payload.size());
    out.append(subgraph.payload);
  }
  return out;
}

litert::Expected<OpenVinoGlobalGraph> OpenVinoGlobalGraph::Parse(
    const uint8_t* data, size_t size) {
  if (!HasMagic(data, size)) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: bad magic");
  }
  Reader reader{data, data + sizeof(kMagic), data + size};
  OpenVinoGlobalGraph graph;

  const uint32_t version = reader.U32();
  if (reader.ok && version != kVersion) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: unsupported container version");
  }

  // Buffer directory.
  const uint32_t num_buffers = reader.U32();
  std::vector<DirEntry> dir;
  dir.reserve(num_buffers);
  for (uint32_t i = 0; i < num_buffers && reader.ok; ++i) {
    DirEntry e;
    e.id = reader.U32();
    e.pool_offset = reader.U64();
    e.size = reader.U64();
    dir.push_back(e);
  }

  // Contiguous pool. Slice each buffer out of it by (pool_offset, size).
  const uint64_t pool_size = reader.U64();
  const uint8_t* pool = reader.View(static_cast<size_t>(pool_size));
  if (reader.ok) {
    for (const auto& e : dir) {
      if (e.pool_offset > pool_size || e.size > pool_size - e.pool_offset) {
        return litert::Error(
            kLiteRtStatusErrorRuntimeFailure,
            "OpenVinoGlobalGraph: buffer directory entry out of pool range");
      }
      graph.buffers.emplace(
          e.id, std::string(reinterpret_cast<const char*>(pool + e.pool_offset),
                            static_cast<size_t>(e.size)));
    }
  }

  // Subgraphs.
  const uint32_t num_subgraphs = reader.U32();
  for (uint32_t i = 0; i < num_subgraphs && reader.ok; ++i) {
    Subgraph subgraph;
    const uint32_t name_len = reader.U32();
    reader.Str(subgraph.name, name_len);
    uint8_t dev = 0;
    reader.Bytes(&dev, 1);
    subgraph.device = dev;
    const uint32_t cm_len = reader.U32();
    for (uint32_t j = 0; j < cm_len && reader.ok; ++j) {
      const uint32_t idx = reader.U32();
      const uint32_t bid = reader.U32();
      subgraph.const_map.emplace(idx, bid);
    }
    const uint64_t payload_len = reader.U64();
    reader.Str(subgraph.payload, static_cast<size_t>(payload_len));
    if (reader.ok) graph.subgraphs.emplace(subgraph.name, std::move(subgraph));
  }

  if (!reader.ok) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: truncated/corrupt container");
  }
  return graph;
}

litert::Expected<OpenVinoGlobalGraph::Header>
OpenVinoGlobalGraph::ParseHeader(const uint8_t* data, size_t size) {
  if (!HasMagic(data, size)) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: bad magic");
  }
  Reader reader{data, data + sizeof(kMagic), data + size};
  Header header;

  const uint32_t version = reader.U32();
  if (reader.ok && version != kVersion) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: unsupported container version");
  }

  // Buffer directory: validated below against pool_size but not copied.
  const uint32_t num_buffers = reader.U32();
  std::vector<DirEntry> dir;
  dir.reserve(num_buffers);
  for (uint32_t i = 0; i < num_buffers && reader.ok; ++i) {
    DirEntry e;
    e.id = reader.U32();
    e.pool_offset = reader.U64();
    e.size = reader.U64();
    dir.push_back(e);
  }

  // Contiguous pool: record its span (view) and skip past it -- NO copy.
  const uint64_t pool_size = reader.U64();
  header.pool_size = static_cast<size_t>(pool_size);
  header.pool_data_offset = reader.Offset();
  header.pool = reader.View(static_cast<size_t>(pool_size));

  if (reader.ok) {
    for (const auto& e : dir) {
      if (e.pool_offset > pool_size || e.size > pool_size - e.pool_offset) {
        return litert::Error(
            kLiteRtStatusErrorRuntimeFailure,
            "OpenVinoGlobalGraph: buffer directory entry out of pool range");
      }
    }
  }

  // Subgraphs: payloads are viewed (not copied) into the caller's blob.
  const uint32_t num_subgraphs = reader.U32();
  for (uint32_t i = 0; i < num_subgraphs && reader.ok; ++i) {
    SubgraphView view;
    const uint32_t name_len = reader.U32();
    reader.Str(view.name, name_len);
    uint8_t dev = 0;
    reader.Bytes(&dev, 1);
    view.device = dev;
    const uint32_t cm_len = reader.U32();
    for (uint32_t j = 0; j < cm_len && reader.ok; ++j) {
      const uint32_t idx = reader.U32();
      const uint32_t bid = reader.U32();
      view.const_map.emplace(idx, bid);
    }
    const uint64_t payload_len = reader.U64();
    view.payload_size = static_cast<size_t>(payload_len);
    view.payload = reader.View(static_cast<size_t>(payload_len));
    if (reader.ok) {
      const std::string name = view.name;
      header.subgraphs.emplace(name, std::move(view));
    }
  }

  if (!reader.ok) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: truncated/corrupt container");
  }
  return header;
}

}  // namespace litert::openvino
