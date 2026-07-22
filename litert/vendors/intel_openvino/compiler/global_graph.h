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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_GLOBAL_GRAPH_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_GLOBAL_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

#include "litert/cc/litert_expected.h"

namespace litert::openvino {

// Container for cross-partition weight sharing: all partitions are aggregated
// into one blob holding a shared buffer pool (deduplicated weight bytes) plus a
// per-partition subgraph, and the SAME blob is returned for every partition.
// The dispatcher parses it, selects its subgraph, and resolves that subgraph's
// weights against the shared pool.
//
// Each subgraph's const_map records how its OV payload references the pool: it
// maps a weight-Parameter's input_index to the pool buffer_id it is bound to at
// dispatch (used by the GPU USM path). On the NPU path weights stay Constants
// tagged with WeightlessCacheAttribute(bin_offset) and const_map is empty; the
// bin_offset is the buffer's byte position in the CONTIGUOUS pool (below), which
// NPUW resolves as `mapped_pool->data() + bin_offset`.
//
// Serialized layout (single blob, little-endian):
//   magic  "OVGLOBAL"                       (8 bytes)
//   uint32 version                          (kVersion)
//   -- buffer directory (no bytes here; keeps the pool contiguous) --
//   uint32 num_buffers
//     repeat: uint32 buffer_id, uint64 pool_offset, uint64 size
//   -- contiguous shared pool --
//   uint64 pool_size
//   [pool_size bytes]                       (buffers laid out at their
//                                            pool_offset, ascending buffer_id)
//   -- subgraphs --
//   uint32 num_subgraphs
//     repeat: uint32 name_len, [name], uint8 device_enum,
//             uint32 const_map_len,
//               repeat: uint32 index, uint32 buffer_id
//             uint64 payload_len, [payload bytes]
//
// The pool is stored as one contiguous run (unlike the earlier interleaved
// {id,size,bytes} layout) so the dispatcher can hand exactly the pool bytes to
// a memfd at offset 0 and have NPUW's whole-fd `data() + bin_offset` resolve
// correctly. `ParseHeader` exposes the pool span and per-subgraph payload spans
// as views (no pool copy) for that path; `Parse` still materializes the
// `buffers` map for the GPU USM path.
class OpenVinoGlobalGraph {
 public:
  // Container format version. Bumped when the on-disk layout changes; a
  // dispatcher rejects blobs it does not understand.
  static constexpr uint32_t kVersion = 2;

  // One compiled partition entry in the container.
  struct Subgraph {
    std::string name;                        // e.g. "Partition_0"
    uint8_t device = 0;                      // LiteRtIntelOpenVinoGraphBackend
    std::map<uint32_t, uint32_t> const_map;  // index -> buffer_id (GPU path)
    std::string payload;                     // OV exported blob
  };

  // Shared buffer pool: buffer_id -> raw weight bytes (deduplicated).
  std::map<uint32_t, std::string> buffers;
  // Partition topologies, keyed by graph name (selected at dispatch by
  // function_name / graph order).
  std::map<std::string, Subgraph> subgraphs;

  // Serialize the whole container to one blob (see layout above). The pool is
  // written contiguously in ascending buffer_id order; each buffer's
  // pool_offset in the directory is its byte position in that run.
  std::string Serialize() const;

  // Parse a container blob, materializing `buffers` (a copy of the pool) and
  // `subgraphs`. Used by the GPU path. Returns an error on bad magic/version or
  // truncation.
  static litert::Expected<OpenVinoGlobalGraph> Parse(const uint8_t* data,
                                                     size_t size);

  // Zero-copy view of one subgraph inside a serialized container. `payload`
  // points into the caller's blob (no copy); valid while that blob lives.
  struct SubgraphView {
    std::string name;
    uint8_t device = 0;
    std::map<uint32_t, uint32_t> const_map;
    const uint8_t* payload = nullptr;
    size_t payload_size = 0;
  };

  // Lightweight, zero-copy parse for the NPU path: locates the contiguous pool
  // and every subgraph's payload span WITHOUT copying the (potentially
  // multi-GB) pool. `pool` points at the first pool byte inside `data`
  // (i.e. data + pool_data_offset). Every directory entry is validated so
  // `pool_offset + size <= pool_size`. Returns an error on bad magic/version,
  // truncation, or an out-of-range directory entry.
  struct Header {
    size_t pool_data_offset = 0;      // offset from `data` to first pool byte
    size_t pool_size = 0;             // total contiguous pool bytes
    const uint8_t* pool = nullptr;    // == data + pool_data_offset (view)
    std::map<std::string, SubgraphView> subgraphs;
  };
  static litert::Expected<Header> ParseHeader(const uint8_t* data, size_t size);

  // Fast check: does |data| begin with the OVGLOBAL magic?
  static bool HasMagic(const uint8_t* data, size_t size);

  // Total bytes across the shared buffer pool (the deduplicated weight size).
  size_t BankBytes() const;
};

}  // namespace litert::openvino

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_GLOBAL_GRAPH_H_
