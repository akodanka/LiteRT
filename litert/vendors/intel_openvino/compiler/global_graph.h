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
#include <unordered_map>

#include "litert/cc/litert_expected.h"

namespace litert::openvino {

// Container for cross-partition weight sharing: all partitions are aggregated
// into one blob holding a shared buffer pool (deduplicated weight bytes) plus a
// per-partition subgraph, and the SAME blob is returned for every partition.
// The dispatcher parses it, selects its subgraph, and resolves that subgraph's
// weights against the shared pool.
//
// The shared pool is stored CONTIGUOUS and UNCOMPRESSED at the end of the
// container (no per-buffer headers interleaved with the bytes), so a
// pool-relative byte offset addresses a weight directly. This lets the NPU
// dispatch path mmap the pool region straight out of the fd-backed model file
// (zero-copy, ov::weight_sharing / NPUW). The buffer index (buffer_id ->
// pool_offset + size) is stored separately, before the pool.
//
// Each subgraph's const_map records how its OV payload references the pool: it
// maps a weight-Parameter's input_index to the pool buffer_id it is bound to at
// dispatch. It is used by the GPU USM path; the NPU descriptor path leaves it
// empty (weight identity travels on each Constant's buffer descriptor instead).
//
// Serialized layout (single blob, little-endian):
//   magic  "OVGLOBAL"                       (8 bytes)
//   uint64 pool_data_offset                 (offset from container start to the
//                                            first pool byte)
//   uint64 pool_size                        (total pool bytes == BankBytes())
//   uint32 num_buffers                                          (buffer index)
//     repeat: uint32 buffer_id, uint64 pool_offset, uint64 size
//   uint32 num_subgraphs
//     repeat: uint32 name_len, [name], uint8 device_enum,
//             uint32 const_map_len,
//               repeat: uint32 index, uint32 buffer_id        (const_map)
//             uint64 payload_len, [payload bytes]             (OV exported blob)
//   [pool bytes]                            (contiguous, at pool_data_offset)
class OpenVinoGlobalGraph {
 public:
  // One compiled partition entry in the container.
  struct Subgraph {
    std::string name;                        // e.g. "Partition_0"
    uint8_t device = 0;                      // LiteRtIntelOpenVinoGraphBackend
    std::map<uint32_t, uint32_t> const_map;  // index -> buffer_id (GPU only)
    std::string payload;                     // OV exported blob
  };

  // Shared buffer pool: buffer_id -> raw weight bytes (deduplicated).
  std::map<uint32_t, std::string> buffers;
  // Partition topologies, keyed by graph name (selected at dispatch by
  // function_name / graph order).
  std::map<std::string, Subgraph> subgraphs;

  // Serialize the whole container to one blob (see layout above).
  std::string Serialize() const;

  // Parse a container blob (full copy of the pool into `buffers`). Used by the
  // GPU USM path and the buffer-backed fallback. Returns an error if
  // magic/bounds are invalid.
  static litert::Expected<OpenVinoGlobalGraph> Parse(const uint8_t* data,
                                                     size_t size);

  // Fast check: does |data| begin with the OVGLOBAL magic?
  static bool HasMagic(const uint8_t* data, size_t size);

  // Total bytes across the shared buffer pool (the deduplicated weight size).
  size_t BankBytes() const;

  // Deterministic buffer_id -> pool byte-offset map, matching the ascending
  // buffer_id order Serialize() writes the pool in. The compiler publishes
  // these offsets to the TFLite frontend as each weight's descriptor
  // bin_offset, so they must agree with the serialized layout. Call after
  // `buffers` is populated.
  std::unordered_map<int32_t, std::size_t> ComputePoolOffsets() const;

  // Zero-copy view over a serialized container's header: locates the pool
  // region and each subgraph's payload by offset/size WITHOUT copying the pool
  // or payload bytes. Used by the NPU fd-backed dispatch path (§7.3).
  struct HeaderView {
    // Location of the contiguous pool, relative to the container start.
    size_t pool_data_offset = 0;
    size_t pool_size = 0;

    struct BufferLoc {
      size_t pool_offset = 0;  // byte offset within the pool
      size_t size = 0;
    };
    // buffer_id -> location within the pool (for validating per-weight offsets
    // against pool_size).
    std::map<uint32_t, BufferLoc> buffer_index;

    struct SubgraphView {
      std::string name;
      uint8_t device = 0;
      std::map<uint32_t, uint32_t> const_map;
      size_t payload_offset = 0;  // relative to the container start
      size_t payload_size = 0;
    };
    std::map<std::string, SubgraphView> subgraphs;
  };

  // Header-only parse: copies nothing but the (small) names/const_maps; the
  // pool and payloads are returned as offset/size spans into |data|.
  static litert::Expected<HeaderView> ParseHeader(const uint8_t* data,
                                                  size_t size);
};

}  // namespace litert::openvino

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_GLOBAL_GRAPH_H_
