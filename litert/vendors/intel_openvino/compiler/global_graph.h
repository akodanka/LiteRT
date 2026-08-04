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
// maps a weight-Parameter's friendly_name to the pool buffer_id it is bound to
// at dispatch (matched by name so binding is robust to input reordering across
// import_model).
//
// Serialized layout (single blob, little-endian):
//   magic  "OVGLOBAL"                       (8 bytes)
//   uint16 version                          (format version, see kVersion)
//   uint32 num_buffers
//     directory, repeat num_buffers times (ascending buffer_id):
//       uint32 buffer_id, uint64 pool_offset, uint64 size
//   uint64 pool_size
//   [pool_size bytes]                       (CONTIGUOUS deduplicated pool)
//   uint32 num_subgraphs
//     repeat: uint32 name_len, [name], uint8 device_enum,
//             uint32 const_map_len,
//               repeat: uint32 name_len, [name], uint32 buffer_id  (const_map)
//             uint64 payload_len, [payload bytes]             (OV exported
//             blob)
//
// v2 change over v1: the pool is stored CONTIGUOUSLY (a directory of
// {id, pool_offset, size} followed by all pool bytes back-to-back) instead of
// interleaving {id, size, bytes}. This lets the NPU weightless path stage the
// pool to a temp file as a byte-for-byte copy of the [pool_data_offset,
// pool_data_offset + pool_size) span, so a Constant's WeightlessCacheAttribute
// bin_offset (== its pool_offset) resolves to mmap->data() + bin_offset.
class OpenVinoGlobalGraph {
 public:
  // Container format version, written right after the magic. Bump on any
  // layout change so Parse() can reject blobs it does not understand rather
  // than misparsing. v2 = contiguous pool + directory (see layout above).
  static constexpr uint16_t kVersion = 2;

  // One compiled partition entry in the container.
  struct Subgraph {
    std::string name;    // e.g. "Partition_0"
    uint8_t device = 0;  // LiteRtIntelOpenVinoGraphBackend
    std::map<std::string, uint32_t> const_map;  // friendly_name -> buffer_id
    std::string payload;                        // OV exported blob
  };

  // A borrowed, contiguous byte range into a caller-owned container blob.
  struct Span {
    const uint8_t* data = nullptr;
    size_t size = 0;
  };

  // Zero-copy view of a container: locates the contiguous pool and each
  // subgraph's payload WITHOUT copying the (multi-GB) pool. Produced by
  // ParseHeader; all Span/pointer members alias the caller-owned blob, which
  // must outlive the Header. Used by the NPU weightless dispatch path to find
  // the pool bytes to stage to a temp file.
  struct Header {
    // Byte offset from the container start to the first pool byte, and the
    // total contiguous pool size. The temp file is a copy of
    // [container + pool_data_offset, +pool_size).
    size_t pool_data_offset = 0;
    size_t pool_size = 0;
    Span pool;  // == {container + pool_data_offset, pool_size}
    // Per-subgraph zero-copy views, keyed by graph name.
    struct SubgraphView {
      uint8_t device = 0;
      std::map<std::string, uint32_t> const_map;  // friendly_name -> buffer_id
      Span payload;                               // OV exported blob (aliased)
    };
    std::map<std::string, SubgraphView> subgraphs;
  };

  // Shared buffer pool: buffer_id -> raw weight bytes (deduplicated).
  std::map<uint32_t, std::string> buffers;
  // Partition topologies, keyed by graph name (selected at dispatch by
  // function_name / graph order).
  std::map<std::string, Subgraph> subgraphs;

  // Serialize the whole container to one blob (see layout above).
  std::string Serialize() const;

  // Parse a container blob, materializing the buffers map (copies the pool).
  // Used by the GPU path, which needs the pool bytes keyed by buffer_id to
  // fill its USM-host bank. Returns an error if magic/bounds are invalid.
  static litert::Expected<OpenVinoGlobalGraph> Parse(const uint8_t* data,
                                                     size_t size);

  // Zero-copy parse: returns a Header whose Spans alias |data| (which must
  // outlive the returned Header). Validates the magic, the directory (every
  // {pool_offset, size} lies within pool_size), and all bounds. Used by the
  // NPU weightless path to locate the contiguous pool + payloads without
  // copying the pool. Returns an error on bad magic / bounds.
  static litert::Expected<Header> ParseHeader(const uint8_t* data, size_t size);

  // Fast check: does |data| begin with the OVGLOBAL magic?
  static bool HasMagic(const uint8_t* data, size_t size);

  // Total bytes across the shared buffer pool (the deduplicated weight size).
  size_t BankBytes() const;
};

}  // namespace litert::openvino

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_GLOBAL_GRAPH_H_
