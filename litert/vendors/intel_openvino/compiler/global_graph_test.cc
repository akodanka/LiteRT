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

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace litert {
namespace openvino {
namespace {

// Builds a container with two shared buffers and two subgraphs whose const_maps
// reference those buffers.
OpenVinoGlobalGraph MakeSample() {
  OpenVinoGlobalGraph graph;
  graph.buffers[0] = std::string("\x01\x02\x03\x04", 4);
  graph.buffers[7] = std::string(10, '\xAB');  // non-contiguous id, larger buffer

  OpenVinoGlobalGraph::Subgraph prefill;
  prefill.name = "Partition_0";
  prefill.device = 2;  // e.g. GPU enum
  prefill.const_map = {{0u, 0u}, {3u, 7u}};
  prefill.payload = "prefill-blob-bytes";

  OpenVinoGlobalGraph::Subgraph decode;
  decode.name = "Partition_1";
  decode.device = 2;
  decode.const_map = {{0u, 7u}};
  decode.payload = std::string("\x00\x00\xFF", 3);  // embedded NULs must survive

  graph.subgraphs[prefill.name] = prefill;
  graph.subgraphs[decode.name] = decode;
  return graph;
}

// Serialize -> Parse reproduces the buffer pool, subgraph topology, const_maps,
// device, and payloads exactly (including embedded NUL bytes).
TEST(GlobalGraphTest, RoundTrip) {
  const OpenVinoGlobalGraph in = MakeSample();
  const std::string blob = in.Serialize();

  ASSERT_TRUE(OpenVinoGlobalGraph::HasMagic(
      reinterpret_cast<const uint8_t*>(blob.data()), blob.size()));

  auto parsed = OpenVinoGlobalGraph::Parse(
      reinterpret_cast<const uint8_t*>(blob.data()), blob.size());
  ASSERT_TRUE(parsed.HasValue());
  const OpenVinoGlobalGraph& out = parsed.Value();

  // Buffer pool.
  ASSERT_EQ(out.buffers.size(), in.buffers.size());
  for (const auto& [id, bytes] : in.buffers) {
    ASSERT_TRUE(out.buffers.count(id));
    EXPECT_EQ(out.buffers.at(id), bytes);
  }
  EXPECT_EQ(out.BankBytes(), in.BankBytes());

  // Subgraphs.
  ASSERT_EQ(out.subgraphs.size(), in.subgraphs.size());
  for (const auto& [name, in_subgraph] : in.subgraphs) {
    ASSERT_TRUE(out.subgraphs.count(name));
    const auto& out_subgraph = out.subgraphs.at(name);
    EXPECT_EQ(out_subgraph.name, in_subgraph.name);
    EXPECT_EQ(out_subgraph.device, in_subgraph.device);
    EXPECT_EQ(out_subgraph.payload, in_subgraph.payload);
    EXPECT_EQ(out_subgraph.const_map, in_subgraph.const_map);
  }
}

// BankBytes sums the deduplicated buffer pool.
TEST(GlobalGraphTest, BankBytesSumsPool) {
  const OpenVinoGlobalGraph graph = MakeSample();
  EXPECT_EQ(graph.BankBytes(), 4u + 10u);
}

// ComputePoolOffsets returns ascending-buffer_id byte offsets that match the
// contiguous pool layout Serialize() writes.
TEST(GlobalGraphTest, ComputePoolOffsetsMatchesLayout) {
  const OpenVinoGlobalGraph graph = MakeSample();
  const auto offsets = graph.ComputePoolOffsets();
  // buffers: id 0 (4 bytes) then id 7 (10 bytes) in ascending id order.
  ASSERT_EQ(offsets.size(), 2u);
  EXPECT_EQ(offsets.at(0), 0u);
  EXPECT_EQ(offsets.at(7), 4u);
}

// ParseHeader locates the pool and each payload by offset/size without copying,
// and the reported offsets address the same bytes the full Parse returns.
TEST(GlobalGraphTest, ParseHeaderLocatesPoolAndPayloads) {
  const OpenVinoGlobalGraph in = MakeSample();
  const std::string blob = in.Serialize();
  const auto* data = reinterpret_cast<const uint8_t*>(blob.data());

  auto header = OpenVinoGlobalGraph::ParseHeader(data, blob.size());
  ASSERT_TRUE(header.HasValue());
  const auto& h = header.Value();

  // Pool region is consistent and holds exactly BankBytes().
  EXPECT_EQ(h.pool_size, in.BankBytes());
  EXPECT_LE(h.pool_data_offset + h.pool_size, blob.size());

  // Buffer index offsets match ComputePoolOffsets, and the bytes at
  // pool_data_offset + pool_offset equal the original buffer bytes.
  const auto offsets = in.ComputePoolOffsets();
  ASSERT_EQ(h.buffer_index.size(), in.buffers.size());
  for (const auto& [id, loc] : h.buffer_index) {
    EXPECT_EQ(loc.pool_offset, offsets.at(static_cast<int32_t>(id)));
    const std::string bytes(
        reinterpret_cast<const char*>(data + h.pool_data_offset +
                                      loc.pool_offset),
        loc.size);
    EXPECT_EQ(bytes, in.buffers.at(id));
  }

  // Payload spans point at the original payload bytes.
  ASSERT_EQ(h.subgraphs.size(), in.subgraphs.size());
  for (const auto& [name, sv] : h.subgraphs) {
    const std::string payload(
        reinterpret_cast<const char*>(data + sv.payload_offset),
        sv.payload_size);
    EXPECT_EQ(payload, in.subgraphs.at(name).payload);
    EXPECT_EQ(sv.const_map, in.subgraphs.at(name).const_map);
  }
}

// ParseHeader rejects a corrupt blob rather than over-reading.
TEST(GlobalGraphTest, ParseHeaderRejectsTruncated) {
  const std::string blob = MakeSample().Serialize();
  auto header = OpenVinoGlobalGraph::ParseHeader(
      reinterpret_cast<const uint8_t*>(blob.data()), blob.size() - 1);
  EXPECT_FALSE(header.HasValue());
}

// An empty container round-trips (magic + zero counts).
TEST(GlobalGraphTest, EmptyRoundTrips) {
  OpenVinoGlobalGraph in;
  const std::string blob = in.Serialize();
  auto parsed = OpenVinoGlobalGraph::Parse(
      reinterpret_cast<const uint8_t*>(blob.data()), blob.size());
  ASSERT_TRUE(parsed.HasValue());
  EXPECT_TRUE(parsed.Value().buffers.empty());
  EXPECT_TRUE(parsed.Value().subgraphs.empty());
}

// HasMagic rejects non-container / short input.
TEST(GlobalGraphTest, HasMagicRejectsBadInput) {
  EXPECT_FALSE(OpenVinoGlobalGraph::HasMagic(nullptr, 0));
  const std::string notmagic = "NOTMAGIC.....";
  EXPECT_FALSE(OpenVinoGlobalGraph::HasMagic(
      reinterpret_cast<const uint8_t*>(notmagic.data()), notmagic.size()));
  const std::string tooshort = "OVG";
  EXPECT_FALSE(OpenVinoGlobalGraph::HasMagic(
      reinterpret_cast<const uint8_t*>(tooshort.data()), tooshort.size()));
}

// Parse errors (does not crash / over-read) on bad magic.
TEST(GlobalGraphTest, ParseRejectsBadMagic) {
  const std::string junk = "not-an-ovglobal-container-blob";
  auto parsed = OpenVinoGlobalGraph::Parse(
      reinterpret_cast<const uint8_t*>(junk.data()), junk.size());
  EXPECT_FALSE(parsed.HasValue());
}

// Parse errors on a truncated (mid-buffer) blob rather than over-reading.
TEST(GlobalGraphTest, ParseRejectsTruncated) {
  const std::string blob = MakeSample().Serialize();
  for (size_t cut : {blob.size() / 2, blob.size() - 1}) {
    auto parsed = OpenVinoGlobalGraph::Parse(
        reinterpret_cast<const uint8_t*>(blob.data()), cut);
    EXPECT_FALSE(parsed.HasValue()) << "cut=" << cut;
  }
}

}  // namespace
}  // namespace openvino
}  // namespace litert
