// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#ifndef ODML_LITERT_LITERT_VENDORS_OPENVINO_COMPILER_GRAPH_ITERATOR_H_
#define ODML_LITERT_LITERT_VENDORS_OPENVINO_COMPILER_GRAPH_ITERATOR_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "openvino/frontend/tensorflow_lite/decoder.hpp"
#include "openvino/frontend/tensorflow_lite/graph_iterator.hpp"
#include "openvino/frontend/tensorflow_lite/quantization_info.hpp"
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/vendors/intel_openvino/compiler/decoder.h"
namespace litert {
namespace openvino {

struct OVGraphIndices {
  int32_t input_index_ = 0;
  int32_t output_index_ = 0;
  int32_t const_index_ = 0;
  int32_t op_index_ = 0;
};

// GraphIteratorDelegate traverses through the graph/subgraph i/o's and ops.
// Objective of this class is to create TensorMetaInfo structures to pass to
// DecoderTensor to manage I/Os and fill the op specific information
// in DecoderOperation objects. OpenVINO tensorflow lite frontend takes
// the responsibility for creating OV op nodes.
class GraphIteratorDelegate
    : public ov::frontend::tensorflow_lite::GraphIterator {
 public:
  // Weight-source id published on every weight this iterator identifies. The
  // whole LiteRt shared pool is ONE source (one contiguous byte range that NPUW
  // maps in one go), so a single id suffices; it only has to be non-zero, since
  // zero is TensorMetaInfo's "no identity" sentinel (== ov::wsh's
  // invalid_source_id).
  static constexpr std::size_t kWeightSourceId = 1;

  // |weight_bin_offsets| optionally maps LiteRt Weights::BufferId() -> byte
  // offset in the shared weight pool. When non-null, each constant weight whose
  // BufferId is present is handed to the OpenVINO TFLite frontend with a
  // weight-sharing identity (TensorMetaInfo::m_source_id / m_bin_offset). The
  // frontend then builds that weight's ov::op::v0::Constant on a
  // descriptor-carrying, NON-OWNING buffer, which gets us both halves of NPU
  // weight sharing:
  //   - identity: NPUW recovers the pool offset via
  //     ov::weight_sharing::Extension::get_constant_origin(), so the weight is
  //     referenced by offset instead of baked into the blob;
  //   - aliasing: the Constant points AT the pool bytes rather than at a private
  //     memcpy of them, so every partition's Constant for one BufferId shares a
  //     single address, which is what makes NPUW dedup them.
  // Must outlive this iterator. Null (the default) on the GPU and non-shared
  // paths, which keeps the frontend's plain copying behaviour.
  GraphIteratorDelegate(
      const LiteRtCompilerContext* ctx, const litert::compiler::Subgraph* graph,
      std::string device = "NPU",
      const std::map<int32_t, size_t>* weight_bin_offsets = nullptr)
      : ctx_(ctx),
        subgraph_ptr_(graph),
        device_(std::move(device)),
        weight_bin_offsets_(weight_bin_offsets) {
    for (const auto& input : subgraph_ptr_->Inputs()) {
      if (input.IsSubgraphInput()) {
        iterator_indices_.input_index_++;
      } else if (input.IsConstant()) {
        iterator_indices_.const_index_++;
      }
    }
    for (const auto& output : subgraph_ptr_->Outputs()) {
      iterator_indices_.output_index_++;
    }
    for (const auto& op : subgraph_ptr_->Ops()) {
      iterator_indices_.op_index_++;
    }
  }

  ~GraphIteratorDelegate() = default;

  /// \brief Get a number of operation nodes in the graph
  size_t size() const override;

  /// \brief Set iterator to the start position
  void reset() override;

  /// \brief Move to the next node in the graph
  void next() override;

  /// \brief Returns true if iterator goes out of the range of available nodes
  bool is_end() const override;

  /// \brief Return a pointer to a decoder of the current node
  std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase> get_decoder()
      const override;

  /// \brief Returns the number of sub-graphs that can be enumerated with
  /// get_subgraph
  size_t get_subgraph_size() const override { return 0; }

  /// \brief Returns iterator for a subgraph created on demand
  /// If there is no query for specific sub-graph iterator shouldn't be created
  /// idx should be in range 0..get_subgraph_size()-1
  std::shared_ptr<ov::frontend::tensorflow_lite::GraphIterator> get_subgraph(
      size_t idx) const override {
    LITERT_LOG(LITERT_ERROR, "get_subgraph not implemented");
    return nullptr;
  };

  // How many DISTINCT shared buffers this iterator published an identity for,
  // i.e. how many weights of this partition NPUW can resolve from the pool
  // instead of baking in. Always 0 unless constructed with |weight_bin_offsets|.
  // Meaningful once the frontend has walked the whole graph.
  size_t NumIdentifiedWeights() const { return identified_buffer_ids_.size(); }

 private:
  // Rewrites signed i2 weights to unsigned u2 (NPU/CPU path): flips the MSB
  // of every 2-bit element (XOR 0xAA) and shifts the zero points by +2 so
  // the dequantized values are unchanged.  Updates |tensor_meta_info| in
  // place to point at the converted buffer with element type u2.
  void ConvertI2WeightsToU2(
      const litert::compiler::Tensor& tensor,
      ov::frontend::tensorflow_lite::TensorMetaInfo& tensor_meta_info) const;

  // Rewrites signed i2 weights to signed i4 (GPU path): sign-extends each
  // 2-bit element to 4 bits and repacks two elements per byte.  The zero
  // points are left unchanged because i4 represents the same signed values.
  // Updates |tensor_meta_info| in place to point at the converted buffer
  // with element type i4.
  void ConvertI2WeightsToI4(
      const litert::compiler::Tensor& tensor,
      ov::frontend::tensorflow_lite::TensorMetaInfo& tensor_meta_info) const;

  // Publishes |tensor|'s weight-sharing identity (kWeightSourceId + its pool
  // offset) on |tensor_meta_info|, so the frontend emits a descriptor-backed
  // Constant aliasing the pool bytes. No-op unless |weight_bin_offsets_| is set
  // and holds this tensor's BufferId. Must NOT be called for weights whose bytes
  // were rewritten (see ConvertI2Weights*): those no longer match the pool, and
  // the rewritten copy would not outlive the frontend's non-owning reference.
  void TagWeightIdentity(
      const litert::compiler::Tensor& tensor,
      ov::frontend::tensorflow_lite::TensorMetaInfo& tensor_meta_info) const;

  const LiteRtCompilerContext* ctx_;
  size_t node_index_ = 0;
  const litert::compiler::Subgraph* subgraph_ptr_;
  struct OVGraphIndices iterator_indices_;
  // OpenVINO target device string for this partition ("NPU", "GPU", "CPU").
  std::string device_;
  // Owns converted weight buffers for i2 weight transformations. Each entry
  // holds a copy of the repacked bytes (u2 for NPU/CPU, i4 for GPU) that
  // outlives the decoder returned by get_decoder().
  mutable std::vector<std::vector<uint8_t>> converted_weight_buffers_;
  // BufferId -> shared-pool byte offset, or null when this partition is not
  // part of a shared-weights compile. Borrowed; see the constructor comment.
  const std::map<int32_t, size_t>* weight_bin_offsets_ = nullptr;
  // BufferIds handed an identity so far. A set, not a counter: a weight feeding
  // several ops is decoded once per use, and the frontend may walk the graph
  // more than once.
  mutable std::set<int32_t> identified_buffer_ids_;
};

}  // namespace openvino
}  // namespace litert

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_COMPILER_GRAPH_ITERATOR_H_
