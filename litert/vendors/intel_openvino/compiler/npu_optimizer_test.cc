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

#include "litert/vendors/intel_openvino/compiler/npu_optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include <gtest/gtest.h>

namespace litert {
namespace openvino {
namespace {

// Builds the "split-cache" attention pattern emitted by the LiteRT generative
// converter (gemma4 prefill/decode), matching the layout observed in the
// model:
//   K_cache: [B,H,S_past,D]   K_slice: [B,H,1,D]
//   V_cache: [B,H,D,S_past]   V_slice: [B,H,D,1]   (transposed, adj_y=true)
//   scores = Concat[ MatMul(Q,K_cache,T), MatMul(Q,K_slice,T) ] + mask
//   probs  = Softmax(scores)
//   out    = Add[ MatMul(Slice(probs,past),V_cache,T),
//                 MatMul(Slice(probs,cur),V_slice,T) ]
// |s_past| is the cached sequence length and |s_cur| is the current chunk size
// (the K/V slice length): decode uses s_cur=1, prefill uses s_cur=chunk (e.g.
// 128). The query length equals s_cur in both cases.
// |mask_tile| > 1 makes the mask a Concat of that many copies of a shorter mask
// parameter along the query axis, which is how the exported graph broadcasts one
// [.., S, S_kv] mask across a query length that has GQA groups folded into it.
// |lq| must be divisible by |mask_tile|.
std::shared_ptr<ov::Model> BuildSplitCacheAttention(int64_t s_past = 8,
                                                    int64_t s_cur = 1,
                                                    int64_t mask_tile = 1) {
  using ov::op::v0::Concat;
  using ov::op::v0::Constant;
  using ov::op::v0::MatMul;
  using ov::op::v0::Parameter;
  using ov::op::v1::Add;
  using ov::op::v8::Slice;
  using ov::op::v8::Softmax;

  constexpr int64_t kBatch = 1;
  constexpr int64_t kHeads = 8;
  constexpr int64_t kDim = 256;
  const int64_t lq = s_cur;
  const int64_t s_kv = s_past + s_cur;
  const auto f = ov::element::f32;

  auto q = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), static_cast<size_t>(kHeads),
                   static_cast<size_t>(lq), static_cast<size_t>(kDim)});
  auto k_cache = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), static_cast<size_t>(kHeads),
                   static_cast<size_t>(s_past), static_cast<size_t>(kDim)});
  auto k_slice = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), static_cast<size_t>(kHeads),
                   static_cast<size_t>(s_cur), static_cast<size_t>(kDim)});
  auto v_cache = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), static_cast<size_t>(kHeads),
                   static_cast<size_t>(kDim), static_cast<size_t>(s_past)});
  auto v_slice = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), static_cast<size_t>(kHeads),
                   static_cast<size_t>(kDim), static_cast<size_t>(s_cur)});
  auto mask = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), 1,
                   static_cast<size_t>(lq / mask_tile),
                   static_cast<size_t>(s_kv)});
  ov::Output<ov::Node> mask_out = mask;
  if (mask_tile > 1) {
    mask_out = std::make_shared<Concat>(
        ov::OutputVector(static_cast<size_t>(mask_tile), mask_out),
        /*axis=*/2);
  }

  // scores = Q * K^T (transpose_b), concatenated over the sequence axis.
  auto qk_cache = std::make_shared<MatMul>(q, k_cache, /*transpose_a=*/false,
                                           /*transpose_b=*/true);
  auto qk_slice = std::make_shared<MatMul>(q, k_slice, false, true);
  auto scores = std::make_shared<Concat>(ov::OutputVector{qk_cache, qk_slice},
                                         /*axis=*/-1);
  auto masked = std::make_shared<Add>(scores, mask_out);
  auto probs = std::make_shared<Softmax>(masked, /*axis=*/-1);

  // Split probs back into past / current, then probs * V^T (transpose_b).
  auto start0 = Constant::create(ov::element::i64, ov::Shape{1}, {0});
  auto stop0 = Constant::create(ov::element::i64, ov::Shape{1}, {s_past});
  auto step = Constant::create(ov::element::i64, ov::Shape{1}, {1});
  auto axis = Constant::create(ov::element::i64, ov::Shape{1}, {-1});
  auto slice_past = std::make_shared<Slice>(probs, start0, stop0, step, axis);

  auto start1 = Constant::create(ov::element::i64, ov::Shape{1}, {s_past});
  auto stop1 = Constant::create(ov::element::i64, ov::Shape{1}, {s_kv});
  auto slice_cur = std::make_shared<Slice>(probs, start1, stop1, step, axis);

  auto pv_cache = std::make_shared<MatMul>(slice_past, v_cache, false, true);
  auto pv_slice = std::make_shared<MatMul>(slice_cur, v_slice, false, true);
  auto out = std::make_shared<Add>(pv_cache, pv_slice);

  auto result = std::make_shared<ov::op::v0::Result>(out);
  return std::make_shared<ov::Model>(
      ov::ResultVector{result},
      ov::ParameterVector{q, k_cache, k_slice, v_cache, v_slice, mask},
      "split_cache_attention");
}

// Two independent split-cache attention blocks sharing one mask tensor, which
// is how consecutive decoder layers of the same kind appear in the exported
// model. Each block keeps its own K/V cache parameters: the fusion deliberately
// refuses to touch a cache that has more than one consumer.
std::shared_ptr<ov::Model> BuildTwoBlockSharedMask(int64_t s_past,
                                                   int64_t s_cur,
                                                   int64_t mask_tile) {
  using ov::op::v0::Concat;
  using ov::op::v0::Constant;
  using ov::op::v0::MatMul;
  using ov::op::v0::Parameter;
  using ov::op::v1::Add;
  using ov::op::v8::Slice;
  using ov::op::v8::Softmax;

  constexpr size_t kHeads = 8;
  constexpr size_t kDim = 256;
  const auto lq = static_cast<size_t>(s_cur);
  const int64_t s_kv = s_past + s_cur;
  const auto f = ov::element::f32;
  const auto i64 = ov::element::i64;

  auto mask = std::make_shared<Parameter>(
      f, ov::Shape{1, 1, static_cast<size_t>(s_cur / mask_tile),
                   static_cast<size_t>(s_kv)});
  ov::Output<ov::Node> mask_out = mask;
  if (mask_tile > 1) {
    mask_out = std::make_shared<Concat>(
        ov::OutputVector(static_cast<size_t>(mask_tile), mask_out), /*axis=*/2);
  }

  ov::ParameterVector params{mask};
  ov::ResultVector results;
  for (int block = 0; block < 2; ++block) {
    auto q = std::make_shared<Parameter>(f, ov::Shape{1, kHeads, lq, kDim});
    auto k_cache = std::make_shared<Parameter>(
        f, ov::Shape{1, kHeads, static_cast<size_t>(s_past), kDim});
    auto k_slice = std::make_shared<Parameter>(
        f, ov::Shape{1, kHeads, static_cast<size_t>(s_cur), kDim});
    auto v_cache = std::make_shared<Parameter>(
        f, ov::Shape{1, kHeads, kDim, static_cast<size_t>(s_past)});
    auto v_slice = std::make_shared<Parameter>(
        f, ov::Shape{1, kHeads, kDim, static_cast<size_t>(s_cur)});

    auto scores = std::make_shared<Concat>(
        ov::OutputVector{std::make_shared<MatMul>(q, k_cache, false, true),
                         std::make_shared<MatMul>(q, k_slice, false, true)},
        /*axis=*/-1);
    auto probs = std::make_shared<Softmax>(
        std::make_shared<Add>(scores, mask_out), /*axis=*/-1);

    auto step = Constant::create(i64, ov::Shape{1}, {1});
    auto axis = Constant::create(i64, ov::Shape{1}, {-1});
    auto past = std::make_shared<Slice>(
        probs, Constant::create(i64, ov::Shape{1}, {0}),
        Constant::create(i64, ov::Shape{1}, {s_past}), step, axis);
    auto cur = std::make_shared<Slice>(
        probs, Constant::create(i64, ov::Shape{1}, {s_past}),
        Constant::create(i64, ov::Shape{1}, {s_kv}), step, axis);
    auto out = std::make_shared<Add>(
        std::make_shared<MatMul>(past, v_cache, false, true),
        std::make_shared<MatMul>(cur, v_slice, false, true));

    params.insert(params.end(), {q, k_cache, k_slice, v_cache, v_slice});
    results.push_back(std::make_shared<ov::op::v0::Result>(out));
  }
  return std::make_shared<ov::Model>(results, params, "two_block_shared_mask");
}

template <typename T>
size_t CountOps(const std::shared_ptr<ov::Model>& model) {
  size_t n = 0;
  for (const auto& node : model->get_ops()) {
    if (std::dynamic_pointer_cast<T>(node)) {
      ++n;
    }
  }
  return n;
}

std::shared_ptr<ov::op::v13::ScaledDotProductAttention> FindSdpa(
    const std::shared_ptr<ov::Model>& model) {
  for (const auto& node : model->get_ops()) {
    if (auto sdpa =
            std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(
                node)) {
      return sdpa;
    }
  }
  return nullptr;
}

// Fills |tensor| with deterministic pseudo-random values in [-1, 1).
void FillRandom(ov::Tensor tensor, uint32_t seed) {
  auto* data = tensor.data<float>();
  // Simple LCG so the test is self-contained and reproducible.
  uint64_t state = seed * 2654435761u + 1u;
  for (size_t i = 0; i < tensor.get_size(); ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    data[i] = static_cast<float>((state >> 40) & 0xFFFF) / 32768.0f - 1.0f;
  }
}

// Negative: when the K_cache (or V_cache) tensor is consumed by something
// outside this attention block — typical of layers that share a single KV
// cache across heads/layers — fusing in place would rewrite what the other
// consumer sees. The pass must skip such blocks.
TEST(FuseSplitAttentionToSDPATest, DoesNotFuseSharedKvCache) {
  auto model = BuildSplitCacheAttention(/*s_past=*/8, /*s_cur=*/8);
  // BuildSplitCacheAttention orders parameters as
  //   {q, k_cache, k_slice, v_cache, v_slice, mask}.
  // Add an extra Result that consumes k_cache directly, so K_cache now feeds
  // two inputs (the qk_cache MatMul and this Result) — HasSingleConsumer
  // returns false and the matcher must bail out.
  auto k_cache = model->get_parameters()[1];
  model->add_results({std::make_shared<ov::op::v0::Result>(k_cache)});

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(model);

  EXPECT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(model), 0u);
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(model), 4u);
}

TEST(FuseSplitAttentionToSDPATest, FusesSplitCachePattern) {
  auto model = BuildSplitCacheAttention(/*s_past=*/8, /*s_cur=*/8);

  // Precondition: 4 MatMuls, 1 Softmax, no SDPA.
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(model), 4u);
  EXPECT_EQ(CountOps<ov::op::v8::Softmax>(model), 1u);
  EXPECT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(model), 0u);

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(model);

  // Postcondition: the four attention MatMuls and the Softmax are gone,
  // replaced by exactly one ScaledDotProductAttention op.
  EXPECT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(model), 1u);
  EXPECT_EQ(CountOps<ov::op::v8::Softmax>(model), 0u);
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(model), 0u);
}

// Fuses a copy of |reference| and checks it computes the same thing on the same
// inputs. The reference pads nothing, so this validates the whole alignment
// scheme: zero-padded K/V plus a mask biased at the padded key positions.
void ExpectFusedMatchesReference(const std::shared_ptr<ov::Model>& reference) {
  // Deep-copy before mutating so we can run both versions on identical inputs.
  auto fused = reference->clone();

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(fused);
  ASSERT_NE(FindSdpa(fused), nullptr) << "fusion did not fire";

  ov::Core core;
  auto ref_compiled = core.compile_model(reference, "CPU");
  auto fused_compiled = core.compile_model(fused, "CPU");
  auto ref_req = ref_compiled.create_infer_request();
  auto fused_req = fused_compiled.create_infer_request();

  // Same input tensors fed to both. Inputs are ordered as constructed:
  // {q, k_cache, k_slice, v_cache, v_slice, mask}.
  const size_t num_inputs = reference->inputs().size();
  ASSERT_EQ(num_inputs, fused->inputs().size());
  std::vector<ov::Tensor> inputs;
  for (size_t i = 0; i < num_inputs; ++i) {
    const auto& port = reference->input(i);
    ov::Tensor t(port.get_element_type(), port.get_shape());
    FillRandom(t, static_cast<uint32_t>(i + 1));
    inputs.push_back(t);
    ref_req.set_input_tensor(i, t);
    fused_req.set_input_tensor(i, t);
  }

  ref_req.infer();
  fused_req.infer();

  auto ref_out = ref_req.get_output_tensor(0);
  auto fused_out = fused_req.get_output_tensor(0);
  ASSERT_EQ(ref_out.get_size(), fused_out.get_size());
  const auto* a = ref_out.data<float>();
  const auto* b = fused_out.data<float>();
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref_out.get_size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::abs(a[i] - b[i]));
    ASSERT_FALSE(std::isnan(b[i])) << "fused output has NaN at " << i;
  }
  EXPECT_LT(max_abs_diff, 1e-4f)
      << "fused output diverges from split-cache reference";
}

// Decode shape (s_past=7, s_cur=1): merged KV is 8, so it is padded to 16, and
// the V_cache [B,H,D,S] / V_slice [B,H,D,1] transpose path is exercised.
TEST(FuseSplitAttentionToSDPATest, NumericallyMatchesSplitCache) {
  ExpectFusedMatchesReference(
      BuildSplitCacheAttention(/*s_past=*/7, /*s_cur=*/1));
}

// Same, with the mask supplied as a 4x tile of a shorter mask (merged KV 11 ->
// padded to 16). Covers the tile push-through: padding the end of the key axis
// commutes with the Concat that tiles the query axis, so padding the tile source
// and re-tiling must be indistinguishable from padding the tiled mask.
TEST(FuseSplitAttentionToSDPATest, NumericallyMatchesWithTiledMask) {
  ExpectFusedMatchesReference(
      BuildSplitCacheAttention(/*s_past=*/7, /*s_cur=*/4, /*mask_tile=*/4));
}

// The KV alignment pad must sit on the current-step branch *inside* the Concat,
// not on the Concat output. It then copies one step of KV instead of the whole
// cache, and -- the reason this is asserted rather than left to taste -- NPUW's
// attn::SDPA isolation pattern allows only Unsqueeze/Broadcast/Reshape between
// the KV Concat and the SDPA, so a Pad in that position makes the layer
// invisible to NPUW_ONLINE_ISOLATE=ATTN.
TEST(FuseSplitAttentionToSDPATest, PadsCurrentKvBranchNotMergedKv) {
  constexpr int64_t kSPast = 7;  // merged KV = 8, padded up to 16
  constexpr int64_t kSCur = 1;
  constexpr size_t kAligned = 16;
  auto model = BuildSplitCacheAttention(kSPast, kSCur);

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(model);

  auto sdpa = FindSdpa(model);
  ASSERT_NE(sdpa, nullptr) << "fusion did not fire";

  // K must reach the SDPA straight from the Concat.
  auto k_concat = std::dynamic_pointer_cast<ov::op::v0::Concat>(
      sdpa->input_value(1).get_node_shared_ptr());
  ASSERT_NE(k_concat, nullptr)
      << "K input of SDPA is "
      << sdpa->input_value(1).get_node()->get_type_name() << ", expected Concat";

  // V may only pass through the layout Transpose (its cache is [B,H,D,S]).
  auto v_transpose = std::dynamic_pointer_cast<ov::op::v1::Transpose>(
      sdpa->input_value(2).get_node_shared_ptr());
  ASSERT_NE(v_transpose, nullptr)
      << "V input of SDPA is "
      << sdpa->input_value(2).get_node()->get_type_name();
  EXPECT_NE(std::dynamic_pointer_cast<ov::op::v0::Concat>(
                v_transpose->input_value(0).get_node_shared_ptr()),
            nullptr)
      << "a node other than the Concat feeds the V Transpose";

  // The pad is the Concat's last input and extends only the current step.
  auto k_pad = std::dynamic_pointer_cast<ov::op::v1::Pad>(
      k_concat->input_value(1).get_node_shared_ptr());
  ASSERT_NE(k_pad, nullptr) << "current-step K is not the padded branch";
  EXPECT_EQ(k_pad->get_output_shape(0)[2], kAligned - kSPast)
      << "pad covers more than the current step";
  EXPECT_EQ(k_concat->get_output_shape(0)[2], kAligned);
}

// Consecutive layers of the same kind read the same mask and pad it by the same
// amount, so the pass must build one padded mask and share it -- and must pad
// the tile source, which is |mask_tile| times smaller than the tiled mask.
TEST(FuseSplitAttentionToSDPATest, SharesOnePaddedMaskAcrossBlocks) {
  constexpr int64_t kSPast = 7;  // merged KV = 11, padded up to 16
  constexpr int64_t kSCur = 4;
  constexpr int64_t kTile = 4;
  auto model = BuildTwoBlockSharedMask(kSPast, kSCur, kTile);

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(model);

  ASSERT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(model), 2u);
  // 2 blocks x (K pad + V pad), plus exactly ONE shared mask pad.
  EXPECT_EQ(CountOps<ov::op::v1::Pad>(model), 5u);

  std::vector<std::shared_ptr<ov::op::v13::ScaledDotProductAttention>> sdpas;
  for (const auto& node : model->get_ops()) {
    if (auto s = std::dynamic_pointer_cast<
            ov::op::v13::ScaledDotProductAttention>(node)) {
      sdpas.push_back(s);
    }
  }
  ASSERT_EQ(sdpas.size(), 2u);
  EXPECT_EQ(sdpas[0]->input_value(3), sdpas[1]->input_value(3))
      << "each block built its own padded mask instead of sharing one";

  auto mask_tile = std::dynamic_pointer_cast<ov::op::v0::Concat>(
      sdpas[0]->input_value(3).get_node_shared_ptr());
  ASSERT_NE(mask_tile, nullptr) << "mask tile Concat was not rebuilt";
  auto mask_pad = std::dynamic_pointer_cast<ov::op::v1::Pad>(
      mask_tile->input_value(0).get_node_shared_ptr());
  ASSERT_NE(mask_pad, nullptr) << "the pad was applied to the tiled mask";
  EXPECT_EQ(mask_pad->get_output_shape(0)[2],
            static_cast<size_t>(kSCur / kTile))
      << "the pad is on the tiled mask, not the tile source";
  EXPECT_EQ(mask_pad->get_output_shape(0)[3], 16u);
}

}  // namespace
}  // namespace openvino
}  // namespace litert
