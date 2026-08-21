// Copyright (C) 2026 Intel Corporation
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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_ALIAS_SHARED_CONSTANTS_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_ALIAS_SHARED_CONSTANTS_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>

#include "openvino/core/model.hpp"
#include "litert/vendors/intel_openvino/compiler/weight_bank.h"

namespace litert::openvino {

// NPU cross-partition weight-sharing transform (counterpart to the GPU
// ConvertWeightsToParameters).
//
// Rebuilds each large bank-backed weight Constant in |ov_model| so its data
// pointer ALIASES the deduplicated pool bytes for its BufferId (one stable host
// pointer shared across every partition), then stamps its
// WeightlessCacheAttribute(bin_offset).
//
// Aliasing is the load-bearing step for NPU weight sharing: NPUW dedups weights
// by their Constant data pointer, so pointing every partition's Constant at the
// one pool buffer is what makes NPUW collapse them to a single allocation.
// bin_offset tells NPUW where to mmap the weight at runtime.
//
// A Constant is aliased ONLY when its current bytes are byte-identical to the
// pool bytes for its BufferId. A mismatch means a content-altering frontend
// transform rewrote this weight, so aliasing would feed the graph the wrong
// data; such weights are left baked/per-partition. Comparing bytes is the
// robust guard -- no need to enumerate which transforms alter content.
//
// |pool_offset_of| maps BufferId -> byte offset in the contiguous shared pool
// (the same ascending-id layout Serialize() uses). |partition_idx| is used only
// for logging.
//
// Returns the number of Constants aliased (i.e. actually shareable). Constants
// with no BufferId or below the element threshold stay baked.
size_t AliasAndTagSharedConstants(
    const std::shared_ptr<ov::Model>& ov_model, const WeightBank& weight_bank,
    const std::map<int32_t, size_t>& pool_offset_of, int partition_idx);

// Gives every consumer of a small multi-use Constant its own Constant node.
//
// WHY: the tflite export CSEs constants across decoder layers by value, so one
// Const node ends up with fan-out into several layers. NPUW's FOLD match bank
// then sees layers whose op multisets differ *only* in Const count and aborts
// with "Number of layers in match bank differs from # of function calls".
// Splitting the node restores uniform layer bodies.
//
// This does NOT cost weight sharing, and that is the whole point: there are two
// ways to share a constant and only one of them upsets FOLD.
//   - one NODE with fan-out N  -> layers differ in Const count. Hostile.
//   - N nodes, one DATA POINTER -> layers uniform, NPUW still dedups to a single
//     allocation. Benign. This is what AliasAndTagSharedConstants produces.
// So this pass converts the first form into the second. Run it BEFORE
// AliasAndTagSharedConstants: clones keep the original's friendly name, so
// BufferIdOfName resolves them all to the same BufferId and they are all
// re-aliased onto the one pool buffer.
//
// Clones share the original's data buffer (Constant's copy ctor shares the
// AlignedBuffer by shared_ptr), so this copies no weight bytes.
//
// |max_bytes| is a safety belt: Constants larger than it are left alone so a
// genuinely large shared weight can never be duplicated. On Gemma-4 12B no
// Constant with fan-out > 1 exceeds 2 KB -- the real MatMul weights all have
// fan-out 1 -- so at 64 KB this excludes nothing today.
//
// Returns the number of clones created.
size_t CloneMultiUseConstants(const std::shared_ptr<ov::Model>& ov_model,
                              size_t max_bytes, int partition_idx);

}  // namespace litert::openvino

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_ALIAS_SHARED_CONSTANTS_H_
