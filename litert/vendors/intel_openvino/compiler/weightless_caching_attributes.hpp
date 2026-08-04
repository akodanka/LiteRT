// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Vendored verbatim from the OpenVINO core dev_api header
//   openvino/src/core/dev_api/openvino/core/rt_info/weightless_caching_attributes.hpp
// (verified against OpenVINO HEAD 0fc77a6da6 / SDK 2026.4.0).
//
// This header is part of OpenVINO's dev_api and is NOT shipped in the runtime
// SDK, so it is vendored here. The corresponding symbols
// (ov::WeightlessCacheAttribute, ov::copy_weightless_cache_attr) ARE exported
// by the shipped libopenvino.so, so no additional source is required. The
// attribute lets us tag each large weight Constant with its byte position in
// the deduplicated weight pool (bin_offset); NPUW's weightless import reads it
// back via get_constant_origin and resolves each constant as
// mapped_weights->data() + bin_offset.

#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov {

OPENVINO_API void copy_weightless_cache_attr(const std::shared_ptr<Node>& from, const std::shared_ptr<Node>& to);

/**
 * @brief Holds weightless caching attributes of a single constant.
 *
 * WeightlessCacheAttribute class represents runtime info attribute that holds
 * the values of original size of the constant in bytes and the binary offset of the
 * constant's data in the weights file used by the weightless caching mechanism. It's
 * not copyable in case the data was changed (the original node was replaced by a new
 * one produced during the tranformation pipeline) - in that case weightless caching
 * can't be used for that constant.
 */
class OPENVINO_API WeightlessCacheAttribute : public RuntimeAttribute {
public:
    OPENVINO_RTTI("WeightlessCacheAttribute", "0", RuntimeAttribute);

    WeightlessCacheAttribute() = delete;

    WeightlessCacheAttribute(size_t original_size, size_t bin_offset, ov::element::Type original_dtype)
        : original_size(original_size),
          bin_offset(bin_offset),
          original_dtype(original_dtype) {}

    bool is_copyable() const override;

    size_t original_size;
    size_t bin_offset;
    ov::element::Type original_dtype;
};

}  // namespace ov
