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

#include "litert/vendors/intel_openvino/compiler/openvino_compile_context.h"

#include <cstdlib>
#include <memory>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/runtime/properties.hpp"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/vendors/intel_openvino/compiler/npu_optimizer.h"
#include "litert/vendors/intel_openvino/compiler/openvino_soc_config.h"

namespace litert {
namespace openvino {

OpenVinoCompileContext::OpenVinoCompileContext() {
  configs_map_[ov::hint::performance_mode.name()] =
      ov::hint::PerformanceMode::LATENCY;
}

::litert::Expected<OpenVinoCompileContext> OpenVinoCompileContext::Create(
    const ::litert::Expected< ::litert::intel_openvino::IntelOpenVinoOptions>&
        opts,
    int graph_index) {
  OpenVinoCompileContext context;
  if (!opts.HasValue()) {
    LITERT_LOG(LITERT_INFO, "Using default configuration (LATENCY mode)");
    return context;
  }
  const auto& options = opts.Value();

  // Resolve the OpenVINO target device for this graph from the per-graph
  // graph_backend override.  When no override is set, fall back to NPU.
  LiteRtIntelOpenVinoGraphBackend graph_backend =
      kLiteRtIntelOpenVinoGraphBackendNPU;
  if (graph_index >= 0) {
    auto override = options.GetGraphBackend(graph_index);
    if (override.HasValue()) {
      graph_backend = *override;
      LITERT_LOG(LITERT_INFO, "Graph %d: graph_backend -> %d", graph_index,
                 graph_backend);
    }
  }
  switch (graph_backend) {
    case kLiteRtIntelOpenVinoGraphBackendCPU:
      context.device_ = "CPU";
      break;
    case kLiteRtIntelOpenVinoGraphBackendGPU:
      context.device_ = "GPU";
      break;
    case kLiteRtIntelOpenVinoGraphBackendNPU:
      context.device_ = "NPU";
      break;
    case kLiteRtIntelOpenVinoGraphBackendMax:
      context.device_ = "NPU";
      break;
  }

  LITERT_LOG(LITERT_INFO, "Using Intel OpenVINO device: %s",
             context.device_.c_str());

  auto performance_mode = options.GetPerformanceMode();

  // Add custom configuration options.
  int num_custom_options = options.GetNumConfigsMapOptions();
  for (int i = 0; i < num_custom_options; ++i) {
    auto [key, value] = options.GetConfigsMapOption(i);
    if (!key.empty()) {
      if (key == "optimize_fq_after_matmul") {
        LITERT_LOG(LITERT_INFO, "Custom config: optimize_fq_after_matmul = %s",
                   value.c_str());
        context.eliminate_fq_after_matmul_ = (value == "true");
        continue;
      }
      if (key == "fuse_split_attention_to_sdpa") {
        LITERT_LOG(LITERT_INFO,
                   "Custom config: fuse_split_attention_to_sdpa = %s",
                   value.c_str());
        context.fuse_split_attention_to_sdpa_ = (value == "true");
        continue;
      }
      if (key == "sdpa_pad_kv_to_alignment") {
        LITERT_LOG(LITERT_INFO, "Custom config: sdpa_pad_kv_to_alignment = %s",
                   value.c_str());
        context.sdpa_pad_kv_to_alignment_ = (value == "true");
        continue;
      }
      context.configs_map_[key] = value;
      LITERT_LOG(LITERT_INFO, "Custom config: %s = %s", key.c_str(),
                 value.c_str());
    }
  }

  // Configure performance mode (can be overridden by custom options).
  switch (performance_mode) {
    case kLiteRtIntelOpenVinoPerformanceModeLatency:
      if (context.configs_map_.find(ov::hint::performance_mode.name()) ==
          context.configs_map_.end()) {
        context.configs_map_[ov::hint::performance_mode.name()] =
            ov::hint::PerformanceMode::LATENCY;
        LITERT_LOG(LITERT_INFO, "Performance mode: LATENCY");
      }
      break;
    case kLiteRtIntelOpenVinoPerformanceModeThroughput:
      if (context.configs_map_.find(ov::hint::performance_mode.name()) ==
          context.configs_map_.end()) {
        context.configs_map_[ov::hint::performance_mode.name()] =
            ov::hint::PerformanceMode::THROUGHPUT;
        LITERT_LOG(LITERT_INFO, "Performance mode: THROUGHPUT");
      }
      break;
    case kLiteRtIntelOpenVinoPerformanceModeCumulativeThroughput:
      if (context.configs_map_.find(ov::hint::performance_mode.name()) ==
          context.configs_map_.end()) {
        context.configs_map_[ov::hint::performance_mode.name()] =
            ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
        LITERT_LOG(LITERT_INFO, "Performance mode: CUMULATIVE_THROUGHPUT");
      }
      break;
  }

  // Apply per-graph configs map overrides on top of the model-wide configs.
  if (graph_index >= 0) {
    int num_graph_configs = options.GetNumGraphConfigsMapOptions(graph_index);
    for (int i = 0; i < num_graph_configs; ++i) {
      auto [key, value] = options.GetGraphConfigsMapOption(graph_index, i);
      if (key.empty()) continue;
      if (key == "optimize_fq_after_matmul") {
        context.eliminate_fq_after_matmul_ = (value == "true");
        continue;
      }
      context.configs_map_[key] = value;
      LITERT_LOG(LITERT_INFO, "Graph %d custom config: %s = %s", graph_index,
                 key.c_str(), value.c_str());
    }
  }

  return context;
}

LiteRtStatus OpenVinoCompileContext::ConfigureForSoc(const char* soc_model) {
  if (device_ == "NPU") {
    // NPU weight sharing (gated on the same LITERT_OV_EMBED_WEIGHTS=1 env the
    // compiler plugin uses to build the shared OVGLOBAL container): turn on the
    // NPUW + weightless-cache knobs so export_model emits a *weightless* blob
    // whose Constant records carry the bin_offset we stamp via
    // WeightlessCacheAttribute. Without these the export bakes weights and the
    // dispatch-time memfd handle provider has nothing to resolve against. The
    // default (non-shared) path is untouched.
    const char* embed_env = std::getenv("LITERT_OV_EMBED_WEIGHTS");
    if (embed_env != nullptr && embed_env[0] == '1') {
      auto set_if_unset = [&](const char* key, const char* value) {
        if (configs_map_.find(key) == configs_map_.end()) {
          configs_map_[key] = value;
        }
      };
      set_if_unset("NPU_USE_NPUW", "YES");
      set_if_unset("NPUW_DEVICES", "NPU");
      set_if_unset("NPUW_WEIGHTS_BANK", "shared");
      // NPUW_CWAI ("Closures/Weights As Inputs") is what makes export_model
      // emit a weightless blob keyed by our stamped bin_offsets.
      set_if_unset("NPUW_CWAI", "YES");
      set_if_unset("NPUW_FUNCALL_FOR_ALL", "YES");
      LITERT_LOG(LITERT_INFO,
                 "Weight-sharing NPUW config enabled (NPU_USE_NPUW, "
                 "NPUW_DEVICES=NPU, NPUW_WEIGHTS_BANK=shared, NPUW_CWAI=YES, "
                 "NPUW_FUNCALL_FOR_ALL=YES)");
    }
    return litert::openvino::ConfigureCompilationParams(soc_model,
                                                        configs_map_);
  }
  return kLiteRtStatusOk;
}

void OpenVinoCompileContext::OptimizeModel(
    const std::shared_ptr<ov::Model>& model) const {
  if (device_ == "NPU") {
    NpuOptimizer()
        .SetEliminateMatMulFakeQuantize(eliminate_fq_after_matmul_)
        .SetFuseSplitAttentionToSDPA(fuse_split_attention_to_sdpa_)
        .SetSdpaPadKvToAlignment(sdpa_pad_kv_to_alignment_)
        .Run(model);
  }
}

}  // namespace openvino
}  // namespace litert
