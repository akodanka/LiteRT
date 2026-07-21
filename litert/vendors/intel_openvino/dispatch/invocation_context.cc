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

#include "litert/vendors/intel_openvino/dispatch/invocation_context.h"

#include <algorithm>
#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <ios>
#include <istream>
#include <map>
#include <optional>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>  // dup
#endif

#include "openvino/runtime/file_handle.hpp"

#include "openvino/core/any.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/util/tensor_type_util.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/intel_openvino/bytecode_header.h"
#include "litert/vendors/intel_openvino/compiler/global_graph.h"
#include "litert/vendors/intel_openvino/dispatch/weight_bank_runtime.h"

namespace {

// A stable-per-load identity for an OVGLOBAL container, used to key the
// process-singleton buffer-backed host pool (Document 2 §3.2 / R8). Combines
// the container base pointer and size; within one process a given loaded
// container keeps a fixed address, so this dedups the pool across partitions
// and inferences without hashing the (multi-GB) bytes.
inline uint64_t HashContainerIdentity(const uint8_t* data, size_t size) {
  const uint64_t a = reinterpret_cast<uint64_t>(data);
  const uint64_t b = static_cast<uint64_t>(size);
  uint64_t h = a + 0x9e3779b97f4a7c15ULL;
  h ^= b + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  return h;
}

// This class is copied from the OpenVINO codebase with minor modifications
// for Google C++ Style Guide compliance. It wraps a pre-allocated memory
// buffer to provide a std::streambuf interface, enabling zero-copy stream
// reading.
//
// TODO(b/449624371): Remove SharedStreamBuffer once OpenVINO provides a
// public equivalent.
class SharedStreamBuffer : public std::streambuf {
 public:
  SharedStreamBuffer(const char* data, size_t size)
      : data_(data), size_(size), offset_(0) {}
  explicit SharedStreamBuffer(const void* data, size_t size)
      : SharedStreamBuffer(reinterpret_cast<const char*>(data), size) {}

 protected:
  // override std::streambuf methods
  std::streamsize xsgetn(char* s, std::streamsize count) override {
    auto real_count = std::min<std::streamsize>(size_ - offset_, count);
    std::memcpy(s, data_ + offset_, real_count);
    offset_ += real_count;
    return real_count;
  }

  int_type underflow() override {
    return (size_ == offset_) ? traits_type::eof()
                              : traits_type::to_int_type(*(data_ + offset_));
  }

  int_type uflow() override {
    return (size_ == offset_) ? traits_type::eof()
                              : traits_type::to_int_type(*(data_ + offset_++));
  }

  std::streamsize showmanyc() override { return size_ - offset_; }

  pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
    return seekoff(pos, std::ios_base::beg, which);
  }

  pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                   std::ios_base::openmode which) override {
    if (which != std::ios_base::in) {
      return pos_type(off_type(-1));
    }

    size_t new_offset;
    switch (dir) {
      case std::ios_base::beg:
        new_offset = off;
        break;
      case std::ios_base::cur:
        new_offset = offset_ + off;
        break;
      case std::ios_base::end:
        new_offset = size_ + off;
        break;
      default:
        return pos_type(off_type(-1));
    }

    // Check bounds
    if (new_offset > size_) {
      return pos_type(off_type(-1));
    }

    offset_ = new_offset;
    return pos_type(offset_);
  }

  // Non-virtual overload with default argument for backward compatibility
  pos_type seekoff(off_type off, std::ios_base::seekdir dir) {
    return seekoff(off, dir, std::ios_base::in);
  }

 private:
  const char* data_;
  const size_t size_;
  size_t offset_;
};

}  // namespace

litert::Expected<LiteRtDispatchInvocationContextT::Ptr>
LiteRtDispatchInvocationContextT::Create(
    LiteRtDispatchDeviceContextT& device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    const IntelOpenVinoOptions* intel_openvino_opts) {
  // The container (or raw bytecode) starts at base_addr + offset.
  const uint8_t* container =
      static_cast<const uint8_t*>(exec_bytecode_buffer->base_addr) +
      exec_bytecode_buffer->offset;
  const size_t container_size = exec_bytecode_buffer->size;
  const void* exec_bytecode_ptr = container;
  auto exec_bytecode_size = container_size;

  // GlobalGraph weight sharing: the compiler returns ONE container blob for all
  // partitions (magic "OVGLOBAL") holding a shared buffer pool + per-subgraph
  // {payload, const_map, device}. Select THIS partition's subgraph (by
  // function_name, which the plugin sets to the graph name; fall back to the
  // sole/first subgraph). Two consumption strategies over the one container:
  //
  //  * NPU + fd-backed model (this design, §7): parse the header ONLY (no pool
  //    copy) and mmap the pool region straight out of the model file's fd at
  //    import; the payload is imported weightless. Zero weight copies.
  //  * GPU (or NPU without an fd): full Parse (copies the pool) and bind USM
  //    views below.
  //
  // Non-shared models skip this and use the raw bytecode directly.
  std::optional<litert::openvino::OpenVinoGlobalGraph> global_graph;
  std::map<uint32_t, uint32_t> selected_const_map;
  std::optional<LiteRtIntelOpenVinoGraphBackend> selected_device;
  std::vector<uint8_t> selected_payload;
  // NPU fd-backed weightless region (Option B). npu_fd_shared gates it.
  bool npu_fd_shared = false;
  size_t pool_file_offset = 0;
  size_t pool_region_size = 0;
  // NPU buffer-backed weightless host region (Document 2, fd == -1).
  // npu_host_shared gates it; host_region_ptr/size point at the pool in memory,
  // host_region_keepalive keeps those bytes alive for the compiled model.
  bool npu_host_shared = false;
  const void* host_region_ptr = nullptr;
  size_t host_region_size = 0;
  std::shared_ptr<void> host_region_keepalive;
  if (litert::openvino::OpenVinoGlobalGraph::HasMagic(container,
                                                      container_size)) {
    // Header-only parse first: cheap, copies no pool/payload bytes. This is
    // enough to select the subgraph and decide the consumption strategy.
    LITERT_ASSIGN_OR_RETURN(
        litert::openvino::OpenVinoGlobalGraph::HeaderView header,
        litert::openvino::OpenVinoGlobalGraph::ParseHeader(container,
                                                           container_size));

    const litert::openvino::OpenVinoGlobalGraph::HeaderView::SubgraphView*
        selected = nullptr;
    if (function_name != nullptr && function_name[0] != '\0') {
      auto it = header.subgraphs.find(function_name);
      if (it != header.subgraphs.end()) selected = &it->second;
    }
    if (selected == nullptr && !header.subgraphs.empty()) {
      selected = &header.subgraphs.begin()->second;  // fall back to first
    }
    if (selected == nullptr) {
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "GlobalGraph: no subgraph for function_name");
    }
    LITERT_LOG(LITERT_INFO,
               "GlobalGraph: selected subgraph '%s' (%zu byte payload) for "
               "function_name '%s'",
               selected->name.c_str(), selected->payload_size,
               function_name ? function_name : "(null)");
    selected_const_map = selected->const_map;
    selected_device =
        static_cast<LiteRtIntelOpenVinoGraphBackend>(selected->device);
    const std::string selected_device_str =
        litert::openvino::GraphBackendToString(*selected_device);

    // Weight-sharing consumption strategy (design §6). Every NPU branch
    // converges on the SAME NPUW weightless import + descriptor-first identity;
    // they differ only in how the pool bytes reach OpenVINO. Log the chosen
    // branch so a silent copy is never mistaken for zero-copy.
    if (selected_device_str == "NPU" && exec_bytecode_buffer->fd >= 0) {
      // [BEST] Doc 1: mmap the pool sub-region out of the fd (zero copies).
      // The pool's absolute offset in the file is the blob's base in the file
      // + the bytecode start relative to that base + the pool's start within
      // the container.
      npu_fd_shared = true;
      pool_file_offset = exec_bytecode_buffer->alloc_base_file_offset +
                         exec_bytecode_buffer->offset + header.pool_data_offset;
      pool_region_size = header.pool_size;
      exec_bytecode_ptr = container + selected->payload_offset;
      exec_bytecode_size = selected->payload_size;
      LITERT_LOG(LITERT_INFO,
                 "GlobalGraph(NPU,fd): pool_file_offset=%zu pool_size=%zu "
                 "(fd=%d)",
                 pool_file_offset, pool_region_size, exec_bytecode_buffer->fd);
    } else if (selected_device_str == "NPU") {
      // Doc 2: fd == -1. The pool is already resident in the caller's model
      // buffer at container + pool_data_offset. Feed NPUW that host region so
      // weight sharing stays enabled (dedup, single container, descriptor
      // identity) -- only the zero-copy mmap is unavailable.
      npu_host_shared = true;
      const uint8_t* pool_host_ptr = container + header.pool_data_offset;
      const size_t pool_size = header.pool_size;
      exec_bytecode_ptr = container + selected->payload_offset;
      exec_bytecode_size = selected->payload_size;

      // A stable identity for this container, so the per-process host pool is
      // materialized once and reused across partitions/inferences (R8).
      const uint64_t container_id = HashContainerIdentity(container, container_size);

      if (std::getenv("LITERT_OV_WS_HOST_REGION_NONOWNING") != nullptr) {
        // [B1] Zero-copy: point NPUW straight at the caller's buffer. Only safe
        // when the buffer outlives the compiled model; the keep-alive here is a
        // non-owning alias, so this is opt-in.
        host_region_ptr = pool_host_ptr;
        host_region_size = pool_size;
        host_region_keepalive = std::shared_ptr<void>(
            const_cast<void*>(static_cast<const void*>(container)),
            [](void*) {});  // non-owning: caller owns the model buffer
        LITERT_LOG(LITERT_INFO,
                   "GlobalGraph(NPU,buffer,B1): non-owning host region "
                   "ptr=%p size=%zu (zero-copy; caller must keep buffer alive)",
                   pool_host_ptr, pool_size);
      } else {
        // [B2, safe default] One owned copy of the pool into a process
        // singleton keyed by container_id; reused by every partition.
        auto pool = litert::openvino::GetOrMakeSharedHostPool(
            container_id, pool_host_ptr, pool_size);
        host_region_ptr = pool->data();
        host_region_size = pool->size();
        host_region_keepalive = pool;  // owns the bytes -> lifetime guaranteed
        LITERT_LOG(LITERT_INFO,
                   "GlobalGraph(NPU,buffer,B2): owned shared host pool "
                   "ptr=%p size=%zu (one copy, fd unavailable)",
                   host_region_ptr, host_region_size);
      }
    } else {
      // GPU: full Parse copies the pool for the USM bind path (unchanged).
      LITERT_ASSIGN_OR_RETURN(global_graph,
                              litert::openvino::OpenVinoGlobalGraph::Parse(
                                  container, container_size));
      auto it = global_graph->subgraphs.find(selected->name);
      if (it == global_graph->subgraphs.end()) {
        return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                             "GlobalGraph: subgraph vanished after full parse");
      }
      // Copy the selected payload into an owned buffer; we import from it below.
      selected_payload.assign(it->second.payload.begin(),
                              it->second.payload.end());
      exec_bytecode_ptr = selected_payload.data();
      exec_bytecode_size = selected_payload.size();
    }
  }

  // If the compiler embedded a self-describing header, honor the device
  // recorded there.  Per-partition bytecode produced by the LiteRT OpenVINO
  // compiler plugin always carries this header, so each partition can be
  // dispatched to its own target device (NPU/CPU/GPU) even when the
  // model-wide options request a different default.
  std::string device = "NPU";  // Default device
  LiteRtIntelOpenVinoGraphBackend embedded_graph_backend =
      kLiteRtIntelOpenVinoGraphBackendNPU;
  size_t payload_offset = 0;
  bool device_from_header = litert::openvino::TryParseBytecodeHeader(
      exec_bytecode_ptr, exec_bytecode_size, &embedded_graph_backend,
      &payload_offset);
  if (device_from_header) {
    device = litert::openvino::GraphBackendToString(embedded_graph_backend);
    exec_bytecode_ptr =
        static_cast<const uint8_t*>(exec_bytecode_ptr) + payload_offset;
    exec_bytecode_size -= payload_offset;
    LITERT_LOG(LITERT_INFO, "Dispatch: using device '%s' from bytecode header",
               device.c_str());
  } else if (selected_device.has_value()) {
    // GlobalGraph subgraphs carry their target device in the container.
    device = litert::openvino::GraphBackendToString(*selected_device);
    LITERT_LOG(LITERT_INFO, "Dispatch: using device '%s' from GlobalGraph",
               device.c_str());
  } else {
    LITERT_LOG(LITERT_INFO,
               "Dispatch: no bytecode header found, defaulting to '%s'",
               device.c_str());
  }

  // Validate that the requested device is actually available on this system
  // before setting the device.
  auto core = device_context.getCore();
  if (!core) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get OpenVINO core from device context");
  }
  {
    const std::vector<std::string>& available_devices =
        OpenVINOSharedCore::GetInstance()->GetAvailableDevices();

    auto matches = [&device](const std::string& name) {
      if (name == device) return true;
      auto dot = name.find('.');
      return dot != std::string::npos && name.substr(0, dot) == device;
    };

    if (std::none_of(available_devices.begin(), available_devices.end(),
                     matches)) {
      std::string available_list;
      for (const auto& d : available_devices) {
        if (!available_list.empty()) available_list += ", ";
        available_list += d;
      }
      LITERT_LOG(LITERT_ERROR,
                 "Dispatch: requested OpenVINO device '%s' is not available. "
                 "Available devices: [%s]",
                 device.c_str(), available_list.c_str());
      return litert::Error(
          kLiteRtStatusErrorRuntimeFailure,
          "Requested OpenVINO device is not available on this system");
    }
  }
  LITERT_LOG(LITERT_INFO, "Using Intel OpenVINO device: %s", device.c_str());

  OpenVINOSharedCore::GetInstance()->SetDevice(device);

  if (!exec_bytecode_ptr || exec_bytecode_size == 0) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Empty bytecode buffer");
  }

  SharedStreamBuffer membuf(static_cast<const char*>(exec_bytecode_ptr),
                            exec_bytecode_size);
  std::istream model_stream(&membuf);
  if (!model_stream) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to open model bytecode stream");
  }

  // Resolve a GlobalGraph subgraph's weights against the shared pool.
  //   * NPU + fd (npu_fd_shared): import weightless and hand NPUW a handle
  //     provider (dup of the model fd) plus the pool sub-region so NPUW mmaps
  //     the pool in place -- zero weight copies. Constant identity travels on
  //     each Constant's buffer descriptor (ov::weight_sharing), published at
  //     compile time, so no Context crosses the boundary.
  //   * NPU + buffer (npu_host_shared, fd == -1): import weightless and hand
  //     NPUW an already-resident host region (Document 2). Same descriptor
  //     identity/dedup contract; only the weight source differs.
  //   * GPU (gpu_shared): weights are Parameters imported plainly, then bound
  //     to views into a shared USM-host buffer below.
  //   * Non-shared: import the payload directly.
  const bool gpu_shared = global_graph.has_value() && device == "GPU";
  ov::CompiledModel compiled_model;
  try {
    if (npu_fd_shared) {
      // dup() the fd: HandleHolder inside load_mmap_object closes what it is
      // given, and the original fd is owned by the LiteRT model mapping.
#if defined(_WIN32)
      return litert::Error(
          kLiteRtStatusErrorRuntimeFailure,
          "fd-backed NPU weight sharing is not supported on Windows");
#else
      const int model_fd = exec_bytecode_buffer->fd;
      ov::AnyMap import_properties;
      import_properties["NPUW_WEIGHTS_HANDLE_PROVIDER"] =
          ov::FileHandleProvider([model_fd]() -> ov::FileHandle {
            return ::dup(model_fd);
          });
      import_properties["NPUW_WEIGHTS_HANDLE_REGION_OFFSET"] =
          static_cast<std::size_t>(pool_file_offset);
      import_properties["NPUW_WEIGHTS_HANDLE_REGION_SIZE"] =
          static_cast<std::size_t>(pool_region_size);
      import_properties[ov::enable_weightless.name()] = true;
      import_properties["NPU_USE_NPUW"] = "YES";
      compiled_model =
          core->import_model(model_stream, device, import_properties);
#endif
    } else if (npu_host_shared) {
      // Buffer-backed (fd == -1): hand NPUW the host region (ptr/size) and a
      // keep-alive so it wraps the pool as MappedMemory and reads weights
      // straight from host memory. Same weightless import + descriptor-first
      // identity as the fd path; only the weight source differs.
      ov::AnyMap import_properties;
      import_properties["NPUW_WEIGHTS_HOST_REGION_PTR"] =
          reinterpret_cast<std::uintptr_t>(host_region_ptr);
      import_properties["NPUW_WEIGHTS_HOST_REGION_SIZE"] =
          static_cast<std::size_t>(host_region_size);
      import_properties["NPUW_WEIGHTS_HOST_REGION_KEEPALIVE"] =
          host_region_keepalive;
      import_properties[ov::enable_weightless.name()] = true;
      import_properties["NPU_USE_NPUW"] = "YES";
      compiled_model =
          core->import_model(model_stream, device, import_properties);
      LITERT_LOG(LITERT_INFO,
                 "GlobalGraph(NPU,buffer): weightless import complete "
                 "(host region ptr=%p size=%zu)",
                 host_region_ptr, host_region_size);
    } else {
      compiled_model = core->import_model(model_stream, device);
    }
  } catch (const std::exception& e) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure, e.what());
  }

  auto infer_request = compiled_model.create_infer_request();

  // Bind the shared weights onto the infer request and hold the views for its
  // lifetime (set_input_tensor does not take ownership).
  std::vector<ov::Tensor> bound_weights;
  if (gpu_shared) {
    LITERT_ASSIGN_OR_RETURN(
        std::vector<litert::openvino::BoundWeight> bound,
        litert::openvino::BindSharedWeightsGpu(*core, *global_graph,
                                               compiled_model,
                                               selected_const_map));
    bound_weights.reserve(bound.size());
    for (auto& b : bound) {
      infer_request.set_input_tensor(b.input_index, b.view);
      bound_weights.push_back(std::move(b.view));
    }
    LITERT_LOG(LITERT_INFO, "GlobalGraph: bound %zu shared weights on GPU",
               bound_weights.size());
  }

  LITERT_LOG(LITERT_INFO, "Openvino InvocationContext Initialize SUCCESS");
  // TODO: add support for loading cached model
  return Ptr(new LiteRtDispatchInvocationContextT(infer_request, device_context,
                                                  num_inputs, num_outputs,
                                                  std::move(bound_weights)));
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetTensorBufferRequirements(
    const LiteRtRankedTensorType& tensor_type) {
  LiteRtTensorBufferType supported_tensor_buffer_types[] = {
      kLiteRtTensorBufferTypeOpenVINOTensorBuffer,
      // OpenVINO RemoteTensor doesn't support copy-free AHWB buffer. Until
      // it's supported, we use DMA-BUF.
      kLiteRtTensorBufferTypeDmaBuf,
      kLiteRtTensorBufferTypeAhwb,
  };

  int num_supported_tensor_buffer_types =
      sizeof(supported_tensor_buffer_types) /
      sizeof(supported_tensor_buffer_types[0]);
  auto buffer_size = litert::internal::GetNumPackedBytes(tensor_type);
  if (!buffer_size) {
    return litert::Unexpected(buffer_size.Error());
  }

  LiteRtTensorBufferRequirements requirements;
  auto status =
      device_context_.runtime_context()->create_tensor_buffer_requirements(
          num_supported_tensor_buffer_types, supported_tensor_buffer_types,
          *buffer_size, 0, /*strides=*/nullptr, &requirements);
  if (status != kLiteRtStatusOk)
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to get buffer requirements");

  return requirements;
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetInputRequirements(
    int input_index, const LiteRtRankedTensorType& tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetOutputRequirements(
    int output_index, const LiteRtRankedTensorType& tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}

litert::Expected<void> LiteRtDispatchInvocationContextT::AttachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  LITERT_ASSIGN_OR_RETURN(ov::Tensor ov_tensor,
                          device_context_.getOVTensor(tensor_buffer_handle));
  // TODO: visit this if need to maintain graph indices for inputs and outputs
  // in dispatch_api
  infer_request_.set_input_tensor(graph_input_index, ov_tensor);
  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::AttachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  LITERT_ASSIGN_OR_RETURN(ov::Tensor ov_tensor,
                          device_context_.getOVTensor(tensor_buffer_handle));
  // TODO: visit this if need to maintain graph indices for inputs and outputs
  // in dispatch_api
  infer_request_.set_output_tensor(graph_output_index, ov_tensor);
  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::Invoke() {
  infer_request_.start_async();
  if (!infer_request_.wait_for(
          std::chrono::milliseconds(kInferRequestTimeoutMs)))
    return litert::Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        "Failed to execute inference request due to timeout");
  return {};
}
