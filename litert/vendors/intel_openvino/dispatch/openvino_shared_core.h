// Copyright 2025 Google LLC.
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

#ifndef ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_SHARED_CORE_H_
#define ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_SHARED_CORE_H_

#include <cstddef>
#include <memory>
#include <mutex>  // NOLINT
#include <optional>
#include <string>
#include <vector>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

class OpenVINOSharedCore {
 public:
  OpenVINOSharedCore(const OpenVINOSharedCore&) = delete;
  OpenVINOSharedCore(OpenVINOSharedCore&&) = delete;
  OpenVINOSharedCore& operator=(const OpenVINOSharedCore&) = delete;
  OpenVINOSharedCore& operator=(OpenVINOSharedCore&&) = delete;

  static OpenVINOSharedCore* GetInstance();

  // Return the core shared_pointer.
  std::shared_ptr<ov::Core> getCore() const { return core_; }

  void SetDevice(const std::string device) {
    absl::MutexLock lock(&state_mutex_);
    device_ = device;
    remote_context_.reset();
  }
  std::string GetDevice() {
    absl::MutexLock lock(&state_mutex_);
    return device_;
  }

  ov::RemoteContext GetRemoteContext() {
    absl::MutexLock lock(&state_mutex_);
    if (!remote_context_.has_value()) {
      remote_context_ = core_->get_default_context(device_);
    }
    return *remote_context_;
  }

  // Returns the list of OpenVINO devices reported by `core_->
  // get_available_devices()`.  Queried lazily on first call and cached for
  // the lifetime of the process (the set of installed devices does not
  // change at runtime).  Thread-safe.  Returns an empty vector if the
  // underlying query throws.
  const std::vector<std::string>& GetAvailableDevices();

  // NPU weight sharing: stage the deduplicated weight pool into an anonymous
  // in-memory file (memfd) once per process and return its fd. The pool bytes
  // are written at offset 0 so a whole-fd mmap by NPUW yields base == pool
  // start, and each Constant resolves as `mapped->data() + bin_offset`.
  //
  // The returned fd is OWNED by this singleton (closed in the dtor); callers
  // must NOT close it. NPUW invokes the handle provider more than once and
  // close()s whatever it receives, so the provider must hand out a fresh
  // dup(fd) on every call -- never this fd directly. Cached after the first
  // successful call (subsequent calls ignore |data|/|size| and return the
  // same fd). Returns -1 on failure (e.g. memfd_create/ftruncate/pwrite).
  int EnsureBankMemfd(const void* data, size_t size);

 private:
  OpenVINOSharedCore();
  ~OpenVINOSharedCore();

  std::shared_ptr<ov::Core> core_;
  // Guards device_ and remote_context_.
  absl::Mutex state_mutex_;
  std::string device_ ABSL_GUARDED_BY(state_mutex_) = "NPU";  // Default device
  std::optional<ov::RemoteContext> remote_context_
      ABSL_GUARDED_BY(state_mutex_);
  std::once_flag available_devices_once_;
  std::vector<std::string> available_devices_;

  // Anonymous in-memory weight bank (memfd), lazily created by
  // EnsureBankMemfd and owned for the process lifetime. Guarded by bank_mu_.
  std::mutex bank_mu_;
  int bank_memfd_ = -1;
};

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_SHARED_CORE_H_
