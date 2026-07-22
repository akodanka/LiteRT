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

#include "litert/vendors/intel_openvino//dispatch/openvino_shared_core.h"

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include "openvino/runtime/core.hpp"
#include "litert/c/internal/litert_logging.h"

OpenVINOSharedCore::OpenVINOSharedCore()
    : core_(std::make_shared<ov::Core>()) {}

OpenVINOSharedCore::~OpenVINOSharedCore() {
#if defined(__linux__)
  if (bank_memfd_ >= 0) {
    ::close(bank_memfd_);
    bank_memfd_ = -1;
  }
#endif
}

// static
OpenVINOSharedCore* OpenVINOSharedCore::GetInstance() {
  static OpenVINOSharedCore* instance = new OpenVINOSharedCore();
  return instance;
}

const std::vector<std::string>& OpenVINOSharedCore::GetAvailableDevices() {
  std::call_once(available_devices_once_, [this]() {
    try {
      available_devices_ = core_->get_available_devices();
    } catch (const std::exception&) {
      available_devices_.clear();
    }
  });
  return available_devices_;
}

int OpenVINOSharedCore::EnsureBankMemfd(const void* data, size_t size) {
#if defined(__linux__)
  std::lock_guard<std::mutex> lock(bank_mu_);
  if (bank_memfd_ >= 0) {
    return bank_memfd_;
  }
  // Anonymous, close-on-exec in-memory file (Linux >= 3.17). Backed by tmpfs:
  // no disk I/O, no filename races, no cleanup-on-crash concern.
  int fd = ::memfd_create("litert_ov_bank", MFD_CLOEXEC);
  if (fd < 0) {
    LITERT_LOG(LITERT_ERROR, "EnsureBankMemfd: memfd_create failed");
    return -1;
  }
  // Size must be >= pool_size or NPUW's set_from_fd throws "Requested mapping
  // range exceeds file size".
  if (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
    LITERT_LOG(LITERT_ERROR, "EnsureBankMemfd: ftruncate(%zu) failed", size);
    ::close(fd);
    return -1;
  }
  // Write the whole pool at offset 0 so the memfd base == pool start; then
  // NPUW's whole-fd map resolves each Constant as data() + bin_offset.
  const char* src = static_cast<const char*>(data);
  for (size_t off = 0; off < size;) {
    ssize_t n = ::pwrite(fd, src + off, size - off, static_cast<off_t>(off));
    if (n <= 0) {
      LITERT_LOG(LITERT_ERROR, "EnsureBankMemfd: pwrite failed at offset %zu",
                 off);
      ::close(fd);
      return -1;
    }
    off += static_cast<size_t>(n);
  }
  bank_memfd_ = fd;  // owned; closed in dtor. Callers dup() per use.
  LITERT_LOG(LITERT_INFO, "EnsureBankMemfd: staged %zu bytes into memfd", size);
  return bank_memfd_;
#else
  (void)data;
  (void)size;
  LITERT_LOG(LITERT_ERROR,
             "EnsureBankMemfd: memfd weight bank is only supported on Linux");
  return -1;
#endif
}
