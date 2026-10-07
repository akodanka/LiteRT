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

#include "litert/vendors/intel_openvino/dispatch/weight_bank_runtime.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/intel_openvino/compiler/global_graph.h"

// Temp-file staging is platform-split. POSIX uses mkstemp, which creates the
// file atomically (O_CREAT|O_EXCL, mode 0600); std::filesystem has no
// equivalent, so the portable spelling would be an exists()-then-open probe
// loop that races. Keeping <filesystem> out of the ELF build also keeps
// libstdc++'s vague-linkage std::filesystem::path symbols out of this .so,
// where a differently-built module on the loader path (OpenVINO's libs) could
// interpose them. MSVC / clang-cl emit COFF and have no such interposition, so
// Windows keeps std::filesystem for temp-dir discovery and cleanup, and uses
// CreateFile(CREATE_NEW) for the atomic create.
#ifdef LITERT_WINDOWS_OS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// <windows.h> must precede the CRT headers below on some toolchains.
#include <io.h>

#include <cstdio>
#include <filesystem>  // NOLINT
#include <random>
#include <system_error>  // NOLINT
#else
#include <errno.h>
#include <stdlib.h>  // mkstemp: POSIX, not declared by <cstdlib>.
#include <unistd.h>
#if defined(__linux__)
#include <sys/sendfile.h>
#endif
#endif

// copy_file_range(2): glibc >= 2.27, bionic >= API 34. Evaluated after the libc
// headers above so __GLIBC__ / __ANDROID_API__ are visible.
#if defined(__linux__)
#if defined(__ANDROID__)
#if defined(__ANDROID_API__) && __ANDROID_API__ >= 34
#define LITERT_HAVE_COPY_FILE_RANGE 1
#endif
#elif defined(__GLIBC__) && \
    (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 27))
#define LITERT_HAVE_COPY_FILE_RANGE 1
#endif
#endif
#ifndef LITERT_HAVE_COPY_FILE_RANGE
#define LITERT_HAVE_COPY_FILE_RANGE 0
#endif

namespace litert::openvino {
namespace {

// Size of one user-space copy step (pointer fallback and pread/ReadFile
// bounce buffer). Small enough that the bounce buffer is negligible next to
// the pool, large enough to keep syscall overhead low. Also keeps every single
// write under the 32-bit count limits of WriteFile / the MSVC CRT.
constexpr size_t kChunkBytes = size_t{16} << 20;  // 16 MiB

// Outcome of one staging attempt. |fatal| means no other strategy can succeed
// (destination full, source shorter than the pool span, ...), so the caller
// stops instead of falling through.
struct CopyStatus {
  bool ok = false;
  bool fatal = false;
  std::string error;  // human-readable, for logging
};

#ifdef LITERT_WINDOWS_OS

std::string WinErrorString(DWORD err) {
  char buf[256];
  const DWORD len = ::FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, err,
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), buf, sizeof(buf), nullptr);
  std::string msg = len > 0 ? std::string(buf, len) : std::string();
  while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r' ||
                          msg.back() == ' ' || msg.back() == '.')) {
    msg.pop_back();
  }
  if (msg.empty()) msg = "unknown error";
  return msg + " (" + std::to_string(err) + ")";
}

// Chunked WriteFile of [data, data+size) at |out|'s current position. Returns
// 0 on success, else the failing GetLastError().
DWORD WriteAllWin(HANDLE out, const void* data, size_t size) {
  const char* p = static_cast<const char*>(data);
  while (size > 0) {
    const DWORD want = static_cast<DWORD>(std::min(size, kChunkBytes));
    DWORD wrote = 0;
    if (!::WriteFile(out, p, want, &wrote, nullptr)) return ::GetLastError();
    if (wrote == 0) return ERROR_WRITE_FAULT;
    p += wrote;
    size -= wrote;
  }
  return 0;
}

// Copies [in_off, in_off+size) of the model file behind |in_fd| to |out| via a
// kChunkBytes bounce buffer. Positional reads (OVERLAPPED offset) so the copy
// never touches the model's file mapping; the pool pages therefore never enter
// this process's working set. The handle behind |in_fd| is dup'd from the
// model's, and ReadFile on a synchronous handle moves the shared file pointer,
// so it is saved and restored around the copy.
CopyStatus CopyFdToBankWin(int in_fd, uint64_t in_off, size_t size,
                           HANDLE out) {
  CopyStatus st;
  const HANDLE in = reinterpret_cast<HANDLE>(_get_osfhandle(in_fd));
  if (in == INVALID_HANDLE_VALUE) {
    st.error = "_get_osfhandle(" + std::to_string(in_fd) + ") failed";
    return st;
  }
  LARGE_INTEGER zero{};
  LARGE_INTEGER saved{};
  const bool have_saved = ::SetFilePointerEx(in, zero, &saved, FILE_CURRENT);

  std::vector<char> buf(kChunkBytes);
  size_t done = 0;
  while (done < size) {
    const DWORD want = static_cast<DWORD>(std::min(size - done, kChunkBytes));
    const uint64_t pos = in_off + done;
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(pos & 0xFFFFFFFFULL);
    ov.OffsetHigh = static_cast<DWORD>(pos >> 32);
    DWORD got = 0;
    if (!::ReadFile(in, buf.data(), want, &got, &ov)) {
      const DWORD err = ::GetLastError();
      st.fatal = (err == ERROR_HANDLE_EOF);
      st.error = "ReadFile at offset " + std::to_string(pos) + " failed: " +
                 WinErrorString(err);
      break;
    }
    if (got == 0) {
      st.fatal = true;
      st.error = "model file ended at offset " + std::to_string(pos) +
                 ", before the end of the pool span";
      break;
    }
    if (const DWORD err = WriteAllWin(out, buf.data(), got); err != 0) {
      st.fatal = true;  // destination-side; would recur on any strategy
      st.error = "WriteFile failed: " + WinErrorString(err);
      break;
    }
    done += got;
  }
  if (have_saved) ::SetFilePointerEx(in, saved, nullptr, FILE_BEGIN);
  st.ok = (done == size);
  return st;
}

#else  // POSIX

#if defined(__linux__)
using FileOff = off64_t;
#define LITERT_PREAD ::pread64
#define LITERT_PWRITE ::pwrite64
#else
using FileOff = off_t;
#define LITERT_PREAD ::pread
#define LITERT_PWRITE ::pwrite
#endif

// mkstemp needs the directory spelled out in its template, so mirror what
// std::filesystem::temp_directory_path() probes. Android has no /tmp, so
// staging there requires TMPDIR to be set -- unchanged from the previous
// std::filesystem spelling, which failed the same way.
std::string PosixTempDir() {
  for (const char* var : {"TMPDIR", "TMP", "TEMP", "TEMPDIR"}) {
    if (const char* value = std::getenv(var);
        value != nullptr && value[0] != '\0') {
      return value;
    }
  }
  return "/tmp";
}

// Destination-side errors recur whatever strategy is used to read the source;
// don't waste time (or log noise) falling through on them.
bool IsDestinationErrno(int err) {
  switch (err) {
    case ENOSPC:
    case EFBIG:
    case EIO:
#ifdef EDQUOT
    case EDQUOT:
#endif
      return true;
    default:
      return false;
  }
}

// Positional, chunked write of [data, data+size) to |fd| at |offset|. A single
// write(2) is capped at 0x7ffff000 bytes on Linux and can be cut short by a
// signal, so the multi-GB pool needs this loop: a short write is not an error.
// Returns 0 on success, else the failing errno.
int PwriteAll(int fd, const void* data, size_t size, uint64_t offset) {
  const char* p = static_cast<const char*>(data);
  while (size > 0) {
    const size_t want = std::min(size, kChunkBytes);
    const ssize_t written =
        LITERT_PWRITE(fd, p, want, static_cast<FileOff>(offset));
    if (written < 0) {
      if (errno == EINTR) continue;
      return errno;
    }
    p += written;
    size -= static_cast<size_t>(written);
    offset += static_cast<uint64_t>(written);
  }
  return 0;
}

// Copies [in_off, in_off+size) of |in_fd| to [0, size) of |out_fd| without
// faulting the model mapping: kernel-side copy_file_range, then sendfile, then
// a pread/pwrite loop through a kChunkBytes bounce buffer. Each fallback is
// logged with the reason. Progress carries across strategies, so a strategy
// that fails mid-way is resumed, not restarted, by the next one. Only explicit
// offsets are used, so the file position shared with the model's dup'd fd is
// never moved.
CopyStatus CopyFdToBankPosix(int in_fd, uint64_t in_off, int out_fd,
                             size_t size, const char** strategy) {
  CopyStatus st;
  size_t done = 0;

  auto source_truncated = [&](uint64_t pos) {
    st.fatal = true;
    st.error = "model file ended at offset " + std::to_string(pos) +
               ", before the end of the pool span";
  };

#if LITERT_HAVE_COPY_FILE_RANGE
  {
    *strategy = "copy_file_range";
    int err = 0;
    while (done < size) {
      FileOff src = static_cast<FileOff>(in_off + done);
      FileOff dst = static_cast<FileOff>(done);
      // Per-call count is capped by the kernel (0x7ffff000); a short copy is
      // not an error.
      const size_t want = std::min(size - done, size_t{1} << 30);
      const ssize_t n = ::copy_file_range(in_fd, &src, out_fd, &dst, want, 0);
      if (n < 0) {
        if (errno == EINTR) continue;
        err = errno;
        break;
      }
      if (n == 0) {
        source_truncated(in_off + done);
        return st;
      }
      done += static_cast<size_t>(n);
    }
    if (done == size) {
      st.ok = true;
      return st;
    }
    if (IsDestinationErrno(err)) {
      st.fatal = true;
      st.error = std::string("copy_file_range failed: ") + std::strerror(err);
      return st;
    }
    LITERT_LOG(LITERT_WARNING,
               "NPU weight sharing: copy_file_range failed after %zu/%zu "
               "bytes (%s); falling back to sendfile",
               done, size, std::strerror(err));
  }
#endif  // LITERT_HAVE_COPY_FILE_RANGE

#if defined(__linux__)
  {
    *strategy = "sendfile";
    int err = 0;
    // sendfile writes at out_fd's file position (no out offset parameter);
    // |out_fd| is private to this bank, so moving it is safe.
    if (::lseek64(out_fd, static_cast<off64_t>(done), SEEK_SET) < 0) {
      err = errno;
    }
    while (err == 0 && done < size) {
      off64_t src = static_cast<off64_t>(in_off + done);
      const size_t want = std::min(size - done, size_t{1} << 30);
      const ssize_t n = ::sendfile64(out_fd, in_fd, &src, want);
      if (n < 0) {
        if (errno == EINTR) continue;
        err = errno;
        break;
      }
      if (n == 0) {
        source_truncated(in_off + done);
        return st;
      }
      done += static_cast<size_t>(n);
    }
    if (done == size) {
      st.ok = true;
      return st;
    }
    if (IsDestinationErrno(err)) {
      st.fatal = true;
      st.error = std::string("sendfile failed: ") + std::strerror(err);
      return st;
    }
    LITERT_LOG(LITERT_WARNING,
               "NPU weight sharing: sendfile failed after %zu/%zu bytes (%s); "
               "falling back to pread/pwrite",
               done, size, std::strerror(err));
  }
#endif  // __linux__

  *strategy = "pread/pwrite";
  std::vector<char> buf(kChunkBytes);
  while (done < size) {
    const size_t want = std::min(size - done, kChunkBytes);
    const ssize_t n = LITERT_PREAD(in_fd, buf.data(), want,
                                   static_cast<FileOff>(in_off + done));
    if (n < 0) {
      if (errno == EINTR) continue;
      st.error = std::string("pread failed: ") + std::strerror(errno);
      return st;
    }
    if (n == 0) {
      source_truncated(in_off + done);
      return st;
    }
    if (const int err = PwriteAll(out_fd, buf.data(), static_cast<size_t>(n),
                                  done);
        err != 0) {
      st.fatal = IsDestinationErrno(err);
      st.error = std::string("pwrite failed: ") + std::strerror(err);
      return st;
    }
    done += static_cast<size_t>(n);
  }
  st.ok = true;
  return st;
}

#undef LITERT_PREAD
#undef LITERT_PWRITE
#endif  // LITERT_WINDOWS_OS

}  // namespace

litert::Expected<std::vector<BoundWeight>> GpuSharedBank::Bind(
    ov::Core& core, const OpenVinoGlobalGraph& global_graph,
    const ov::CompiledModel& compiled_model,
    const std::map<std::string, uint32_t>& const_map) {
  ov::RemoteContext ctx = core.get_default_context("GPU");

  // Hold the lock across the whole body: the first partition allocates+fills
  // the pool, later partitions read base_/offset_. Serializes concurrent
  // partitions driving the same model (this bank is owned per device context /
  // per model).
  absl::MutexLock lock(gpu_bank_mutex_);
  if (!bank_ready_) {
    size_t total = 0;
    for (const auto& entry : global_graph.buffers) total += entry.bytes.size();
    // Allocate the pool as one usm-host buffer of bytes; per-weight views
    // reinterpret slices as the weight's element type below.
    gpu_usm_ =
        ctx.create_tensor(ov::element::u8, ov::Shape{total},
                          {{ov::intel_gpu::shared_mem_type.name(),
                            ov::intel_gpu::SharedMemType::USM_HOST_BUFFER}});
    base_ =
        gpu_usm_.get_params().at(ov::intel_gpu::mem_handle.name()).as<void*>();
    auto* base = static_cast<uint8_t*>(base_);
    size_t off = 0;
    // TODO(PR #8745 #7): recheck USM view alignment.
    for (const auto& entry : global_graph.buffers) {  // ascending buffer_id
      std::memcpy(base + off, entry.bytes.data(), entry.bytes.size());
      weight_offset_[entry.id] = off;
      off += entry.bytes.size();
    }
    bank_ready_ = true;
    LITERT_LOG(LITERT_INFO,
               "GlobalGraph: allocated shared usm-host bank (%zu bytes)",
               total);
  }

  auto* base = static_cast<uint8_t*>(base_);
  std::vector<BoundWeight> bound;
  const auto& inputs = compiled_model.inputs();
  for (size_t p = 0; p < inputs.size(); ++p) {
    // Match by the input's friendly_name (set at compile when the weight was
    // promoted to a Parameter), not by port index -- import_model may reorder
    // inputs, so positional keying is unsafe.
    const auto it = const_map.find(inputs[p].get_node()->get_friendly_name());
    if (it == const_map.end()) {
      continue;  // real activation input, not a shared weight
    }
    const auto off_it = weight_offset_.find(it->second);
    if (off_it == weight_offset_.end()) {
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "GlobalGraph: const_map buffer_id not in pool");
    }
    // Wrap the slice of the shared usm-host buffer as a remote tensor view of
    // the weight's own element type (zero-copy: same USM pointer, no reorder).
    ov::RemoteTensor view =
        ctx.create_tensor(inputs[p].get_element_type(), inputs[p].get_shape(),
                          {{ov::intel_gpu::shared_mem_type.name(),
                            ov::intel_gpu::SharedMemType::USM_USER_BUFFER},
                           {ov::intel_gpu::mem_handle.name(),
                            static_cast<ov::intel_gpu::gpu_handle_param>(
                                base + off_it->second)}});
    bound.push_back(BoundWeight{p, std::move(view)});
  }
  return bound;
}

NpuSharedBank::~NpuSharedBank() {
  // The device context outlives every compiled model imported from this bank,
  // so NPUW's mmap of the file is already gone by the time this runs.
  std::string path;
  {
    absl::MutexLock lock(npu_bank_mutex_);
    if (!owns_bank_file_) return;
    path = bank_path_;
  }
  if (path.empty()) return;
#ifdef LITERT_WINDOWS_OS
  std::error_code ec;
  std::filesystem::remove(std::filesystem::path(path), ec);
#else
  ::unlink(path.c_str());
#endif
}

std::string NpuSharedBank::Path() const {
  absl::MutexLock lock(npu_bank_mutex_);
  return bank_path_;
}

std::string NpuSharedBank::EnsureOnDisk(const PoolSource& src) {
  absl::MutexLock lock(npu_bank_mutex_);
  if (!bank_path_.empty()) return bank_path_;  // write once, reuse

  const void* data = src.data;
  const size_t size = src.size;

  // Guard against a null/empty pool: writing 0 bytes yields an empty file that
  // NPUW would mmap to a zero-length region, so data()+bin_offset is undefined.
  if (data == nullptr || size == 0) {
    LITERT_LOG(LITERT_ERROR,
               "EnsureOnDisk: refusing to stage null/empty pool "
               "(data=%p size=%zu)",
               data, size);
    return {};
  }

  // Deployment override: point at a pre-staged bank and skip the write.
  if (const char* override_path = std::getenv("LITERT_OV_WEIGHTS_PATH");
      override_path != nullptr && override_path[0] != '\0') {
    bank_path_ = override_path;
    LITERT_LOG(LITERT_INFO,
               "NPU weight sharing: using LITERT_OV_WEIGHTS_PATH override '%s'",
               bank_path_.c_str());
    return bank_path_;
  }

  // Prefer copying from the model file itself: that never touches the model's
  // mapping, so the pool pages never become resident in this process. Without
  // an fd the only source is the in-memory view, which faults every page of
  // the pool into the model mapping for the lifetime of the LiteRtModel.
  const bool have_fd = src.fd >= 0;
  if (!have_fd) {
    LITERT_LOG(LITERT_WARNING,
               "NPU weight sharing: model is not file-backed (fd=%d); staging "
               "%zu-byte pool from memory, which faults it into RSS",
               src.fd, size);
  }
  const char* strategy = "write(ptr)";

#ifdef LITERT_WINDOWS_OS
  std::error_code ec;
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path(ec);
  if (ec) {
    LITERT_LOG(LITERT_ERROR,
               "NPU weight sharing: no temp directory available: %s",
               ec.message().c_str());
    return {};
  }

  // No mkstemp on Windows; probe random names in the system temp dir.
  // CREATE_NEW makes the probe atomic (fails with ERROR_FILE_EXISTS rather
  // than clobbering a concurrently created file).
  std::random_device rd;  // NOLINT
  std::filesystem::path file_path;
  HANDLE out = INVALID_HANDLE_VALUE;
  for (int attempt = 0; attempt < 64 && out == INVALID_HANDLE_VALUE;
       ++attempt) {
    const uint64_t r =
        (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
    char name[40];
    std::snprintf(name, sizeof(name), "litert_ov_bank_%016llx.bin",
                  static_cast<unsigned long long>(r));
    file_path = tmp_dir / name;
    out = ::CreateFileW(file_path.c_str(), GENERIC_WRITE, /*dwShareMode=*/0,
                        /*lpSecurityAttributes=*/nullptr, CREATE_NEW,
                        FILE_ATTRIBUTE_NORMAL, /*hTemplateFile=*/nullptr);
    if (out == INVALID_HANDLE_VALUE) {
      const DWORD err = ::GetLastError();
      if (err != ERROR_FILE_EXISTS) {
        LITERT_LOG(LITERT_ERROR,
                   "NPU weight sharing: failed to create temp file '%s': %s",
                   file_path.string().c_str(), WinErrorString(err).c_str());
        return {};
      }
    }
  }
  if (out == INVALID_HANDLE_VALUE) {
    LITERT_LOG(LITERT_ERROR,
               "NPU weight sharing: could not find a free temp file name");
    return {};
  }

  CopyStatus st;
  if (have_fd) {
    strategy = "ReadFile/WriteFile";
    st = CopyFdToBankWin(src.fd, src.file_offset, size, out);
    if (!st.ok && !st.fatal) {
      LITERT_LOG(LITERT_WARNING,
                 "NPU weight sharing: fd copy (%s) failed: %s; falling back "
                 "to staging from memory, which faults the pool into RSS",
                 strategy, st.error.c_str());
      // Restart from byte 0: the pointer path rewrites the whole file.
      LARGE_INTEGER zero{};
      if (!::SetFilePointerEx(out, zero, nullptr, FILE_BEGIN) ||
          !::SetEndOfFile(out)) {
        st.fatal = true;
        st.error = "failed to rewind temp file: " +
                   WinErrorString(::GetLastError());
      }
    }
  }
  if (!st.ok && !st.fatal) {
    strategy = "write(ptr)";
    if (const DWORD err = WriteAllWin(out, data, size); err != 0) {
      st.error = "WriteFile failed: " + WinErrorString(err);
    } else {
      st.ok = true;
    }
  }
  // Treat a close error as a write error: flush failures can surface only
  // here.
  if (st.ok && !::CloseHandle(out)) {
    st.ok = false;
    st.error = "CloseHandle failed: " + WinErrorString(::GetLastError());
  } else if (!st.ok) {
    ::CloseHandle(out);
  }
  if (!st.ok) {
    LITERT_LOG(LITERT_ERROR,
               "NPU weight sharing: failed to write temp file '%s' via %s: %s",
               file_path.string().c_str(), strategy, st.error.c_str());
    std::filesystem::remove(file_path, ec);
    return {};
  }

  bank_path_ = file_path.string();
#else
  std::string name_template = PosixTempDir();
  if (name_template.empty() || name_template.back() != '/') {
    name_template.push_back('/');
  }
  name_template += "litert_ov_bank_XXXXXX";
  // mkstemp rewrites the XXXXXX in place and creates the file atomically, so
  // two processes can never end up staging into the same bank, and mode 0600
  // keeps the pool unreadable by other users of a shared temp dir.
  std::vector<char> path_buf(name_template.begin(), name_template.end());
  path_buf.push_back('\0');
  const int fd = ::mkstemp(path_buf.data());
  if (fd < 0) {
    LITERT_LOG(LITERT_ERROR, "NPU weight sharing: mkstemp('%s') failed: %s",
               name_template.c_str(), std::strerror(errno));
    return {};
  }
  const std::string file_path(path_buf.data());

  CopyStatus st;
  if (have_fd) {
    st = CopyFdToBankPosix(src.fd, src.file_offset, fd, size, &strategy);
    if (!st.ok && !st.fatal) {
      LITERT_LOG(LITERT_WARNING,
                 "NPU weight sharing: fd copy (%s) failed: %s; falling back "
                 "to staging from memory, which faults the pool into RSS",
                 strategy, st.error.c_str());
      // Restart from byte 0: the pointer path rewrites the whole file.
      if (::ftruncate(fd, 0) != 0) {
        st.fatal = true;
        st.error =
            std::string("ftruncate failed: ") + std::strerror(errno);
      }
    }
  }
  if (!st.ok && !st.fatal) {
    strategy = "write(ptr)";
    if (const int err = PwriteAll(fd, data, size, /*offset=*/0); err != 0) {
      st.error = std::string("pwrite failed: ") + std::strerror(err);
    } else {
      st.ok = true;
    }
  }
  // Close even when the copy failed, and treat a close error as a write
  // error: ENOSPC / EIO can surface only here.
  if (::close(fd) != 0 && st.ok) {
    st.ok = false;
    st.error = std::string("close failed: ") + std::strerror(errno);
  }
  if (!st.ok) {
    LITERT_LOG(LITERT_ERROR,
               "NPU weight sharing: failed to write temp file '%s' via %s: %s",
               file_path.c_str(), strategy, st.error.c_str());
    ::unlink(file_path.c_str());
    return {};
  }

  bank_path_ = file_path;
#endif  // LITERT_WINDOWS_OS
  owns_bank_file_ = true;
  LITERT_LOG(LITERT_INFO,
             "NPU weight sharing: staged %zu-byte weights bank to '%s' via %s "
             "(source: %s, file offset %llu)",
             size, bank_path_.c_str(), strategy,
             have_fd ? "model fd" : "memory",
             static_cast<unsigned long long>(have_fd ? src.file_offset : 0));
  return bank_path_;
}

}  // namespace litert::openvino
