/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace stream_executor {
class Stream;
}  // namespace stream_executor

namespace tensorflow {

class GPUDeviceContext : public DeviceContext {
 public:
  // Does not take ownership of streams.
  GPUDeviceContext(int stream_id, se::Stream* stream,
#if TENSORFLOW_USE_ROCM
                   se::Stream* nccl_stream,
#endif
                   se::Stream* host_to_device_stream,
                   se::Stream* device_to_host_stream,
                   gtl::InlinedVector<se::Stream*, 4> device_to_device_stream)
      : stream_id_(stream_id),
        stream_(stream),
#if TENSORFLOW_USE_ROCM
        nccl_stream_(nccl_stream),
#endif
        host_to_device_stream_(host_to_device_stream),
        device_to_host_stream_(device_to_host_stream),
        device_to_device_stream_(device_to_device_stream) {
  }

  ~GPUDeviceContext() override {}

  se::Stream* stream() const override { return stream_; }
#if TENSORFLOW_USE_ROCM
  se::Stream* nccl_stream() const { return nccl_stream_; }
#endif
  se::Stream* host_to_device_stream() const { return host_to_device_stream_; }
  se::Stream* device_to_host_stream() const { return device_to_host_stream_; }
  se::Stream* device_to_device_stream(int index) const {
    return device_to_device_stream_[index % device_to_device_stream_.size()];
  }
  int stream_id() const { return stream_id_; }

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done,
                             bool sync_dst_compute) const override;

  void CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
                             Device* device, Tensor* cpu_tensor,
                             StatusCallback done) override;

  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              StatusCallback done) const override;

  void MaintainLifetimeOnStream(const Tensor* t,
                                se::Stream* stream) const override {}

  Status ThenExecute(Device* device, se::Stream* stream,
                     std::function<void()> func) override;

  void StreamUnification() {
#if TENSORFLOW_USE_ROCM
        unification_stored_streams_.emplace_back(nccl_stream_);
        nccl_stream_ = stream_;
#endif
        unification_stored_streams_.emplace_back(host_to_device_stream_);
        host_to_device_stream_ = stream_;
        unification_stored_streams_.emplace_back(device_to_host_stream_);
        device_to_host_stream_ = stream_;
        int d2d_stream_sizes = device_to_device_stream_.size();
        for (int i = 0; i < d2d_stream_sizes; i++ ) {
          unification_stored_streams_.emplace_back(device_to_device_stream_[i]);
          device_to_device_stream_[i] = stream_;
        }
  }

  void ResetStreamUnification() {
#if TENSORFLOW_USE_ROCM
        nccl_stream_ = unification_stored_streams_.front();
        unification_stored_streams_.erase(unification_stored_streams_.begin());
#endif
        host_to_device_stream_ = unification_stored_streams_.front();
        unification_stored_streams_.erase(unification_stored_streams_.begin());
        device_to_host_stream_ = unification_stored_streams_.front();
        unification_stored_streams_.erase(unification_stored_streams_.begin());
        int d2d_stream_sizes = device_to_device_stream_.size();
        for (int i = 0; i < d2d_stream_sizes; i++ ) {
          device_to_device_stream_[i] = unification_stored_streams_.front();
          unification_stored_streams_.erase(unification_stored_streams_.begin());
        }
        unification_stored_streams_.clear();
  }

 private:
  int stream_id_;
  // The default primary stream to use for this context.
  // All the memory belongs to this stream.
  se::Stream* stream_;
#if TENSORFLOW_USE_ROCM
  // The stream to use for nccl operations.
  se::Stream* nccl_stream_;
#endif
  // The stream to use for copying data from host into GPU.
  se::Stream* host_to_device_stream_;
  // The stream to use for copying data from GPU to host.
  se::Stream* device_to_host_stream_;
  // Streams to use for copying data between GPUs.
  gtl::InlinedVector<se::Stream*, 4> device_to_device_stream_;
  std::vector<se::Stream*> unification_stored_streams_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_DEVICE_CONTEXT_H_
