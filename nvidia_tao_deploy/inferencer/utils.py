# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base utility functions for TensorRT inferencer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import pycuda.autoinit # noqa pylint: disable=unused-import
import pycuda.driver as cuda
import tensorrt as trt


class HostDeviceMem(object):
    """Clean data structure to handle host/device memory."""

    def __init__(self, host_mem, device_mem, npshape, name: str = None):
        """Initialize a HostDeviceMem data structure.

        Args:
            host_mem (cuda.pagelocked_empty): A cuda.pagelocked_empty memory buffer.
            device_mem (cuda.mem_alloc): Allocated memory pointer to the buffer in the GPU.
            npshape (tuple): Shape of the input dimensions.

        Returns:
            HostDeviceMem instance.
        """
        self.host = host_mem
        self.device = device_mem
        self.numpy_shape = npshape
        self.name = name

    def __str__(self):
        """String containing pointers to the TRT Memory."""
        return "Name: " + self.name + "\nHost:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        """Return the canonical string representation of the object."""
        return self.__str__()

def do_inference(context, inputs, outputs, stream, return_raw=False):
    """Inference function for TensorRT 10 using the new tensor I/O API.

    Args:
        context: TensorRT execution context.
        inputs: List of HostDeviceMem input buffers.
        outputs: List of HostDeviceMem output buffers.
        stream: PyCUDA stream.
        return_raw: If True, return HostDeviceMem objects; else return host arrays.

    Returns:
        Inference results as raw buffers or NumPy arrays.
    """
    # Set input tensor addresses and copy input data
    for inp in inputs:
        context.set_tensor_address(inp.name, int(inp.device))
        cuda.memcpy_htod_async(inp.device, inp.host, stream)

    # Set output tensor addresses
    for out in outputs:
        context.set_tensor_address(out.name, int(out.device))

    # Run inference
    context.execute_async_v3(stream_handle=stream.handle)

    # Copy outputs from device to host
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)

    # Wait for all operations to complete
    stream.synchronize()

    return outputs if return_raw else [out.host for out in outputs]

def allocate_buffers(engine, context=None, reshape=False):
    """Allocates host and device buffer for TRT 10 engine inference.

    Args:
        engine (trt.ICudaEngine): TensorRT engine
        context (trt.IExecutionContext): Required for dynamic shape engines
        reshape (bool): Whether to reshape host memory (e.g., for FRCNN)

    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: device addresses for set_tensor_address
        stream (cuda.Stream): CUDA stream for async transfers
    """
    assert context is not None, "TRT 10 requires context to get dynamic tensor shapes"

    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    # Explicit override of expected types for known plugin layers
    binding_to_type = {
        "Input": np.float32,
        "NMS": np.float32,
        "NMS_1": np.int32,
        "BatchedNMS": np.int32,
        "BatchedNMS_1": np.float32,
        "BatchedNMS_2": np.float32,
        "BatchedNMS_3": np.float32,
        "generate_detections": np.float32,
        "mask_head/mask_fcn_logits/BiasAdd": np.float32,
        "softmax_1": np.float32,
        "input_1": np.float32,
        "inputs": np.float32,
        "pred_boxes": np.float32,
        "pred_logits": np.float32,
        "pred_masks": np.float32,
    }

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        is_input = mode == trt.TensorIOMode.INPUT

        # Get shape from the context (dynamic) or engine (static)
        dims = context.get_tensor_shape(name)
        size = trt.volume(dims)
        size = max(size, 1)  # avoid 0-sized outputs like BatchedNMS

        # Determine dtype
        dtype = binding_to_type.get(name, trt.nptype(engine.get_tensor_dtype(name)))

        # Allocate host/device memory
        host_mem = cuda.pagelocked_empty(size, dtype)

        if reshape and not is_input:
            host_mem = host_mem.reshape(*dims)

        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        mem = HostDeviceMem(host_mem, device_mem, dims, name=name)
        if is_input:
            inputs.append(mem)
        else:
            outputs.append(mem)

    return inputs, outputs, bindings, stream
