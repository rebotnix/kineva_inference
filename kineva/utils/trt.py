"""
Kineva Inference Engine
Copyright (C) Rebotnix

Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
For license details, see: https://www.gnu.org/licenses/agpl-3.0.html
Project website: https://rebotnix.com
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(trt_file_path):
    with open(trt_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine, mode="anomaly"):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    if mode == "anomaly":
        n_tensors = engine.num_io_tensors
        for i in range(n_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(shape, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if name == "query_image" or name == "prompt_image":
                inputs.append({'name': name, 'host': host_mem, 'device': device_mem, 'shape': shape, 'dtype': dtype})
            else:
                outputs.append({'name': name, 'host': host_mem, 'device': device_mem, 'shape': shape, 'dtype': dtype})
        
    elif mode == "ultralytics" or mode=="rfdetr" or mode=="kineva":
        for binding in engine:
            binding_shape = engine.get_tensor_shape(binding)
            dtype = trt.nptype(engine.get_tensor_dtype(binding))
            size = trt.volume(binding_shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                #if engine.binding_is_input(binding):
                inputs.append({'name': binding, 'host': host_mem, 'device': device_mem, 'shape': binding_shape})
            else:
                outputs.append({'name': binding, 'host': host_mem, 'device': device_mem, 'shape': binding_shape})
        

    return inputs, outputs, bindings, stream

def infer(context, inputs, outputs, stream, mode="anomaly"):
    if mode=="anomaly":
        # Copy inputs to device asynchronously
        for inp in inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], stream)

        # Set input shapes by binding name if dynamic
        for inp in inputs:
            context.set_input_shape(inp['name'], inp['shape'])

        # Set input tensor addresses
        for inp in inputs:
            context.set_tensor_address(inp['name'], int(inp['device']))

        # Set output tensor addresses
        for out in outputs:
            context.set_tensor_address(out['name'], int(out['device']))

        # Run inference
        context.execute_async_v3(stream_handle=stream.handle)

        # Copy outputs back to host asynchronously
        for out in outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)

        # Synchronize the stream
        stream.synchronize()

        # Return reshaped outputs
        return [out['host'].reshape(out['shape']) for out in outputs]
    
    elif mode == "ultralytics" or mode == "rfdetr" or mode == "kineva":
        # Copy input host -> device and set tensor addresses
        for inp in inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
            context.set_tensor_address(inp['name'], int(inp['device']))

        # Set output tensor addresses
        for out in outputs:
            context.set_tensor_address(out['name'], int(out['device']))

        # Run inference (no bindings argument here)
        context.execute_async_v3(stream_handle=stream.handle)

        # Copy output device -> host
        for out in outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)

        stream.synchronize()

        # Return outputs reshaped
        return [out['host'].reshape(out['shape']) for out in outputs]