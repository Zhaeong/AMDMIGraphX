/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/gpu/kernel.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/gpu/pack_args.hpp>
#include <cassert>
#include <migraphx/half.hpp>

#ifdef _WIN32
#include <hip/hip_ext.h>
#else
// extern declare the function since hip/hip_ext.h header is broken
extern hipError_t hipExtModuleLaunchKernel(hipFunction_t, // NOLINT
                                           uint32_t,
                                           uint32_t,
                                           uint32_t,
                                           uint32_t,
                                           uint32_t,
                                           uint32_t,
                                           size_t,
                                           hipStream_t,
                                           void**,
                                           void**,
                                           hipEvent_t = nullptr,
                                           hipEvent_t = nullptr,
                                           uint32_t   = 0);
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

extern std::string hip_error(int error);

using hip_module_ptr = MIGRAPHX_MANAGE_PTR(hipModule_t, hipModuleUnload);

struct kernel_impl
{
    hip_module_ptr module = nullptr;
    hipFunction_t fun     = nullptr;
};

static hip_module_ptr load_module(const char* image)
{
    hipModule_t raw_m;
    auto status = hipModuleLoadData(&raw_m, image);
    hip_module_ptr m{raw_m};
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to load module: " + hip_error(status));
    return m;
}

kernel::kernel(const char* image, const std::string& name) : impl(std::make_shared<kernel_impl>())
{
    impl->module = load_module(image);
    auto status  = hipModuleGetFunction(&impl->fun, impl->module.get(), name.c_str());
    if(hipSuccess != status)
        MIGRAPHX_THROW("Failed to get function: " + name + ": " + hip_error(status));
}

bool kernel::empty() const { return impl == nullptr; }

static void launch_kernel(hipFunction_t fun,
                          hipStream_t stream,
                          std::size_t global,
                          std::size_t local,
                          void* kernargs,
                          std::size_t size,
                          hipEvent_t start,
                          hipEvent_t stop)
{
    assert(global > 0);
    assert(local > 0);
    void* config[] = {
// HIP_LAUNCH_PARAM_* are macros that do horrible things
#ifdef MIGRAPHX_USE_CLANG_TIDY
        nullptr, kernargs, nullptr, &size, nullptr
#else
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        kernargs,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &size,
        HIP_LAUNCH_PARAM_END
#endif
    };

    auto status = hipExtModuleLaunchKernel(fun,
                                           global,
                                           1,
                                           1,
                                           local,
                                           1,
                                           1,
                                           0,
                                           stream,
                                           nullptr,
                                           reinterpret_cast<void**>(&config),
                                           start,
                                           stop);
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to launch kernel: " + hip_error(status));
    if(stop != nullptr)
    {
        status = hipEventSynchronize(stop);
        if(status != hipSuccess)
            MIGRAPHX_THROW("Failed to sync event: " + hip_error(status));
    }
}

void kernel::launch(hipStream_t stream,
                    std::size_t global,
                    std::size_t local,
                    pointers args,
                    hipEvent_t start,
                    hipEvent_t stop) const
{
    assert(impl != nullptr);
    void* kernargs   = reinterpret_cast<void*>(args.data());
    std::size_t size = args.bytes();

    if (args.size() == 5)
    {
        using migraphx::half;

        size_t size_bytes = 1 * sizeof(half);

        std::vector<half> in_q{half{1.2}};
        std::vector<half> in_k{half{1.2}};
        std::vector<half> in_v{half{1.2}};

        // in_q
        half* d_q_in{};
        hipMalloc(&d_q_in, size_bytes);
        hipMemcpy(d_q_in, in_q.data(), size_bytes, hipMemcpyHostToDevice);
        void** dd_q_ptr = nullptr;
        hipMalloc(&dd_q_ptr, sizeof(void*));
        hipMemcpy(dd_q_ptr, &d_q_in, sizeof(void*), hipMemcpyHostToDevice);

        // in_k
        half* d_k_in{};
        hipMalloc(&d_k_in, size_bytes);
        hipMemcpy(d_k_in, in_k.data(), size_bytes, hipMemcpyHostToDevice);
        void** dd_k_ptr = nullptr;
        hipMalloc(&dd_k_ptr, sizeof(void*));
        hipMemcpy(dd_k_ptr, &d_k_in, sizeof(void*), hipMemcpyHostToDevice);

        // in_v
        half* d_v_in{};
        hipMalloc(&d_v_in, size_bytes);
        hipMemcpy(d_v_in, in_v.data(), size_bytes, hipMemcpyHostToDevice);
        void** dd_v_ptr = nullptr;
        hipMalloc(&dd_v_ptr, sizeof(void*));
        hipMemcpy(dd_v_ptr, &d_v_in, sizeof(void*), hipMemcpyHostToDevice);

        // output
        half* d_out{};
        hipMalloc(&d_out, size_bytes);
        void** dd_ptr = nullptr;
        hipMalloc(&dd_ptr, sizeof(void*));
        hipMemcpy(dd_ptr, &d_out, sizeof(void*), hipMemcpyHostToDevice);

        // other inputs
        constexpr int q_sequence_length  = 1;
        constexpr int kv_sequence_lenght = 1;
        constexpr int head_dim           = 2;
        constexpr int batch_size         = 2;
        constexpr int q_head_num         = 2;
        constexpr int kv_head_num        = 2;
        constexpr float scale            = 0.158114f;

        size_t offset  = 0;
        char new_k_args[256] = {};

        *(reinterpret_cast<void***>(&new_k_args[offset])) = dd_q_ptr;
        offset += sizeof(dd_q_ptr);
        *(reinterpret_cast<void***>(&new_k_args[offset])) = dd_k_ptr;
        offset += sizeof(dd_k_ptr);
        *(reinterpret_cast<void***>(&new_k_args[offset])) = dd_v_ptr;
        offset += sizeof(dd_v_ptr);

        *(reinterpret_cast<void***>(&new_k_args[offset])) = dd_ptr;
        offset += sizeof(dd_ptr);

        *(reinterpret_cast<int*>(&new_k_args[offset])) = q_sequence_length;
        offset += sizeof(q_sequence_length);
        *(reinterpret_cast<int*>(&new_k_args[offset])) = kv_sequence_lenght;
        offset += sizeof(kv_sequence_lenght);
        *(reinterpret_cast<int*>(&new_k_args[offset])) = head_dim;
        offset += sizeof(head_dim);
        *(reinterpret_cast<int*>(&new_k_args[offset])) = head_dim;
        offset += sizeof(head_dim);
        *(reinterpret_cast<int*>(&new_k_args[offset])) = batch_size;
        offset += sizeof(batch_size);
        *(reinterpret_cast<int*>(&new_k_args[offset])) = q_head_num;
        offset += sizeof(q_head_num);
        *(reinterpret_cast<int*>(&new_k_args[offset])) = kv_head_num;
        offset += sizeof(kv_head_num);
        *(reinterpret_cast<float*>(&new_k_args[offset])) = scale;
        offset += sizeof(scale);

        launch_kernel(impl->fun, stream, global, local, new_k_args, offset, start, stop);
    }
    else
    {
        launch_kernel(impl->fun, stream, global, local, kernargs, size, start, stop);
    }
}

void kernel::launch(hipStream_t stream,
                    std::size_t global,
                    std::size_t local,
                    const std::vector<kernel_argument>& args,
                    hipEvent_t start,
                    hipEvent_t stop) const
{
    assert(impl != nullptr);
    std::vector<char> kernargs = pack_args(args);
    std::size_t size           = kernargs.size();

    launch_kernel(impl->fun, stream, global, local, kernargs.data(), size, start, stop);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
