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
//#include <migraphx/gpu/half.hpp>

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
    if(args.size() == 5)
    {
         using migraphx::half;

         size_t size_bytes = 16384 * sizeof(half);

         std::vector<half> in_q{half{2.0}};
         std::vector<half> in_k{half{3.0}};
         std::vector<half> in_v{half{4.0}};

        //size_t size_bytes = 1 * sizeof(half_float::half);
        //std::vector<half_float::half> in_q(size);
        //std::vector<half_float::half> in_k(size);
        //std::vector<half_float::half> in_v(size);

        // std::vector<half_float::half> h_out(size);

        in_q[0] = 2.0f;
        in_k[0] = 3.0f;
        in_v[0] = 4.0f;

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

        constexpr int batch_size      = 1;
        constexpr int sequence_length = 16384;
        constexpr int head_num        = 8;
        constexpr int head_dim        = 40;
        constexpr float scale         = 0.353553f;

        size_t offset        = 0;
        char new_k_args[256] = {};

        *(reinterpret_cast<void***>(&new_k_args[offset])) = dd_q_ptr;
        offset += sizeof(dd_q_ptr);
        *(reinterpret_cast<void***>(&new_k_args[offset])) = dd_k_ptr;
        offset += sizeof(dd_k_ptr);
        *(reinterpret_cast<void***>(&new_k_args[offset])) = dd_v_ptr;
        offset += sizeof(dd_v_ptr);

        *(reinterpret_cast<void***>(&new_k_args[offset])) = dd_ptr;
        offset += sizeof(dd_ptr);

        //*(reinterpret_cast<void***>(&new_k_args[offset])) = args[4];
        // offset += sizeof(args[4]);

        *(reinterpret_cast<int*>(&new_k_args[offset])) = batch_size;
        offset += sizeof(batch_size);
        *(reinterpret_cast<int*>(&new_k_args[offset])) = sequence_length;
        offset += sizeof(sequence_length);
        *(reinterpret_cast<int*>(&new_k_args[offset])) = sequence_length;
        offset += sizeof(sequence_length);
        *(reinterpret_cast<int*>(&new_k_args[offset])) = head_num;
        offset += sizeof(head_num);
        *(reinterpret_cast<int*>(&new_k_args[offset])) = head_dim;
        offset += sizeof(head_dim);
        *(reinterpret_cast<float*>(&new_k_args[offset])) = scale;
        offset += sizeof(scale);

        // constexpr const char* module_file_name =
        // "D:\\owen\\ModelInferencingScripts\\hip\\amdmlss_kernels\\mha\\multi_head_attention_unpacked_128_64x192x48_64x48x64_forward_no_strides_fp16-hip-amdgcn-amd-amdhsa-gfx1201.out";

        // hipModule_t module_new;
        // auto status = hipModuleLoad(&module_new, module_file_name);

        // if(status != hipSuccess)
        //     MIGRAPHX_THROW("Failed to load module: " + hip_error(status));

        // hipFunction_t kernel_new;
        // auto status2 = hipModuleGetFunction(&kernel_new, module_new,
        // "_ZN2ck16tensor_operation6device41kernel_multi_head_attention_wmma_unpackedINS1_49DeviceBatchedGemmSoftmaxGemmPermute_Wmma_CShuffleILi2ELi1ELi1ELi1ELi1EDF16_DF16_DF16_DF16_NS_5TupleIJEEEfS5_ffNS0_12element_wise11PassThroughES7_NS6_5ScaleES7_S7_LNS1_18GemmSpecializationE15ELNS1_20TensorSpecializationE0ELSA_0ELSA_0ELSA_0ELi1ELi128ELi64ELi192ELi48ELi8ELi8ELi48ELi64ELi8ELi16ELi16ELi16ELi1ELi12ELi3ENS_8SequenceIJLi2ELi64ELi1EEEENSB_IJLi1ELi0ELi2EEEESD_Li2ELi8ELi8ELb1ESC_SD_SD_Li2ELi8ELi8ELb1ENSB_IJLi2ELi8ELi8EEEENSB_IJLi0ELi2ELi1EEEESF_Li1ELi2ELi1ELb0ELi1ELi1ENSB_IJLi1ELi64ELi1ELi2EEEELi8ELNS1_21MaskingSpecializationE0ELNS_13LoopSchedulerE0ELNS_15PipelineVersionE0EEENS_35GridwiseBatchedGemmSoftmaxGemm_WmmaIDF16_DF16_fDF16_ffDF16_S7_S7_S8_S7_S7_LNS_25InMemoryDataOperationEnumE0ENS_16TensorDescriptorINS4_IJNS_5EmbedINS4_IJiiEEESP_Lb0EEENS_23Merge_v2_magic_divisionINS4_IJiEEEEEST_NS_8RightPadIiiLb0EEESV_NS_7UnMergeINS4_IJiNS_17integral_constantIiLi1EEENSX_IiLi2EEENSX_IiLi8EEEEEELb0EEENSW_INS4_IJiNSX_IiLi4EEENSX_IiLi16EEEEEELb0EEEEEENS4_IJNSB_IJLi0EEEENSB_IJLi1EEEENSB_IJLi2EEEENSB_IJLi3EEEENSB_IJLi4EEEENSB_IJLi6EEEENSB_IJLi5EEEEEEENS4_IJNSB_IJLi1ELi2EEEES1B_S1C_S1E_S1D_NSB_IJLi7ELi8ELi9ELi10EEEENSB_IJLi11ELi12ELi13EEEEEEENSB_IJLi7ELi11ELi12ELi8ELi9ELi13ELi10EEEExEENSN_INS4_IJSQ_ST_ST_SV_SV_NSW_INS4_IJiS10_EEELb0EEENS_11PassThroughIiEEEEES1F_NS4_IJS1G_S1B_S1C_S1E_S1D_NSB_IJLi7ELi8EEEENSB_IJLi9EEEEEEENSB_IJLi7ELi9ELi8EEEExEES1V_NSN_INS4_IJSQ_ST_ST_SV_SV_EEENS4_IJS18_S19_S1A_S1B_S1C_EEENS4_IJS1G_S1B_S1C_S1E_S1D_EEENSB_IJLi5ELi6EEEExEELi64ELi192ELi48ELi8ELi8ELi48ELi64ELi8ELi16ELi16ELi16ELi1ELi12ELi3ELi128ESC_SD_SD_Li2ELi8ELi8ELb1ELb0ELb1ESC_SD_SD_Li2ELi8ELi8ELb1ELb1ELb1ESE_SF_SF_Li1ELi2ELi1ELb0ELb1ELb0ELi1ELi1ESG_Li8ELb1ELb0ELi1ELSI_0ELSJ_0EEEDF16_DF16_DF16_DF16_S7_S7_S8_S7_S7_Lb0EEEvPPKT1_PPKT2_PPKT3_PPT4_iiiiif");

        // if(status2 != hipSuccess)
        //     MIGRAPHX_THROW("Failed to load module: " + hip_error(status));

        launch_kernel(impl->fun, stream, 256, local, new_k_args, offset, start, stop);
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
