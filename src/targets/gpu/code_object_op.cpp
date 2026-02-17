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
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/pmr/vector.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_REGISTER_OP(code_object_op);

shape code_object_op::compute_shape(std::vector<shape> inputs) const
{
    std::transform(inputs.begin(), inputs.end(), inputs.begin(), [](const shape& s) {
        return s.normalize_standard();
    });
    auto einputs = expected_inputs;
    std::transform(einputs.begin(), einputs.end(), einputs.begin(), [](const shape& s) {
        return s.normalize_standard();
    });
    if(not migraphx::equal(flatten(einputs), flatten(inputs), &shape::is_compatible))
        MIGRAPHX_THROW("Input shapes have changed: [" + to_string_range(einputs) + "] -> [" +
                       to_string_range(inputs) + "]");
    return output;
}

static bool needs_flatten(const std::vector<argument>& args)
{
    return std::any_of(args.begin(), args.end(), [&](const argument& arg) {
        return arg.get_shape().type() == shape::tuple_type;
    });
}

template <class F>
static void visit_flatten_args(const std::vector<argument>& args, F f)
{
    if(needs_flatten(args))
        f(flatten(args));
    else
        f(args);
}

argument
code_object_op::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    if(args.size() == 5)
    {
         //for(int i = 0; i < args.size() - 1; i++)
         //{
         //   auto first_arg = args[i];
         //   auto shape     = first_arg.get_shape().lens();
         //   auto strides     = first_arg.get_shape().strides();
         //   auto elements  = first_arg.get_shape().elements();
         //   std::size_t bytes = first_arg.get_shape().bytes();
         //   char* dataa    = first_arg.data();
         //   //using migraphx::half;
         //   int elem = elements;
         //   std::vector<half> h_out;
         //   h_out.reserve(elem);
         //   auto status = hipMemcpy(h_out.data(), dataa, bytes, hipMemcpyDeviceToHost);
         //   if(status != hipSuccess)
         //       MIGRAPHX_THROW("Failed to launch kernel: " + hip_error(status));
         //   std::vector<float> float_conv(elem);
         //   for(int j = 0; j < elements; j++)
         //   {
         //       float_conv[j] = h_out[j].to_float();
         //   }
         //}

        auto query = args[0];
        auto query_strides = query.get_shape().strides();

        auto query_elements = query.get_shape().elements();
        std::size_t query_bytes = query.get_shape().bytes();
        std::vector<half> query_out(query_elements);
        auto status_query = hipMemcpy(query_out.data(), query.data(), query_bytes, hipMemcpyDeviceToHost);
        if(status_query != hipSuccess)
            MIGRAPHX_THROW("Failed to launch kernel: " + hip_error(status_query));
        std::vector<float> query_float_conv(query_elements);
        for(int j = 0; j < query_elements; j++)
        {
            query_float_conv[j] = query_out[j].to_float();
        }

        auto key              = args[1];
        auto key_strides        = key.get_shape().strides();

        auto key_elements       = key.get_shape().elements();
        std::size_t key_bytes = key.get_shape().bytes();
        std::vector<half> key_out(key_elements);
        auto status_key = hipMemcpy(key_out.data(), key.data(), key_bytes, hipMemcpyDeviceToHost);
        if(status_key != hipSuccess)
            MIGRAPHX_THROW("Failed to launch kernel: " + hip_error(status_key));
        std::vector<float> key_float_conv(key_elements);
        for(int j = 0; j < key_elements; j++)
        {
            key_float_conv[j] = key_out[j].to_float();
        }

        auto value              = args[2];
        auto value_strides      = value.get_shape().strides();

        auto value_elements     = value.get_shape().elements();
        std::size_t value_bytes = value.get_shape().bytes();
        std::vector<half> value_out(value_elements);
        auto status_value =
            hipMemcpy(value_out.data(), value.data(), value_bytes, hipMemcpyDeviceToHost);
        if(status_value != hipSuccess)
            MIGRAPHX_THROW("Failed to launch kernel: " + hip_error(status_value));
        std::vector<float> value_float_conv(value_elements);
        for(int j = 0; j < value_elements; j++)
        {
            value_float_conv[j] = value_out[j].to_float();
        }

        auto scale              = args[3];        

        auto outval              = args[4];
        auto outval_strides     = outval.get_shape().strides();
        std::size_t outval_bytes = outval.get_shape().bytes();

        //pmr::vector<void*> kargs;

         std::vector<kernel_argument> kargs;

        hipDeviceptr_t d_q_in    = query.data();
        hipDeviceptr_t* dd_q_ptr = nullptr;
        hipMalloc(&dd_q_ptr, sizeof(void*));
        hipMemcpy(dd_q_ptr, &d_q_in, sizeof(void*), hipMemcpyHostToDevice);
        kargs.emplace_back(dd_q_ptr);

        hipDeviceptr_t d_k_in    = key.data();
        hipDeviceptr_t* dd_k_ptr = nullptr;
        hipMalloc(&dd_k_ptr, sizeof(void*));
        hipMemcpy(dd_k_ptr, &d_k_in, sizeof(void*), hipMemcpyHostToDevice);
        kargs.emplace_back(dd_k_ptr);

        hipDeviceptr_t d_v_in    = value.data();
        hipDeviceptr_t* dd_v_ptr = nullptr;
        hipMalloc(&dd_v_ptr, sizeof(void*));
        hipMemcpy(dd_v_ptr, &d_v_in, sizeof(void*), hipMemcpyHostToDevice);
        kargs.emplace_back(dd_v_ptr);

        hipDeviceptr_t d_out_in    = outval.data();
        hipDeviceptr_t* dd_out_ptr = nullptr;
        hipMalloc(&dd_out_ptr, sizeof(void*));
        hipMemcpy(dd_out_ptr, &d_out_in, sizeof(void*), hipMemcpyHostToDevice);
        kargs.emplace_back(dd_out_ptr);


        auto query_shape = query.get_shape().lens();

        int batch_size         = query_shape[0];
        int q_sequence_length  = query_shape[2];
        int kv_sequence_length = query_shape[2];
        int head_num           = query_shape[1];
        int head_dim           = query_shape[3];

        kargs.push_back(batch_size);
        kargs.push_back(q_sequence_length);
        kargs.push_back(kv_sequence_length);
        kargs.push_back(head_num);
        kargs.push_back(head_dim);


        hipDeviceptr_t scale_ptr = scale.data();

        auto scale_elements = scale.get_shape().elements();
        std::size_t scale_bytes = scale.get_shape().bytes();
        std::vector<float> scale_out(scale_elements);

        auto status_scale = hipMemcpy(scale_out.data(), scale_ptr, scale_bytes, hipMemcpyDeviceToHost);
        if(status_scale != hipSuccess)
            MIGRAPHX_THROW("Failed to launch kernel: " + hip_error(status_scale));

        //float scale_in              = 0.5f;
        kargs.push_back(scale_out[0]);
        //kargs.push_back(scale_ptr);
        
        // q: {1, 2, 3, 4}, {4, 12, 24, 1}
        uint32_t q_stride_d0 = query_strides[0];
        uint32_t q_stride_d1 = query_strides[1];
        uint32_t q_stride_d2 = query_strides[2];
        uint32_t q_stride_d3 = query_strides[3];

        // k: {1, 2, 4, 3}, {4, 12, 1, 24}
        uint32_t k_stride_d0 = key_strides[0];
        uint32_t k_stride_d1 = key_strides[1];
        uint32_t k_stride_d2 = key_strides[2];
        uint32_t k_stride_d3 = key_strides[3];

        // v: {1, 2, 3, 4}, {4, 12, 24, 1}
        uint32_t v_stride_d0 = value_strides[0];
        uint32_t v_stride_d1 = value_strides[1];
        uint32_t v_stride_d2 = value_strides[2];
        uint32_t v_stride_d3 = value_strides[3];

        // {1, 2, 3, 4}, {24, 12, 4, 1}
        uint32_t output_stride_d0 = outval_strides[0];
        uint32_t output_stride_d1 = outval_strides[1];
        uint32_t output_stride_d2 = outval_strides[2];
        uint32_t output_stride_d3 = outval_strides[3];

        kargs.push_back(q_stride_d0);
        kargs.push_back(q_stride_d1);
        kargs.push_back(q_stride_d2);
        kargs.push_back(q_stride_d3);

        kargs.push_back(k_stride_d0);
        kargs.push_back(k_stride_d1);
        kargs.push_back(k_stride_d2);
        kargs.push_back(k_stride_d3);

        kargs.push_back(v_stride_d0);
        kargs.push_back(v_stride_d1);
        kargs.push_back(v_stride_d2);
        kargs.push_back(v_stride_d3);

        kargs.push_back(output_stride_d0);
        kargs.push_back(output_stride_d1);
        kargs.push_back(output_stride_d2);
        kargs.push_back(output_stride_d3);

        auto [start, stop] = ctx.get_perf_events();
        k.launch(ctx.get_stream().get(), global, local, kargs, start, stop);

        hipStreamSynchronize(ctx.get_stream().get());

        auto out_elements = outval.get_shape().elements();
        std::vector<half> h_out(out_elements);

        auto status = hipMemcpy(h_out.data(), d_out_in, outval_bytes, hipMemcpyDeviceToHost);
        if(status != hipSuccess)
            MIGRAPHX_THROW("Failed to launch kernel: " + hip_error(status));
        std::vector<float> out_float_conv(out_elements);

        for(int j = 0; j < h_out.size(); j++)
        {
            out_float_conv[j] = h_out[j].to_float();
        }


        return args[4];
        //return args[get_output_arg(args.size())];


        //for(int i = 0; i < args.size() - 1; i++)
        //{
        //    auto first_arg = args[i];
        //    auto shape     = first_arg.get_shape().lens();
        //    auto strides     = first_arg.get_shape().strides();
        //    auto elements  = first_arg.get_shape().elements() * 3;
        //    std::size_t bytes = first_arg.get_shape().bytes();
        //    char* dataa    = first_arg.data();
        //    using migraphx::half;
        //    //int size = 1 * 2 * 3 * 1 * sizeof(half);

        //    int elem = elements;
        //    int size = elem * sizeof(half);

        //    std::vector<half> h_out(elem);
        //    auto status = hipMemcpy(h_out.data(), dataa, bytes, hipMemcpyDeviceToHost);
        //    if(status != hipSuccess)
        //        MIGRAPHX_THROW("Failed to launch kernel: " + hip_error(status));
        //    std::vector<float> float_conv(elem);

        //    for(int j = 0; j < h_out.size(); j++)
        //    {
        //        float_conv[j] = h_out[j].to_float();
        //    }
        //}

        //int out_elem = 1 * 2 * 3 * 1;
        //int out_size = out_elem * sizeof(half);

        //std::vector<half> h_output(out_elem);
        //for(int i = 0; i < h_output.size(); i++)
        //{
        //    h_output[i] = half(5.0f);
        //}
        //char* out_data = args[3].data();
        //auto status    = hipMemcpy(out_data, h_output.data(), out_size, hipMemcpyHostToDevice);

    }
    else
    {
        // auto status = hipDeviceSynchronize();
#if MIGRAPHX_HAS_PMR
        std::array<char, 256> storage;
        std::pmr::monotonic_buffer_resource resource{storage.data(), storage.size()};
        pmr::vector<void*> kargs(&resource);
#else
        pmr::vector<void*> kargs;
#endif
        visit_flatten_args(args, [&](const auto& fargs) {
            kargs.reserve(fargs.size());
            std::transform(fargs.begin(),
                           fargs.end(),
                           std::back_inserter(kargs),
                           [](const argument& a) { return a.data(); });
        });
        auto [start, stop] = ctx.get_perf_events();
        k.launch(ctx.get_stream().get(), global, local, kargs, start, stop);
        return args[get_output_arg(args.size())];
    }
}
void code_object_op::finalize(context&, const shape&, const std::vector<shape>&)
{
    assert(not code_object.empty());
    k = kernel(code_object, symbol_name);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
