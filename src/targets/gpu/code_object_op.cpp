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
#include <fstream>

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
        


        auto query = args[0];

        auto query_shape   = query.get_shape().lens();
        int head_dim     = query_shape[3];


        auto query_strides = query.get_shape().strides();
                
        // add 2 * head_dim to get the full
        std::size_t query_bytes = query.get_shape().bytes() + (head_dim * 2 * sizeof(half));
        auto query_elements     = query_bytes / (sizeof(half));
        std::vector<half> query_out(query_elements);
        auto status_query = hipMemcpy(query_out.data(), query.data(), query_bytes, hipMemcpyDeviceToHost);
        if(status_query != hipSuccess)
            MIGRAPHX_THROW("Failed to launch kernel: " + hip_error(status_query));
        std::vector<float> query_float_conv(query_elements);
        for(int j = 0; j < query_elements; j++)
        {
            query_float_conv[j] = query_out[j].to_float();
        }

        // ── Dump packed_q to hpp ────────────────────────────────────────
        {
            const size_t query_float_size = query_float_conv.size();
            std::ofstream f("packed_qkv_static_input.hpp");
            f << "// Packed Q input" << "\n";
            f << "// Total elements: " << query_float_size << "\n\n";
            f << "static const float QKV_data[" << query_float_size << "] = {\n";
            for (size_t i = 0; i < query_float_size; ++i) {
                if (i % 8 == 0) f << "    ";
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.8ff", query_float_conv[i]);
                f << buf;
                if (i + 1 < query_float_size) f << ",";
                if ((i + 1) % 8 == 0 || i + 1 == query_float_size)
                    f << "\n";
                else
                    f << "    ";
            }
            f << "};\n";
            std::cout << "packed_q_static written to packed_q_static.hpp (" << query_float_size << " elements)\n";
        }


        

        int batch_size         = query_shape[0];
        int q_sequence_length  = query_shape[2];
        int kv_sequence_length = query_shape[2];
        int head_num           = query_shape[1];
        

        int B = batch_size;
        int H = head_num;
        int S = q_sequence_length;  // query sequence length
        int N = kv_sequence_length; // key/value sequence length
        int D = head_dim;           // head dimension

        const int qn = batch_size*head_num*q_sequence_length*head_dim*3;   



        // QKV_data layout: [B, S, H, 3, D]  (seq-major, ONNX packed format)
        // Kernel expects:  [B, H, S, 3*D]  (head-major, interleaved QKV per token)
        // Transpose S and H axes while preserving the 3*D inner block.
        std::vector<half> h_qkv(qn);
        for (std::size_t b = 0; b < static_cast<std::size_t>(batch_size); ++b)
        for (std::size_t s = 0; s < static_cast<std::size_t>(q_sequence_length); ++s)
        for (std::size_t h = 0; h < static_cast<std::size_t>(head_num); ++h)
        for (int d3 = 0; d3 < 3 * head_dim; ++d3)
        {
            // src: [B, S, H, 3*D]
            std::size_t src = b * (q_sequence_length * head_num * 3 * head_dim)
                            + s * (head_num * 3 * head_dim)
                            + h * (3 * head_dim)
                            + d3;
            // dst: [B, H, S, 3*D]
            std::size_t dst = b * (head_num * q_sequence_length * 3 * head_dim)
                            + h * (q_sequence_length * 3 * head_dim)
                            + s * (3 * head_dim)
                            + d3;
            h_qkv[dst] = query_out[src];
        }




        // // ── GPU: upload contiguous fp16 tensors ───────────────────────────────────
        // std::vector<half> hq(qn); //, hk(kn), hv(kn);
        
        // for (int i = 0; i < qn; ++i) hq[i] = migraphx::half(query_float_conv[i]);
        // for (int i = 0; i < kn; ++i) { hk[i] = migraphx::half(Kc[i]); hv[i] = migraphx::half(Vc[i]); }


        hipDeviceptr_t dq, dk, dv, dout;
        hipMalloc(&dq,   qn * sizeof(half));
        // hipMalloc(&dk,   kn * sizeof(half));
        // hipMalloc(&dv,   kn * sizeof(half));
        // // HIP_CHECK(hipMalloc(&dout, qn * sizeof(__half)));
        hipMemcpy(dq,   h_qkv.data(), qn * sizeof(half), hipMemcpyHostToDevice);
        // hipMemcpy(dk,   hk.data(), kn * sizeof(half), hipMemcpyHostToDevice);
        // hipMemcpy(dv,   hv.data(), kn * sizeof(half), hipMemcpyHostToDevice);
        // HIP_CHECK(hipMemcpy(dout, ho.data(), qn * sizeof(__half), hipMemcpyHostToDevice));


        auto scale              = args[3];        

        auto outval              = args[4];
        auto outval_strides     = outval.get_shape().strides();
        std::size_t outval_bytes = outval.get_shape().bytes();


         std::vector<kernel_argument> kargs;

        hipDeviceptr_t d_q_in    = query.data();
        hipDeviceptr_t* dd_q_ptr = nullptr;
        hipMalloc(&dd_q_ptr, sizeof(void*));
        // hipMemcpy(dd_q_ptr, &d_q_in, sizeof(void*), hipMemcpyHostToDevice);
        hipMemcpy(dd_q_ptr, &dq, sizeof(void*), hipMemcpyHostToDevice);
        kargs.emplace_back(dd_q_ptr);

        hipDeviceptr_t d_out_in    = outval.data();
        hipDeviceptr_t* dd_out_ptr = nullptr;
        hipMalloc(&dd_out_ptr, sizeof(void*));
        hipMemcpy(dd_out_ptr, &d_out_in, sizeof(void*), hipMemcpyHostToDevice);
        kargs.emplace_back(dd_out_ptr);


        

        kargs.push_back(batch_size);
        kargs.push_back(q_sequence_length);
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


        uint32_t qd0 = H * S * D, qd1 = S * D, qd2 = D, qd3 = 1;
        uint32_t kd0 = H * N * D, kd1 = N * D, kd2 = D, kd3 = 1;
        uint32_t vd0 = H * N * D, vd1 = N * D, vd2 = 1, vd3 = D; // V d2/d3 swapped per kernel convention

        uint32_t od0 = H * S * D, od1 = S * D, od2 = D, od3 = 1;


        uint32_t stride_d0 = static_cast<uint32_t>(head_num * q_sequence_length * 3 * head_dim);
        uint32_t stride_d1 = static_cast<uint32_t>(q_sequence_length * 3 * head_dim);
        uint32_t stride_d2 = static_cast<uint32_t>(3 * head_dim);
        uint32_t stride_d3 = 1u;

        // q: {1, 2, 3, 4}, {4, 12, 24, 1}
        uint32_t q_stride_d0 = stride_d0;
        uint32_t q_stride_d1 = stride_d1;
        uint32_t q_stride_d2 = stride_d2;
        uint32_t q_stride_d3 = stride_d3;

        // k: {1, 2, 4, 3}, {4, 12, 1, 24}
        uint32_t k_stride_d0 = stride_d0;
        uint32_t k_stride_d1 = stride_d1;
        uint32_t k_stride_d2 = stride_d2;
        uint32_t k_stride_d3 = stride_d3;

        // v: {1, 2, 3, 4}, {4, 12, 24, 1}
        uint32_t v_stride_d0 = stride_d0;
        uint32_t v_stride_d1 = stride_d1;
        uint32_t v_stride_d2 = stride_d3; // swapped for v
        uint32_t v_stride_d3 = stride_d2;


        // {1, 2, 3, 4}, {24, 12, 4, 1}
        // uint32_t output_stride_d0 = outval_strides[0];
        // uint32_t output_stride_d1 = outval_strides[1];
        // uint32_t output_stride_d2 = outval_strides[2];
        // uint32_t output_stride_d3 = outval_strides[3];

        uint32_t out_stride_d0 = static_cast<uint32_t>(head_num * q_sequence_length * head_dim);
        uint32_t out_stride_d1 = static_cast<uint32_t>(q_sequence_length * head_dim);
        uint32_t out_stride_d2 = static_cast<uint32_t>(head_dim);
        uint32_t out_stride_d3 = 1u;

        uint32_t output_stride_d0 = out_stride_d0;
        uint32_t output_stride_d1 = out_stride_d1;
        uint32_t output_stride_d2 = out_stride_d2;
        uint32_t output_stride_d3 = out_stride_d3;

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


        const int grid  = B * H * S * 2;
        const unsigned int grid_size = static_cast<unsigned>(batch_size) * head_num * q_sequence_length * 2u;

        const int block = 128;


        auto [start, stop] = ctx.get_perf_events();
        // k.launch(ctx.get_stream().get(), global, local, kargs, start, stop);
        k.launch(ctx.get_stream().get(), grid, block, kargs, start, stop);

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
