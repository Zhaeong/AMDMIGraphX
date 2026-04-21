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

        
        std::size_t query_bytes = query.get_shape().bytes();
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
            std::ofstream f("packed_q_static.hpp");
            f << "// Packed Q input" << "\n";
            f << "// Total elements: " << query_float_size << "\n\n";
            f << "static const float Q_data[" << query_float_size << "] = {\n";
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

        auto key              = args[1];
        auto key_strides        = key.get_shape().strides();        
        std::size_t key_bytes = key.get_shape().bytes();
        auto key_elements     = key_bytes / (sizeof(half));
        std::vector<half> key_out(key_elements);
        auto status_key = hipMemcpy(key_out.data(), key.data(), key_bytes, hipMemcpyDeviceToHost);
        if(status_key != hipSuccess)
            MIGRAPHX_THROW("Failed to launch kernel: " + hip_error(status_key));
        std::vector<float> key_float_conv(key_elements);
        for(int j = 0; j < key_elements; j++)
        {
            key_float_conv[j] = key_out[j].to_float();
        }

        // ── Dump packed_k to hpp ────────────────────────────────────────
        {
            const size_t key_float_size = key_float_conv.size();
            std::ofstream f("packed_k_static.hpp");
            f << "// Packed Q input" << "\n";
            f << "// Total elements: " << key_float_size << "\n\n";
            f << "static const float K_data[" << key_float_size << "] = {\n";
            for (size_t i = 0; i < key_float_size; ++i) {
                if (i % 8 == 0) f << "    ";
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.8ff", key_float_conv[i]);
                f << buf;
                if (i + 1 < key_float_size) f << ",";
                if ((i + 1) % 8 == 0 || i + 1 == key_float_size)
                    f << "\n";
                else
                    f << "    ";
            }
            f << "};\n";
            std::cout << "packed_k_static written to packed_k_static.hpp (" << key_float_size << " elements)\n";
        }

        auto value              = args[2];
        auto value_strides      = value.get_shape().strides();        
        std::size_t value_bytes = value.get_shape().bytes();
        auto value_elements     = value_bytes / (sizeof(half));
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

        // ── Dump packed_v to hpp ────────────────────────────────────────
        {
            const size_t value_float_size = value_float_conv.size();
            std::ofstream f("packed_v_static.hpp");
            f << "// Packed Q input" << "\n";
            f << "// Total elements: " << value_float_size << "\n\n";
            f << "static const float V_data[" << value_float_size << "] = {\n";
            for (size_t i = 0; i < value_float_size; ++i) {
                if (i % 8 == 0) f << "    ";
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.8ff", value_float_conv[i]);
                f << buf;
                if (i + 1 < value_float_size) f << ",";
                if ((i + 1) % 8 == 0 || i + 1 == value_float_size)
                    f << "\n";
                else
                    f << "    ";
            }
            f << "};\n";
            std::cout << "packed_v_static written to packed_v_static.hpp (" << value_float_size << " elements)\n";
        }


        auto query_shape = query.get_shape().lens();

        int batch_size         = query_shape[0];
        int q_sequence_length  = query_shape[2];
        int kv_sequence_length = query_shape[2];
        int head_num           = query_shape[1];
        int head_dim           = query_shape[3];

        int B = batch_size;
        int H = head_num;
        int S = q_sequence_length;  // query sequence length
        int N = kv_sequence_length; // key/value sequence length
        int D = head_dim;           // head dimension

        const int qn = batch_size*head_num*q_sequence_length*head_dim;   // 20480
        const int kn = batch_size*head_num*kv_sequence_length*head_dim;    // 20480


        // const int qn = B*H*S*D;   // 20480
        // const int kn = B*H*N*D;   // 20480


        // Packed strides provided for each array (in elements):
        // constexpr int QD0 = 40,  QD1 = 120, QD2 = 960, QD3 = 1;
        // constexpr int KD0 = 40,  KD1 = 120, KD2 = 1,   KD3 = 960;
        // constexpr int VD0 = 40,  VD1 = 120, VD2 = 960, VD3 = 1;

        int QD0 = query_strides[0],  QD1 = query_strides[1], QD2 = query_strides[2], QD3 = query_strides[3];
        // int KD0 = key_strides[0],  KD1 = key_strides[1], KD2 = key_strides[2],   KD3 = key_strides[3];
        int KD0 = key_strides[0],  KD1 = key_strides[1], KD2 = key_strides[3],   KD3 = key_strides[2]; // neeed to reverse index 3 and 2

        int VD0 = value_strides[0],  VD1 = value_strides[1], VD2 = value_strides[2], VD3 = value_strides[3];

        printf("Unpacking packed Q/K/V to contiguous layout...\n");
        std::vector<float> Qc(qn), Kc(kn), Vc(kn);
        for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h)
        for (int s = 0; s < S; ++s)
        for (int d = 0; d < D; ++d)
        {
            Qc[b*(H*S*D) + h*(S*D) + s*D + d] = query_float_conv[b*QD0 + h*QD1 + s*QD2 + d*QD3];
            Kc[b*(H*N*D) + h*(N*D) + s*D + d] = key_float_conv[b*KD0 + h*KD1 + s*KD2 + d*KD3];
            Vc[b*(H*N*D) + h*(N*D) + s*D + d] = value_float_conv[b*VD0 + h*VD1 + s*VD2 + d*VD3];
        }


        {
            const size_t q_unpacked_float_size = Qc.size();
            std::ofstream f("unpacked_q_static.hpp");
            f << "// Total elements: " << q_unpacked_float_size << "\n\n";
            f << "static const float Q_data[" << q_unpacked_float_size << "] = {\n";
            for (size_t i = 0; i < q_unpacked_float_size; ++i) {
                if (i % 8 == 0) f << "    ";
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.8ff", Qc[i]);
                f << buf;
                if (i + 1 < q_unpacked_float_size) f << ",";
                if ((i + 1) % 8 == 0 || i + 1 == q_unpacked_float_size)
                    f << "\n";
                else
                    f << "    ";
            }
            f << "};\n";
        }
        {

            const size_t k_unpacked_float_size = Kc.size();
            std::ofstream f("unpacked_k_static.hpp");
            f << "// Total elements: " << k_unpacked_float_size << "\n\n";
            f << "static const float K_data[" << k_unpacked_float_size << "] = {\n";
            for (size_t i = 0; i < k_unpacked_float_size; ++i) {
                if (i % 8 == 0) f << "    ";
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.8ff", Kc[i]);
                f << buf;
                if (i + 1 < k_unpacked_float_size) f << ",";
                if ((i + 1) % 8 == 0 || i + 1 == k_unpacked_float_size)
                    f << "\n";
                else
                    f << "    ";
            }
            f << "};\n";
        }
        {

            const size_t v_unpacked_float_size = Vc.size();
            std::ofstream f("unpacked_v_static.hpp");
            f << "// Total elements: " << v_unpacked_float_size << "\n\n";
            f << "static const float V_data[" << v_unpacked_float_size << "] = {\n";
            for (size_t i = 0; i < v_unpacked_float_size; ++i) {
                if (i % 8 == 0) f << "    ";
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.8ff", Vc[i]);
                f << buf;
                if (i + 1 < v_unpacked_float_size) f << ",";
                if ((i + 1) % 8 == 0 || i + 1 == v_unpacked_float_size)
                    f << "\n";
                else
                    f << "    ";
            }
            f << "};\n";


        }


        // ── GPU: upload contiguous fp16 tensors ───────────────────────────────────
        std::vector<half> hq(qn), hk(kn), hv(kn);
        
        for (int i = 0; i < qn; ++i) hq[i] = migraphx::half(Qc[i]);
        for (int i = 0; i < kn; ++i) { hk[i] = migraphx::half(Kc[i]); hv[i] = migraphx::half(Vc[i]); }


        hipDeviceptr_t dq, dk, dv, dout;
        hipMalloc(&dq,   qn * sizeof(half));
        hipMalloc(&dk,   kn * sizeof(half));
        hipMalloc(&dv,   kn * sizeof(half));
        // HIP_CHECK(hipMalloc(&dout, qn * sizeof(__half)));
        hipMemcpy(dq,   hq.data(), qn * sizeof(half), hipMemcpyHostToDevice);
        hipMemcpy(dk,   hk.data(), kn * sizeof(half), hipMemcpyHostToDevice);
        hipMemcpy(dv,   hv.data(), kn * sizeof(half), hipMemcpyHostToDevice);
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

        hipDeviceptr_t d_k_in    = key.data();
        hipDeviceptr_t* dd_k_ptr = nullptr;
        hipMalloc(&dd_k_ptr, sizeof(void*));
        // hipMemcpy(dd_k_ptr, &d_k_in, sizeof(void*), hipMemcpyHostToDevice);
        hipMemcpy(dd_k_ptr, &dk, sizeof(void*), hipMemcpyHostToDevice);
        kargs.emplace_back(dd_k_ptr);

        hipDeviceptr_t d_v_in    = value.data();
        hipDeviceptr_t* dd_v_ptr = nullptr;
        hipMalloc(&dd_v_ptr, sizeof(void*));
        // hipMemcpy(dd_v_ptr, &d_v_in, sizeof(void*), hipMemcpyHostToDevice);
        hipMemcpy(dd_v_ptr, &dv, sizeof(void*), hipMemcpyHostToDevice);
        kargs.emplace_back(dd_v_ptr);

        hipDeviceptr_t d_out_in    = outval.data();
        hipDeviceptr_t* dd_out_ptr = nullptr;
        hipMalloc(&dd_out_ptr, sizeof(void*));
        hipMemcpy(dd_out_ptr, &d_out_in, sizeof(void*), hipMemcpyHostToDevice);
        kargs.emplace_back(dd_out_ptr);


        

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




        uint32_t qd0 = H * S * D, qd1 = S * D, qd2 = D, qd3 = 1;
        uint32_t kd0 = H * N * D, kd1 = N * D, kd2 = D, kd3 = 1;
        uint32_t vd0 = H * N * D, vd1 = N * D, vd2 = 1, vd3 = D; // V d2/d3 swapped per kernel convention

        uint32_t od0 = H * S * D, od1 = S * D, od2 = D, od3 = 1;

        // q: {1, 2, 3, 4}, {4, 12, 24, 1}
        uint32_t q_stride_d0 = qd0;
        uint32_t q_stride_d1 = qd1;
        uint32_t q_stride_d2 = qd2;
        uint32_t q_stride_d3 = qd3;

        // k: {1, 2, 4, 3}, {4, 12, 1, 24}
        uint32_t k_stride_d0 = kd0;
        uint32_t k_stride_d1 = kd1;
        uint32_t k_stride_d2 = kd2;
        uint32_t k_stride_d3 = kd3;

        // v: {1, 2, 3, 4}, {4, 12, 24, 1}
        uint32_t v_stride_d0 = vd0;
        uint32_t v_stride_d1 = vd1;
        uint32_t v_stride_d2 = vd2;
        uint32_t v_stride_d3 = vd3;


        //// q: {1, 2, 3, 4}, {4, 12, 24, 1}
        //uint32_t q_stride_d0 = query_strides[0];
        //uint32_t q_stride_d1 = query_strides[1];
        //uint32_t q_stride_d2 = query_strides[2];
        //uint32_t q_stride_d3 = query_strides[3];

        //// k: {1, 2, 4, 3}, {4, 12, 1, 24}
        //uint32_t k_stride_d0 = key_strides[0];
        //uint32_t k_stride_d1 = key_strides[1];
        //uint32_t k_stride_d2 = key_strides[2];
        //uint32_t k_stride_d3 = key_strides[3];

        //// v: {1, 2, 3, 4}, {4, 12, 24, 1}
        //uint32_t v_stride_d0 = value_strides[0];
        //uint32_t v_stride_d1 = value_strides[1];
        //uint32_t v_stride_d2 = value_strides[2];
        //uint32_t v_stride_d3 = value_strides[3];
        
        // uint32_t v_stride_d2 = value_strides[3];
        // uint32_t v_stride_d3 = value_strides[2];

        // uint32_t vd0=40,  vd1=120, vd2=1,   vd3=960; 

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


        const int grid  = B * H * S * 2;
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
