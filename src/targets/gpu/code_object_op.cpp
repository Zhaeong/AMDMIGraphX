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
    if(args.size() == 4)
    {
        for(int i = 0; i < args.size() - 1; i++)
        {
            auto first_arg = args[i];
            auto shape     = first_arg.get_shape().lens();
            auto strides     = first_arg.get_shape().strides();
            char* dataa    = first_arg.data();
            using migraphx::half;
            //int size = 1 * 2 * 3 * 1 * sizeof(half);

            int elem = 1 * 3 * 2 * 3 * 1;
            int size = elem * sizeof(half);

            std::vector<half> h_out(elem);
            auto status = hipMemcpy(h_out.data(), dataa, size, hipMemcpyDeviceToHost);
            if(status != hipSuccess)
                MIGRAPHX_THROW("Failed to launch kernel: " + hip_error(status));
            std::vector<float> float_conv(elem);

            for(int j = 0; j < h_out.size(); j++)
            {
                float_conv[j] = h_out[j].to_float();
            }
        }

        int out_elem = 1 * 2 * 3 * 1;
        int out_size = out_elem * sizeof(half);

        std::vector<half> h_output(out_elem);
        for(int i = 0; i < h_output.size(); i++)
        {
            h_output[i] = half(5.0f);
        }
        char* out_data = args[3].data();
        auto status    = hipMemcpy(out_data, h_output.data(), out_size, hipMemcpyHostToDevice);

    }
    //auto status = hipDeviceSynchronize();
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
void code_object_op::finalize(context&, const shape&, const std::vector<shape>&)
{
    assert(not code_object.empty());
    k = kernel(code_object, symbol_name);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
