#include "layers/conv_layer.hpp"

#include <cassert>
#include <iostream>
#include <omp.h>

ConvLayer::ConvLayer(size_t input_h, size_t input_w, size_t input_c, size_t kernel_size, size_t filter_count,
                     std::mt19937& gen)
    : in_h_(input_h)
    , in_w_(input_w)
    , in_c_(input_c)
    , k_size_(kernel_size)
    , out_c_(filter_count)
{
    out_h_ = input_h - kernel_size + 1;
    out_w_ = input_w - kernel_size + 1;

    filters_.reshape(Shape({ out_c_, in_c_ * k_size_ * k_size_ }));
    biases_.reshape(Shape({ out_c_, 1 }));

    scalar_t std_dev = std::sqrt(2.0f / static_cast<scalar_t>(in_c_ * kernel_size * kernel_size));
    filters_.random_normal(std_dev, gen);
    biases_.fill(0.0f);
}

const Tensor& ConvLayer::forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool is_training) const
{
    if (!ctx)
    {
        ctx = std::make_unique<ConvContext>();
    }
    auto* conv_ctx = static_cast<ConvContext*>(ctx.get());

    size_t batch_size = input.shape()[0];

    if (conv_ctx->output.shape().rank() == 0 || conv_ctx->output.shape()[0] != batch_size)
    {
        conv_ctx->output.reshape(Shape({ batch_size, out_c_, out_h_, out_w_ }));
    }

    size_t required_col_size = (in_c_ * k_size_ * k_size_) * (batch_size * out_h_ * out_w_);
    if (conv_ctx->col_buffer.size() < required_col_size)
    {
        conv_ctx->col_buffer.reshape(Shape({ in_c_ * k_size_ * k_size_, batch_size * out_h_ * out_w_ }));
        conv_ctx->gemm_out.reshape(Shape({ out_c_, batch_size * out_h_ * out_w_ }));
    }
    else
    {
        // Use Shape to avoid reallocating if size is enough
        conv_ctx->col_buffer.reshape(Shape({ in_c_ * k_size_ * k_size_, batch_size * out_h_ * out_w_ }));
        conv_ctx->gemm_out.reshape(Shape({ out_c_, batch_size * out_h_ * out_w_ }));
    }

#pragma omp parallel for collapse(2) if (batch_size * in_c_ > 4)
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t c = 0; c < in_c_; ++c)
        {
            for (size_t ky = 0; ky < k_size_; ++ky)
            {
                for (size_t kx = 0; kx < k_size_; ++kx)
                {
                    size_t row_idx = c * k_size_ * k_size_ + ky * k_size_ + kx;
                    for (size_t y = 0; y < out_h_; ++y)
                    {
                        for (size_t x = 0; x < out_w_; ++x)
                        {
                            size_t col_idx = b * (out_h_ * out_w_) + y * out_w_ + x;
                            conv_ctx->col_buffer(row_idx, col_idx) = input(b, c, y + ky, x + kx);
                        }
                    }
                }
            }
        }
    }

    Tensor::matmul(filters_, conv_ctx->col_buffer, conv_ctx->gemm_out);

#pragma omp parallel for collapse(3) if (batch_size * out_c_ * out_h_ > 100)
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t f = 0; f < out_c_; ++f)
        {
            for (size_t i = 0; i < out_h_ * out_w_; ++i)
            {
                size_t y = i / out_w_;
                size_t x = i % out_w_;
                size_t col_idx = b * (out_h_ * out_w_) + i;
                conv_ctx->output(b, f, y, x) = conv_ctx->gemm_out(f, col_idx) + biases_(f, 0);
            }
        }
    }

    (void)is_training;
    return conv_ctx->output;
}

const Tensor& ConvLayer::backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx, bool is_training)
{
    assert(is_training && "Backward doit uniquement etre appele durant l'entrainement !");
    (void)is_training;
    auto* conv_ctx = static_cast<ConvContext*>(ctx.get());

    size_t batch_size = gradient.shape()[0];

    if (conv_ctx->grad_input.shape().rank() == 0 || conv_ctx->grad_input.shape()[0] != batch_size)
    {
        conv_ctx->grad_input.reshape(Shape({ batch_size, in_c_, in_h_, in_w_ }));
    }
    conv_ctx->grad_input.fill(0.0f);

    if (filters_grad_.size() == 0)
    {
        filters_grad_.reshape(Shape({ out_c_, in_c_ * k_size_ * k_size_ }));
        biases_grad_.reshape(biases_.shape());
    }

    size_t required_grad_size = out_c_ * (batch_size * out_h_ * out_w_);
    if (conv_ctx->grad_view.size() < required_grad_size)
    {
        conv_ctx->grad_view.reshape(Shape({ out_c_, batch_size * out_h_ * out_w_ }));
        conv_ctx->grad_col.reshape(Shape({ in_c_ * k_size_ * k_size_, batch_size * out_h_ * out_w_ }));
    }
    else
    {
        conv_ctx->grad_view.reshape(Shape({ out_c_, batch_size * out_h_ * out_w_ }));
        conv_ctx->grad_col.reshape(Shape({ in_c_ * k_size_ * k_size_, batch_size * out_h_ * out_w_ }));
    }

#pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t f = 0; f < out_c_; ++f)
        {
            for (size_t y = 0; y < out_h_; ++y)
            {
                for (size_t x = 0; x < out_w_; ++x)
                {
                    size_t col_idx = b * (out_h_ * out_w_) + y * out_w_ + x;
                    conv_ctx->grad_view(f, col_idx) = gradient(b, f, y, x);
                }
            }
        }
    }

    biases_grad_.fill(0.0f);
    for (size_t f = 0; f < out_c_; ++f)
    {
        scalar_t sum = 0.0f;
        for (size_t col = 0; col < batch_size * out_h_ * out_w_; ++col)
        {
            sum += conv_ctx->grad_view(f, col);
        }
        biases_grad_(f, 0) = sum;
    }

    Tensor::matmul(conv_ctx->grad_view, conv_ctx->col_buffer, filters_grad_, false, true);

    Tensor::matmul(filters_, conv_ctx->grad_view, conv_ctx->grad_col, true, false);

#pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t c = 0; c < in_c_; ++c)
        {
            for (size_t ky = 0; ky < k_size_; ++ky)
            {
                for (size_t kx = 0; kx < k_size_; ++kx)
                {
                    size_t row_idx = c * k_size_ * k_size_ + ky * k_size_ + kx;
                    for (size_t y = 0; y < out_h_; ++y)
                    {
                        for (size_t x = 0; x < out_w_; ++x)
                        {
                            size_t col_idx = b * (out_h_ * out_w_) + y * out_w_ + x;
                            conv_ctx->grad_input(b, c, y + ky, x + kx) += conv_ctx->grad_col(row_idx, col_idx);
                        }
                    }
                }
            }
        }
    }

    return conv_ctx->grad_input;
}

void ConvLayer::clear_gradients()
{
    if (filters_grad_.size() > 0)
        filters_grad_.fill(0.0f);
    if (biases_grad_.size() > 0)
        biases_grad_.fill(0.0f);
}

Shape3D ConvLayer::get_output_shape(const Shape3D& input_shape) const
{
    return { out_c_, input_shape.height - k_size_ + 1, input_shape.width - k_size_ + 1 };
}

void ConvLayer::save(std::ostream& os) const
{
    uint32_t marker = make_marker("CONV");
    os.write(reinterpret_cast<const char*>(&marker), sizeof(marker));

    uint64_t in_h = in_h_, in_w = in_w_, in_c = in_c_, k_size = k_size_, out_c = out_c_;
    os.write(reinterpret_cast<const char*>(&in_h), sizeof(in_h));
    os.write(reinterpret_cast<const char*>(&in_w), sizeof(in_w));
    os.write(reinterpret_cast<const char*>(&in_c), sizeof(in_c));
    os.write(reinterpret_cast<const char*>(&k_size), sizeof(k_size));
    os.write(reinterpret_cast<const char*>(&out_c), sizeof(out_c));

    filters_.save(os);
    biases_.save(os);
}

void ConvLayer::load(std::istream& is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char*>(&marker), sizeof(marker));
    if (marker != make_marker("CONV"))
        throw std::runtime_error("Arch mismatch in ConvLayer load");

    uint64_t in_h, in_w, in_c, k_size, out_c;
    is.read(reinterpret_cast<char*>(&in_h), sizeof(in_h));
    is.read(reinterpret_cast<char*>(&in_w), sizeof(in_w));
    is.read(reinterpret_cast<char*>(&in_c), sizeof(in_c));
    is.read(reinterpret_cast<char*>(&k_size), sizeof(k_size));
    is.read(reinterpret_cast<char*>(&out_c), sizeof(out_c));

    in_h_ = in_h;
    in_w_ = in_w;
    in_c_ = in_c;
    k_size_ = k_size;
    out_c_ = out_c;
    out_h_ = in_h_ - k_size_ + 1;
    out_w_ = in_w_ - k_size_ + 1;

    filters_.load(is);
    biases_.load(is);
}

nlohmann::json ConvLayer::get_config() const
{
    return { { "type", "Conv" },
             { "filters", static_cast<int>(out_c_) },
             { "kernel_size", static_cast<int>(k_size_) } };
}