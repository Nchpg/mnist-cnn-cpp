#include "layers/conv_layer.hpp"

#include <omp.h>

ConvLayer::ConvLayer(size_t input_h, size_t input_w, size_t input_c,
                     size_t kernel_size, size_t filter_count, std::mt19937 &gen)
    : in_h_(input_h)
    , in_w_(input_w)
    , in_c_(input_c)
    , k_size_(kernel_size)
    , out_c_(filter_count)
{
    out_h_ = input_h - kernel_size + 1;
    out_w_ = input_w - kernel_size + 1;

    filters_.reshape(Shape({ out_c_, in_c_, k_size_, k_size_ }));
    biases_.reshape(Shape({ out_c_, 1 }));

    scalar_t std_dev = std::sqrt(
        2.0f / static_cast<scalar_t>(in_c_ * kernel_size * kernel_size));
    filters_.random_normal(std_dev, gen);
    biases_.fill(0.0f);

    filters_grad_.reshape(filters_.shape());
    biases_grad_.reshape(biases_.shape());
}

const Tensor &ConvLayer::forward(const Tensor &input)
{
    size_t batch_size = input.shape()[0];
    input_ptr_ = &input;

    if (output_.shape().rank() == 0 || output_.shape()[0] != batch_size)
    {
        output_.reshape(Shape({ batch_size, out_c_, out_h_, out_w_ }));
        grad_input_.reshape(input.shape());
        col_buffer_.reshape(
            Shape({ in_c_ * k_size_ * k_size_, batch_size * out_h_ * out_w_ }));
        gemm_out_.reshape(Shape({ out_c_, batch_size * out_h_ * out_w_ }));
    }

    filters_.reshape(Shape({ out_c_, in_c_ * k_size_ * k_size_ }));

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
                            col_buffer_(row_idx, col_idx) = input(b, c, y + ky, x + kx);
                        }
                    }
                }
            }
        }
    }

    Tensor::matmul(filters_, col_buffer_, gemm_out_);

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
                output_(b, f, y, x) = gemm_out_(f, col_idx) + biases_(f, 0);
            }
        }
    }

    filters_.reshape(Shape({ out_c_, in_c_, k_size_, k_size_ }));

    return output_;
}

const Tensor &ConvLayer::backward(const Tensor &gradient)
{
    size_t batch_size = gradient.shape()[0];

    grad_input_.fill(0.0f);

    filters_.reshape(Shape({ out_c_, in_c_ * k_size_ * k_size_ }));
    filters_grad_.reshape(Shape({ out_c_, in_c_ * k_size_ * k_size_ }));

    if (grad_view_.shape().rank() == 0
        || grad_view_.shape()[1] != batch_size * out_h_ * out_w_)
    {
        grad_view_.reshape(Shape({ out_c_, batch_size * out_h_ * out_w_ }));
        grad_col_.reshape(
            Shape({ in_c_ * k_size_ * k_size_, batch_size * out_h_ * out_w_ }));
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
                    grad_view_(f, col_idx) = gradient(b, f, y, x);
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
            sum += grad_view_(f, col);
        }
        biases_grad_(f, 0) = sum;
    }

    Tensor::matmul(grad_view_, col_buffer_, filters_grad_, false, true);

    Tensor::matmul(filters_, grad_view_, grad_col_, true, false);

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
                            grad_input_(b, c, y + ky, x + kx) += grad_col_(row_idx, col_idx);
                        }
                    }
                }
            }
        }
    }

    filters_.reshape(Shape({ out_c_, in_c_, k_size_, k_size_ }));
    filters_grad_.reshape(Shape({ out_c_, in_c_, k_size_, k_size_ }));

    return grad_input_;
}

void ConvLayer::clear_gradients()
{
    filters_grad_.fill(0.0f);
    biases_grad_.fill(0.0f);
}

Shape3D ConvLayer::get_output_shape(const Shape3D &input_shape) const
{
    return { out_c_, input_shape.height - k_size_ + 1,
             input_shape.width - k_size_ + 1 };
}

void ConvLayer::save(std::ostream &os) const
{
    uint32_t marker = make_marker("CONV");
    os.write(reinterpret_cast<const char *>(&marker), sizeof(marker));

    uint64_t in_h = in_h_, in_w = in_w_, in_c = in_c_, k_size = k_size_,
             out_c = out_c_;
    os.write(reinterpret_cast<const char *>(&in_h), sizeof(in_h));
    os.write(reinterpret_cast<const char *>(&in_w), sizeof(in_w));
    os.write(reinterpret_cast<const char *>(&in_c), sizeof(in_c));
    os.write(reinterpret_cast<const char *>(&k_size), sizeof(k_size));
    os.write(reinterpret_cast<const char *>(&out_c), sizeof(out_c));

    filters_.save(os);
    biases_.save(os);
}

void ConvLayer::load(std::istream &is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    if (marker != make_marker("CONV"))
        throw std::runtime_error("Arch mismatch in ConvLayer load");

    uint64_t in_h, in_w, in_c, k_size, out_c;
    is.read(reinterpret_cast<char *>(&in_h), sizeof(in_h));
    is.read(reinterpret_cast<char *>(&in_w), sizeof(in_w));
    is.read(reinterpret_cast<char *>(&in_c), sizeof(in_c));
    is.read(reinterpret_cast<char *>(&k_size), sizeof(k_size));
    is.read(reinterpret_cast<char *>(&out_c), sizeof(out_c));

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