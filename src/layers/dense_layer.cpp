#include "layers/dense_layer.hpp"

#include <cassert>
#include <fstream>
#include <stdexcept>

DenseLayer::DenseLayer(size_t input_size, size_t output_size, std::mt19937& gen)
    : input_size_(input_size)
    , output_size_(output_size)
{
    weights_.reshape(Shape({ output_size_, input_size_ }));
    biases_.reshape(Shape({ output_size_, 1 }));

    const scalar_t std_dev = std::sqrt(2.0f / static_cast<scalar_t>(input_size));
    weights_.random_normal(std_dev, gen);
    biases_.fill(0.0f);
}

const Tensor& DenseLayer::forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool is_training) const
{
    if (!ctx)
    {
        ctx = std::make_unique<DenseContext>();
    }
    auto* dense_ctx = static_cast<DenseContext*>(ctx.get());

    const size_t batch_size = input.shape()[0];

    if (is_training)
    {
        dense_ctx->input_ptr = &input;
    }

    if (dense_ctx->activations.rank() != 2 || dense_ctx->activations.shape()[0] != batch_size
        || dense_ctx->activations.shape()[1] != output_size_)
    {
        dense_ctx->activations.reshape(Shape({ batch_size, output_size_ }));
    }

    Tensor::matmul(input, weights_, dense_ctx->activations, false, true);

    scalar_t* act_ptr = dense_ctx->activations.data_ptr();
    const scalar_t* bias_ptr = biases_.data_ptr();
#pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t i = 0; i < output_size_; ++i)
        {
            act_ptr[b * output_size_ + i] += bias_ptr[i];
        }
    }

    return dense_ctx->activations;
}

const Tensor& DenseLayer::backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx, bool is_training)
{
    assert(is_training && "Backward doit uniquement etre appele durant l'entrainement !");
    (void)is_training;
    auto* dense_ctx = static_cast<DenseContext*>(ctx.get());
    const size_t batch_size = gradient.shape()[0];

    if (dense_ctx->grad_input.size() == 0 || dense_ctx->grad_input.shape()[0] != batch_size)
    {
        dense_ctx->grad_input.reshape(Shape({ batch_size, input_size_ }));
    }

    Tensor::matmul(gradient, weights_, dense_ctx->grad_input, false, false);

    if (weights_grad_.size() == 0)
    {
        weights_grad_.reshape(weights_.shape());
        biases_grad_.reshape(biases_.shape());
    }
    assert(dense_ctx->input_ptr != nullptr);
    Tensor::matmul(gradient, *(dense_ctx->input_ptr), weights_grad_, true, false);

    biases_grad_.fill(0.0f);
    const scalar_t* grad_ptr = gradient.data_ptr();
    scalar_t* bias_grad_ptr = biases_grad_.data_ptr();
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t i = 0; i < output_size_; ++i)
        {
            bias_grad_ptr[i] += grad_ptr[b * output_size_ + i];
        }
    }

    return dense_ctx->grad_input;
}

void DenseLayer::clear_gradients()
{
    if (weights_grad_.size() > 0)
        weights_grad_.fill(0.0f);
    if (biases_grad_.size() > 0)
        biases_grad_.fill(0.0f);
}

void DenseLayer::save(std::ostream& os) const
{
    uint32_t marker = make_marker("DNSL");
    os.write(reinterpret_cast<const char*>(&marker), sizeof(marker));

    uint64_t in_size = input_size_, out_size = output_size_;
    os.write(reinterpret_cast<const char*>(&in_size), sizeof(in_size));
    os.write(reinterpret_cast<const char*>(&out_size), sizeof(out_size));

    weights_.save(os);
    biases_.save(os);
}

void DenseLayer::load(std::istream& is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char*>(&marker), sizeof(marker));
    if (marker != make_marker("DNSL"))
        throw std::runtime_error("Architecture mismatch in DenseLayer load");

    uint64_t in_size, out_size;
    is.read(reinterpret_cast<char*>(&in_size), sizeof(in_size));
    is.read(reinterpret_cast<char*>(&out_size), sizeof(out_size));

    input_size_ = in_size;
    output_size_ = out_size;

    weights_.load(is);
    biases_.load(is);
}

nlohmann::json DenseLayer::get_config() const
{
    return { { "type", "Dense" }, { "units", output_size_ } };
}

Shape3D DenseLayer::get_output_shape(const Shape3D& input_shape) const
{
    (void)input_shape;
    return { output_size_, 1, 1 };
}