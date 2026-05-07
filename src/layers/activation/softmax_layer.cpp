#include "layers/activation/softmax_layer.hpp"

#include <stdexcept>

#include "layers/activation/activation.hpp"

SoftmaxLayer::SoftmaxLayer()
{}

SoftmaxLayer::~SoftmaxLayer()
{}

const Tensor& SoftmaxLayer::forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool is_training) const
{
    if (input.rank() < 2)
    {
        throw std::invalid_argument("Runtime error: SoftmaxLayer requires at least a 2D tensor (Batch, Features).");
    }
    if (!ctx)
    {
        ctx = std::make_unique<SoftmaxContext>();
    }
    auto* softmax_ctx = static_cast<SoftmaxContext*>(ctx.get());

    softmax_ctx->output.reshape(input.shape());
    Activation::softmax(input, softmax_ctx->output);

    (void)is_training;
    return softmax_ctx->output;
}

const Tensor& SoftmaxLayer::backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx, bool is_training)
{
    auto* softmax_ctx = static_cast<SoftmaxContext*>(ctx.get());
    softmax_ctx->grad_input.reshape(gradient.shape());
    Activation::softmax_backward(softmax_ctx->output, gradient, softmax_ctx->grad_input);
    (void)is_training;
    return softmax_ctx->grad_input;
}

nlohmann::json SoftmaxLayer::get_config() const
{
    return { { "type", "Softmax" } };
}

Shape SoftmaxLayer::get_output_shape(const Shape& input_shape) const
{
    if (input_shape.rank() < 2)
    {
        throw std::invalid_argument("Architecture error: SoftmaxLayer requires at least a 2D input (Batch, Features).");
    }
    return input_shape;
}

Shape SoftmaxLayer::get_input_shape(const Shape& output_shape) const
{
    return output_shape;
}

void SoftmaxLayer::save(std::ostream& os) const
{
    uint32_t marker = make_marker(LAYER_MARKER);
    os.write(reinterpret_cast<const char*>(&marker), sizeof(marker));
}

void SoftmaxLayer::load(std::istream& is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char*>(&marker), sizeof(marker));
    if (marker != make_marker(LAYER_MARKER))
        throw std::runtime_error("Architecture mismatch in SoftmaxLayer load");
}