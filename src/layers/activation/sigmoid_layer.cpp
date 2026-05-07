#include "layers/activation/sigmoid_layer.hpp"

#include <stdexcept>

#include "layers/activation/activation.hpp"

SigmoidLayer::SigmoidLayer()
{}

const Tensor& SigmoidLayer::forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool is_training) const
{
    if (input.rank() < 2)
    {
        throw std::invalid_argument("Runtime error: SigmoidLayer requires at least a 2D tensor (Batch, Features).");
    }
    if (!ctx)
    {
        ctx = std::make_unique<SigmoidContext>();
    }
    auto* sigmoid_ctx = static_cast<SigmoidContext*>(ctx.get());

    sigmoid_ctx->output.reshape(input.shape());
    Activation::sigmoid(input, sigmoid_ctx->output);

    (void)is_training;
    return sigmoid_ctx->output;
}

const Tensor& SigmoidLayer::backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx, bool is_training)
{
    auto* sigmoid_ctx = static_cast<SigmoidContext*>(ctx.get());
    sigmoid_ctx->grad_input.reshape(gradient.shape());
    Activation::sigmoid_backward(sigmoid_ctx->output, gradient, sigmoid_ctx->grad_input);
    (void)is_training;
    return sigmoid_ctx->grad_input;
}

void SigmoidLayer::save(std::ostream& os) const
{
    uint32_t marker = make_marker(LAYER_MARKER);
    os.write(reinterpret_cast<const char*>(&marker), sizeof(marker));
}

void SigmoidLayer::load(std::istream& is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char*>(&marker), sizeof(marker));
    if (marker != make_marker(LAYER_MARKER))
        throw std::runtime_error("Architecture mismatch in SigmoidLayer load");
}

nlohmann::json SigmoidLayer::get_config() const
{
    return { { "type", "Sigmoid" } };
}

Shape SigmoidLayer::get_output_shape(const Shape& input_shape) const
{
    if (input_shape.rank() < 2)
    {
        throw std::invalid_argument("Architecture error: SigmoidLayer requires at least a 2D input (Batch, Features).");
    }
    return input_shape;
}

Shape SigmoidLayer::get_input_shape(const Shape& output_shape) const
{
    return output_shape;
}