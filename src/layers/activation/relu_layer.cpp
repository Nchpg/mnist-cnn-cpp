#include "layers/activation/relu_layer.hpp"

#include <stdexcept>

#include "layers/activation/activation.hpp"

ReluLayer::ReluLayer()
{}

const Tensor& ReluLayer::forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool is_training) const
{
    if (input.rank() < 2)
    {
        throw std::invalid_argument("Runtime error: ReluLayer requires at least a 2D tensor (Batch, Features).");
    }
    if (!ctx)
    {
        ctx = std::make_unique<ReluContext>();
    }
    auto* relu_ctx = static_cast<ReluContext*>(ctx.get());

    if (is_training)
    {
        relu_ctx->input = input;
    }

    relu_ctx->output.reshape(input.shape());
    Activation::relu(input, relu_ctx->output);

    return relu_ctx->output;
}

const Tensor& ReluLayer::backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx, bool is_training)
{
    (void)is_training;
    auto* relu_ctx = static_cast<ReluContext*>(ctx.get());
    relu_ctx->grad_input.reshape(gradient.shape());
    Activation::relu_backward(relu_ctx->input, gradient, relu_ctx->grad_input);
    return relu_ctx->grad_input;
}

void ReluLayer::save(std::ostream& os) const
{
    const uint32_t marker = make_marker(LAYER_MARKER);
    os.write(reinterpret_cast<const char*>(&marker), sizeof(marker));
}

void ReluLayer::load(std::istream& is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char*>(&marker), sizeof(marker));
    if (marker != make_marker(LAYER_MARKER))
        throw std::runtime_error("Architecture mismatch in ReluLayer load");
}

nlohmann::json ReluLayer::get_config() const
{
    return { { "type", "ReLU" } };
}

Shape ReluLayer::get_output_shape(const Shape& input_shape) const
{
    if (input_shape.rank() < 2)
    {
        throw std::invalid_argument("Architecture error: ReluLayer requires at least a 2D input (Batch, Features).");
    }
    return input_shape;
}

Shape ReluLayer::get_input_shape(const Shape& output_shape) const
{
    return output_shape;
}