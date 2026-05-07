#include "layers/flatten_layer.hpp"

#include <stdexcept>

FlattenLayer::FlattenLayer(size_t channels, size_t height, size_t width)
    : input_channels_(channels)
    , input_height_(height)
    , input_width_(width)
{}

const Tensor& FlattenLayer::forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool is_training) const
{
    if (input.rank() != 4)
    {
        throw std::invalid_argument("Runtime error: FlattenLayer expected a 4D tensor.");
    }
    if (!ctx)
    {
        ctx = std::make_unique<FlattenContext>();
    }
    auto* flatten_ctx = static_cast<FlattenContext*>(ctx.get());

    size_t batch_size = input.shape()[0];
    size_t flat_size = input_channels_ * input_height_ * input_width_;

    flatten_ctx->output.reshape(Shape({ batch_size, flat_size }));

    std::copy_n(input.data_ptr(), input.size(), flatten_ctx->output.data_ptr());

    (void)is_training;
    return flatten_ctx->output;
}

const Tensor& FlattenLayer::backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx, bool is_training)
{
    auto* flatten_ctx = static_cast<FlattenContext*>(ctx.get());
    size_t batch_size = gradient.shape()[0];
    flatten_ctx->grad_input.reshape(Shape({ batch_size, input_channels_, input_height_, input_width_ }));

    std::copy(gradient.data().begin(), gradient.data().end(), flatten_ctx->grad_input.data().begin());
    (void)is_training;
    return flatten_ctx->grad_input;
}

void FlattenLayer::save(std::ostream& os) const
{
    uint32_t marker = make_marker("FLAT");
    os.write(reinterpret_cast<const char*>(&marker), sizeof(marker));
    uint64_t c = input_channels_, h = input_height_, w = input_width_;
    os.write(reinterpret_cast<const char*>(&c), sizeof(c));
    os.write(reinterpret_cast<const char*>(&h), sizeof(h));
    os.write(reinterpret_cast<const char*>(&w), sizeof(w));
}

void FlattenLayer::load(std::istream& is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char*>(&marker), sizeof(marker));
    if (marker != make_marker("FLAT"))
        throw std::runtime_error("Architecture mismatch in FlattenLayer load");
    uint64_t c, h, w;
    is.read(reinterpret_cast<char*>(&c), sizeof(c));
    is.read(reinterpret_cast<char*>(&h), sizeof(h));
    is.read(reinterpret_cast<char*>(&w), sizeof(w));
    input_channels_ = c;
    input_height_ = h;
    input_width_ = w;
}

nlohmann::json FlattenLayer::get_config() const
{
    return { { "type", "Flatten" } };
}

Shape FlattenLayer::get_output_shape(const Shape& input_shape) const
{
    if (input_shape.rank() != 4)
    {
        throw std::invalid_argument(
            "Architecture error: FlattenLayer requires a 4D input (Batch, Channels, Height, Width).");
    }
    if (input_shape.channels() != input_channels_ || input_shape.height() != input_height_
        || input_shape.width() != input_width_)
    {
        throw std::invalid_argument(
            "Architecture error: FlattenLayer input dimensions do not match layer configuration.");
    }
    return { input_shape.batch(), input_shape.channels() * input_shape.height() * input_shape.width() };
}

Shape FlattenLayer::get_input_shape(const Shape& output_shape) const
{
    return { output_shape.batch(), input_channels_, input_height_, input_width_ };
}