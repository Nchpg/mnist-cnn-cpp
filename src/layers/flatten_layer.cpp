#include "layers/flatten_layer.hpp"

FlattenLayer::FlattenLayer(size_t channels, size_t height, size_t width)
    : input_channels_(channels)
    , input_height_(height)
    , input_width_(width)
{}

const Tensor &FlattenLayer::forward(const Tensor &input,
                                    std::unique_ptr<LayerContext> &ctx,
                                    bool is_training) const
{
    if (!ctx)
    {
        ctx = std::make_unique<FlattenContext>();
    }
    auto *flatten_ctx = static_cast<FlattenContext *>(ctx.get());

    size_t batch_size = input.shape()[0];
    flatten_ctx->output.reshape(
        Shape({ batch_size, input_channels_ * input_height_ * input_width_ }));

    std::copy(input.data().begin(), input.data().end(),
              flatten_ctx->output.data().begin());

    (void)is_training;
    return flatten_ctx->output;
}

const Tensor &FlattenLayer::backward(const Tensor &gradient,
                                     std::unique_ptr<LayerContext> &ctx,
                                     bool is_training)
{
    auto *flatten_ctx = static_cast<FlattenContext *>(ctx.get());
    size_t batch_size = gradient.shape()[0];
    flatten_ctx->grad_input.reshape(
        Shape({ batch_size, input_channels_, input_height_, input_width_ }));

    std::copy(gradient.data().begin(), gradient.data().end(),
              flatten_ctx->grad_input.data().begin());
    (void)is_training;
    return flatten_ctx->grad_input;
}

void FlattenLayer::save(std::ostream &os) const
{
    uint32_t marker = make_marker("FLAT");
    os.write(reinterpret_cast<const char *>(&marker), sizeof(marker));
    os.write(reinterpret_cast<const char *>(&input_channels_),
             sizeof(input_channels_));
    os.write(reinterpret_cast<const char *>(&input_height_),
             sizeof(input_height_));
    os.write(reinterpret_cast<const char *>(&input_width_),
             sizeof(input_width_));
}

void FlattenLayer::load(std::istream &is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    if (marker != make_marker("FLAT"))
        throw std::runtime_error("Architecture mismatch in FlattenLayer load");
    is.read(reinterpret_cast<char *>(&input_channels_),
            sizeof(input_channels_));
    is.read(reinterpret_cast<char *>(&input_height_), sizeof(input_height_));
    is.read(reinterpret_cast<char *>(&input_width_), sizeof(input_width_));
}

nlohmann::json FlattenLayer::get_config() const
{
    return { { "type", "Flatten" } };
}

Shape3D FlattenLayer::get_output_shape(const Shape3D &input_shape) const
{
    return { 1, 1,
             input_shape.channels * input_shape.height * input_shape.width };
}