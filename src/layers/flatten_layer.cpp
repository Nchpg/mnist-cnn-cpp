#include "layers/flatten_layer.hpp"

#include <stdexcept>

FlattenLayer::FlattenLayer(size_t channels, size_t height, size_t width)
    : input_channels_(channels)
    , input_height_(height)
    , input_width_(width)
{}

/*
 * FLATTEN LAYER MAPPING & TERMINOLOGY:
 *
 * 1. INPUT (X): [Batch x Channels x Height x Width]
 *    4D Tensor from a Convolutional or Pooling layer.
 *
 * 2. OUTPUT (Y): [Batch x (Channels * Height * Width)]
 *    2D Tensor where all spatial and channel dimensions are collapsed into one.
 *    Used to transition to Dense (Fully Connected) layers.
 */

const Tensor& FlattenLayer::forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool /*is_training*/) const
{
    if (input.rank() != 4)
        throw std::invalid_argument("Runtime error: FlattenLayer expected a 4D tensor.");

    FlattenContext* flatten_ctx = get_context<FlattenContext>(ctx);

    const Shape input_shape = input.shape();

    /*
     * STEP 1: DATA COPY
     * -----------------
     * We copy the entire raw data from the input tensor.
     * Flattening doesn't change the values, only their logical organization.
     */
    flatten_ctx->output.data() = input.data();

    /*
     * STEP 2: RESHAPE
     * ---------------
     * Update the metadata (shape and strides) to reflect the 2D "flattened" view.
     */
    flatten_ctx->output.reshape(get_output_shape(input_shape));

    return flatten_ctx->output;
}

const Tensor& FlattenLayer::backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx,
                                     [[maybe_unused]] bool is_training)
{
    /*
     * BACKPROPAGATION OVERVIEW:
     * -------------------------
     * Flattening is essentially a reshape operation. Since it is a
     * identity mapping of the values, the gradient dL/dX is identical
     * to dL/dY, simply reshaped to the input structure.
     */
    FlattenContext* flatten_ctx = get_context<FlattenContext>(ctx);

    const Shape output_shape = gradient.shape();

    /*
     * STEP 1: GRADIENT DATA COPY
     * --------------------------
     * Copy the incoming flattened gradient to the input-shaped tensor.
     */
    flatten_ctx->grad_input.data() = gradient.data();

    /*
     * STEP 2: RESHAPE
     * ---------------
     * Map the [Batch x Features] gradient back to [Batch x C x H x W].
     */
    flatten_ctx->grad_input.reshape(get_input_shape(output_shape));

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