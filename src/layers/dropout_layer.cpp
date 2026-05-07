#include "layers/dropout_layer.hpp"

#include <cassert>
#include <stdexcept>

DropoutLayer::DropoutLayer(scalar_t ratio)
    : ratio_(ratio)
{}

/*
 * DROPOUT LAYER MAPPING & TERMINOLOGY:
 * 
 * 1. INPUT (X): [Batch x Features]
 *    Matrix of activations from the previous layer.
 * 
 * 2. RATIO (p): Scalar (e.g., 0.5)
 *    Probability of dropping out a unit.
 * 
 * 3. MASK (M): [Batch x Features]
 *    Binary mask where each element is 0 with probability p, and 1/(1-p) otherwise
 *    (inverted dropout to keep the expected value of activations unchanged).
 * 
 * 4. OUTPUT (Y): [Batch x Features]
 *    Result: Y = X * M (element-wise).
 */

const Tensor& DropoutLayer::forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool is_training) const
{
    if (input.rank() < 2)
        throw std::invalid_argument("Runtime error: DropoutLayer requires at least a 2D tensor (Batch, Features).");

    /*
     * STEP 1: TEST MODE
     * -----------------
     * During inference, dropout is deactivated.
     * The input is returned as-is (no scaling needed thanks to inverted dropout).
     */
    if (!is_training)
        return input;

    DropoutContext* dropout_ctx = get_context<DropoutContext>(ctx);

    dropout_ctx->mask.reshape(input.shape());
    dropout_ctx->output.reshape(input.shape());

    /*
     * STEP 2: TRAINING MODE (INVERTED DROPOUT)
     * ----------------------------------------
     * Generate a mask and scale the kept activations by 1/(1-p).
     * This ensures the expected value of the output is the same as the input.
     */
    const scalar_t scale = 1.0f / (1.0f - ratio_);
    dropout_ctx->mask.fill_random_mask(ratio_, scale);

    /*
     * STEP 3: APPLY MASK
     * ------------------
     * Element-wise multiplication of the input by the generated mask.
     */
    Tensor::hadamard_product(input, dropout_ctx->mask, dropout_ctx->output);

    return dropout_ctx->output;
}

const Tensor& DropoutLayer::backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx, [[maybe_unused]] bool is_training)
{
    /*
     * BACKPROPAGATION OVERVIEW:
     * -------------------------
     * Dropout is an element-wise multiplication by a mask M.
     * Y = X * M  =>  dY/dX = M
     * Chain Rule: dL/dX = dL/dY * M (element-wise)
     */
    assert(is_training && "Backward must only be called during training!");

    DropoutContext* dropout_ctx = get_context<DropoutContext>(ctx);

    /*
     * STEP 1: APPLY MASK TO GRADIENT
     * ------------------------------
     * Only gradients corresponding to neurons that were NOT dropped are kept.
     * They are also scaled by the same factor 1/(1-p).
     */
    Tensor::hadamard_product(gradient, dropout_ctx->mask, dropout_ctx->grad_input);

    return dropout_ctx->grad_input;
}

void DropoutLayer::save(std::ostream& os) const
{
    uint32_t marker = make_marker("DROP");
    os.write(reinterpret_cast<const char*>(&marker), sizeof(marker));
    os.write(reinterpret_cast<const char*>(&ratio_), sizeof(ratio_));
}

void DropoutLayer::load(std::istream& is)
{
    uint32_t marker;
    scalar_t r;
    is.read(reinterpret_cast<char*>(&marker), sizeof(marker));
    is.read(reinterpret_cast<char*>(&r), sizeof(r));
    if (marker != make_marker("DROP"))
        throw std::runtime_error("Arch mismatch in DropoutLayer binary load");
    ratio_ = r;
}

nlohmann::json DropoutLayer::get_config() const
{
    return { { "type", "Dropout" }, { "ratio", ratio_ } };
}

Shape DropoutLayer::get_output_shape(const Shape& input_shape) const
{
    if (input_shape.rank() < 2) {
        throw std::invalid_argument("Architecture error: DropoutLayer requires at least a 2D input (Batch, Features).");
    }
    return input_shape;
}

Shape DropoutLayer::get_input_shape(const Shape& output_shape) const
{
    return output_shape;
}