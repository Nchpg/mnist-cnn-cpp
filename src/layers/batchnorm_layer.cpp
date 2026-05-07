#include "layers/batchnorm_layer.hpp"

#include <cassert>
#include <stdexcept>

BatchNormLayer::BatchNormLayer(size_t channels, size_t height, size_t width)
    : channels_(channels)
    , height_(height)
    , width_(width)
{
    gamma_.reshape(Shape({ channels }), 1.0f);
    beta_.reshape(Shape({ channels }), 0.0f);
    running_mean_.reshape(Shape({ channels }), 0.0f);
    running_var_.reshape(Shape({ channels }), 1.0f);
}

/*
 * BATCH NORMALIZATION LAYER MAPPING & TERMINOLOGY:
 *
 * 1. INPUT (X): [Batch x Channels x Height x Width]
 *    Spatial feature map. Normalization is performed per channel over the
 *    batch and spatial dimensions.
 *
 * 2. PARAMETERS (gamma, beta): [Channels x 1]
 *    Learnable scaling and shifting factors applied after normalization.
 *
 * 3. RUNNING STATS (mean, var): [Channels x 1]
 *    Moving averages of channel statistics for inference (evaluation mode).
 *
 * 4. OUTPUT (Y): [Batch x Channels x Height x Width]
 *    Normalized and scaled feature map:
 *    Y = Gamma * (X - Mean) / sqrt(Var + eps) + Beta
 */

const Tensor& BatchNormLayer::forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool is_training) const
{
    if (input.rank() != 4)
        throw std::invalid_argument("Runtime error: BatchNormLayer expected a 4D tensor.");

    const Shape input_shape = input.shape();
    const size_t batch_size = input_shape.batch();

    if (input_shape.channels() != channels_ || input_shape.height() != height_ || input_shape.width() != width_)
        throw std::invalid_argument("Architecture error: Input dimensions mismatch in BatchNormLayer.");

    BatchNormContext* bn_ctx = get_context<BatchNormContext>(ctx);

    bn_ctx->x_hat.reshape(input_shape);
    bn_ctx->normalized.reshape(input_shape);
    bn_ctx->saved_mean.reshape(running_mean_.shape());
    bn_ctx->saved_var.reshape(running_var_.shape());

    const scalar_t N = static_cast<scalar_t>(height_ * width_ * batch_size);

#pragma omp parallel for if (channels_ > 1)
    for (size_t c = 0; c < channels_; ++c)
    {
        if (is_training)
        {
            /*
             * STEP 1: COMPUTE STATISTICS (SINGLE PASS)
             * ----------------------------------------
             * Compute both sum and sum of squares in one pass to derive mean and variance.
             */
            scalar_t sum = 0.0f;
            scalar_t sq_sum = 0.0f;
            for (size_t b = 0; b < batch_size; ++b)
            {
                for (size_t y = 0; y < height_; ++y)
                {
                    for (size_t x = 0; x < width_; ++x)
                    {
                        scalar_t val = input(b, c, y, x);
                        sum += val;
                        sq_sum += val * val;
                    }
                }
            }
            bn_ctx->saved_mean(c) = sum / N;
            // Var = E[X^2] - (E[X])^2
            bn_ctx->saved_var(c) = (sq_sum / N) - (bn_ctx->saved_mean(c) * bn_ctx->saved_mean(c));

            /*
             * STEP 2: UPDATE RUNNING STATS (BESSEL'S CORRECTION)
             * --------------------------------------------------
             * Apply Bessel's correction (N / (N - 1)) to estimate the population
             * variance from the sample variance, ensuring stable running statistics.
             */
            scalar_t corrected_var = (N > 1) ? bn_ctx->saved_var(c) * N / (N - 1.0f) : bn_ctx->saved_var(c);
            running_mean_(c) = momentum_ * running_mean_(c) + (1.0f - momentum_) * bn_ctx->saved_mean(c);
            running_var_(c) = momentum_ * running_var_(c) + (1.0f - momentum_) * corrected_var;
        }

        /*
         * STEP 3: NORMALIZE & SCALE
         * -------------------------
         * Apply: Y = (X - Mean) / sqrt(Var + eps) * Gamma + Beta
         */
        scalar_t mean_to_use = is_training ? bn_ctx->saved_mean(c) : running_mean_(c);
        scalar_t var_to_use = is_training ? bn_ctx->saved_var(c) : running_var_(c);
        scalar_t std_inv = 1.0f / std::sqrt(var_to_use + epsilon_);

        scalar_t gamma_c = gamma_(c);
        scalar_t beta_c = beta_(c);

        for (size_t b = 0; b < batch_size; ++b)
        {
            for (size_t y = 0; y < height_; ++y)
            {
                for (size_t x = 0; x < width_; ++x)
                {
                    scalar_t val_x_hat = (input(b, c, y, x) - mean_to_use) * std_inv;
                    bn_ctx->x_hat(b, c, y, x) = val_x_hat;
                    bn_ctx->normalized(b, c, y, x) = val_x_hat * gamma_c + beta_c;
                }
            }
        }
    }

    return bn_ctx->normalized;
}

const Tensor& BatchNormLayer::backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx,
                                       [[maybe_unused]] bool is_training)
{
    /*
     * BACKPROPAGATION DERIVATION (CHAIN RULE):
     * ----------------------------------------
     * Forward: Y = gamma * X_hat + beta
     * where X_hat = (X - mean) * sigma_inv, sigma_inv = 1 / sqrt(var + eps)
     *
     * 1. Gradients for Parameters (Gamma, Beta):
     *    dY/dGamma = X_hat.
     *    -> dL/dGamma = dL/dY * dY/dGamma
     *                 = sum(dL/dY * X_hat) Since gamma is shared across the channel
     *
     *    dY/dBeta = 1
     *    -> dL/dBeta = dL/dY * dY/dBeta
     *                = sum(dL/dY) Since beta is shared across the channel
     *
     * 2. Gradient for Input (dL/dX):
     *    dL/dX = dL/dY * dY/dX_hat * dX_hat/dX
     *          = dL/dY * gamma * dX_hat/dX
     *          = ...
     *          = (gamma * sigma_inv / N) * [N * dL/dY - sum(dL/dY) - X_hat * sum(dL/dY * X_hat)]
     *          = (gamma * sigma_inv / N) * [N * dL/dY - dL/dBeta - X_hat * dL/dGamma]
     *            Since mean and sigma depend in X
     */
    assert(is_training && "Backward must only be called during training!");

    BatchNormContext* bn_ctx = get_context<BatchNormContext>(ctx);

    const Shape input_shape = gradient.shape();
    const size_t batch_size = input_shape.batch();
    const scalar_t N = static_cast<scalar_t>(height_ * width_ * batch_size);

    bn_ctx->grad_input.reshape(input_shape);
    grad_gamma_.reshape(gamma_.shape());
    grad_beta_.reshape(beta_.shape());

#pragma omp parallel for if (channels_ > 1)
    for (size_t c = 0; c < channels_; ++c)
    {
        /*
         * STEP 1: COMPUTE PARAMETER GRADIENTS
         * -----------------------------------
         * dL/dGamma = sum(dL/dY * X_hat)
         * dL/dBeta  = sum(dL/dY)
         */
        scalar_t sum_dy = 0.0f;
        scalar_t sum_dy_xhat = 0.0f;

        for (size_t b = 0; b < batch_size; ++b)
        {
            for (size_t y = 0; y < height_; ++y)
            {
                for (size_t x = 0; x < width_; ++x)
                {
                    scalar_t dy = gradient(b, c, y, x);
                    sum_dy += dy;
                    sum_dy_xhat += dy * bn_ctx->x_hat(b, c, y, x);
                }
            }
        }

        grad_beta_(c) = sum_dy;
        grad_gamma_(c) = sum_dy_xhat;

        /*
         * STEP 2: COMPUTE INPUT GRADIENT (dL/dX)
         * --------------------------------------
         * Using the derived formula:
         * dL/dX = (gamma * sigma_inv / N) * [N * dy - sum(dy) - x_hat * sum(dy * x_hat)]
         *       = (gamma * sigma_inv / N) * [N * dL/dY - dL/dBeta - X_hat * dL/dGamma]
         */
        scalar_t gamma_c = gamma_(c);
        scalar_t std_inv = 1.0f / std::sqrt(bn_ctx->saved_var(c) + epsilon_);
        scalar_t coef = gamma_c * std_inv / N;

        for (size_t b = 0; b < batch_size; ++b)
        {
            for (size_t y = 0; y < height_; ++y)
            {
                for (size_t x = 0; x < width_; ++x)
                {
                    scalar_t dy = gradient(b, c, y, x);
                    scalar_t val_x_hat = bn_ctx->x_hat(b, c, y, x);

                    // dL/dX_{i} = coef * [N * dL/dY_{i} - dL/dBeta_{i} - X_hat_{b,c,y,x} * dL/dGamma_{i}]
                    bn_ctx->grad_input(b, c, y, x) = coef * (N * dy - sum_dy - val_x_hat * sum_dy_xhat);
                }
            }
        }
    }

    return bn_ctx->grad_input;
}

void BatchNormLayer::clear_gradients()
{
    grad_gamma_.fill(0.0f);
    grad_beta_.fill(0.0f);
}

void BatchNormLayer::save(std::ostream& os) const
{
    uint32_t marker = make_marker("BNRM");
    os.write(reinterpret_cast<const char*>(&marker), sizeof(marker));

    uint64_t channels = channels_, height = height_, width = width_;
    os.write(reinterpret_cast<const char*>(&channels), sizeof(channels));
    os.write(reinterpret_cast<const char*>(&height), sizeof(height));
    os.write(reinterpret_cast<const char*>(&width), sizeof(width));

    gamma_.save(os);
    beta_.save(os);
    running_mean_.save(os);
    running_var_.save(os);
}

void BatchNormLayer::load(std::istream& is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char*>(&marker), sizeof(marker));
    if (marker != make_marker("BNRM"))
        throw std::runtime_error("Arch mismatch in BatchNormLayer load");

    uint64_t channels, height, width;
    is.read(reinterpret_cast<char*>(&channels), sizeof(channels));
    is.read(reinterpret_cast<char*>(&height), sizeof(height));
    is.read(reinterpret_cast<char*>(&width), sizeof(width));

    channels_ = channels;
    height_ = height;
    width_ = width;

    gamma_.load(is);
    beta_.load(is);
    running_mean_.load(is);
    running_var_.load(is);
}

nlohmann::json BatchNormLayer::get_config() const
{
    return { { "type", "BatchNorm" } };
}

Shape BatchNormLayer::get_output_shape(const Shape& input_shape) const
{
    if (input_shape.rank() != 4)
    {
        throw std::invalid_argument(
            "Architecture error: BatchNormLayer requires a 4D input (Batch, Channels, Height, Width).");
    }
    if (input_shape.channels() != channels_ || input_shape.height() != height_ || input_shape.width() != width_)
    {
        throw std::invalid_argument(
            "Architecture error: BatchNormLayer input dimensions do not match layer configuration.");
    }
    return input_shape;
}

Shape BatchNormLayer::get_input_shape(const Shape& output_shape) const
{
    return output_shape;
}