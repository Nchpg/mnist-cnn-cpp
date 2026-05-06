#include "layers/batchnorm_layer.hpp"

#include <cassert>

BatchNormLayer::BatchNormLayer(size_t channels, size_t spatial_size)
    : channels_(channels)
    , spatial_size_(spatial_size)
{
    gamma_.reshape(Shape({ channels, 1 }), 1.0f);
    beta_.reshape(Shape({ channels, 1 }), 0.0f);
    running_mean_.reshape(Shape({ channels, 1 }), 0.0f);
    running_var_.reshape(Shape({ channels, 1 }), 1.0f);
}

const Tensor &BatchNormLayer::forward(const Tensor &input,
                                      std::unique_ptr<LayerContext> &ctx,
                                      bool is_training) const
{
    if (!ctx)
    {
        ctx = std::make_unique<BatchNormContext>();
    }
    auto *bn_ctx = static_cast<BatchNormContext *>(ctx.get());

    size_t batch_size = input.shape()[0];

    if (bn_ctx->normalized.shape().rank() == 0
        || bn_ctx->normalized.shape()[0] != batch_size)
    {
        bn_ctx->normalized.reshape(input.shape());
        bn_ctx->saved_mean.reshape(Shape({ channels_, 1 }));
        bn_ctx->saved_var.reshape(Shape({ channels_, 1 }));
    }

    size_t height = input.shape()[2];
    size_t width = input.shape()[3];
    scalar_t N = static_cast<scalar_t>(spatial_size_ * batch_size);

#pragma omp parallel for if (channels_ > 1)
    for (size_t c = 0; c < channels_; ++c)
    {
        if (is_training)
        {
            scalar_t sum = 0.0f;
            for (size_t b = 0; b < batch_size; ++b)
            {
                for (size_t y = 0; y < height; ++y)
                {
                    for (size_t x = 0; x < width; ++x)
                    {
                        sum += input(b, c, y, x);
                    }
                }
            }
            bn_ctx->saved_mean(c, 0) = sum / N;

            scalar_t var_sum = 0.0f;
            for (size_t b = 0; b < batch_size; ++b)
            {
                for (size_t y = 0; y < height; ++y)
                {
                    for (size_t x = 0; x < width; ++x)
                    {
                        scalar_t diff =
                            input(b, c, y, x) - bn_ctx->saved_mean(c, 0);
                        var_sum += diff * diff;
                    }
                }
            }
            bn_ctx->saved_var(c, 0) = var_sum / N;

            scalar_t corrected_var = bn_ctx->saved_var(c, 0);
            if (N > 1)
            {
                corrected_var = bn_ctx->saved_var(c, 0) * N / (N - 1.0f);
            }

            running_mean_(c, 0) = momentum_ * running_mean_(c, 0)
                + (1.0f - momentum_) * bn_ctx->saved_mean(c, 0);
            running_var_(c, 0) = momentum_ * running_var_(c, 0)
                + (1.0f - momentum_) * corrected_var;
        }

        scalar_t mean_to_use =
            is_training ? bn_ctx->saved_mean(c, 0) : running_mean_(c, 0);
        scalar_t var_to_use =
            is_training ? bn_ctx->saved_var(c, 0) : running_var_(c, 0);
        scalar_t std_inv = 1.0f / std::sqrt(var_to_use + epsilon_);

        for (size_t b = 0; b < batch_size; ++b)
        {
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; ++x)
                {
                    scalar_t x_hat =
                        (input(b, c, y, x) - mean_to_use) * std_inv;
                    bn_ctx->normalized(b, c, y, x) = x_hat;
                }
            }
        }
    }

    for (size_t c = 0; c < channels_; ++c)
    {
        scalar_t gamma_c = gamma_(c, 0);
        scalar_t beta_c = beta_(c, 0);

        for (size_t b = 0; b < batch_size; ++b)
        {
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; ++x)
                {
                    scalar_t x_hat = bn_ctx->normalized(b, c, y, x);
                    bn_ctx->normalized(b, c, y, x) = x_hat * gamma_c + beta_c;
                }
            }
        }
    }

    return bn_ctx->normalized;
}

const Tensor &BatchNormLayer::backward(const Tensor &gradient,
                                       std::unique_ptr<LayerContext> &ctx,
                                       bool is_training)
{
    assert(is_training
           && "Backward doit uniquement etre appele durant l'entrainement !");
    auto *bn_ctx = static_cast<BatchNormContext *>(ctx.get());

    size_t batch_size = gradient.shape()[0];
    size_t height = gradient.shape()[2];
    size_t width = gradient.shape()[3];
    scalar_t N = static_cast<scalar_t>(spatial_size_ * batch_size);

    if (bn_ctx->grad_input.shape().rank() == 0
        || bn_ctx->grad_input.shape()[0] != batch_size)
    {
        bn_ctx->grad_input.reshape(gradient.shape());
    }

    if (grad_gamma_.shape().rank() == 0)
    {
        grad_gamma_.reshape(gamma_.shape());
        grad_beta_.reshape(beta_.shape());
    }

#pragma omp parallel for if (channels_ > 1)
    for (size_t c = 0; c < channels_; ++c)
    {
        scalar_t sum_dy = 0.0f;
        scalar_t sum_dy_xhat = 0.0f;

        for (size_t b = 0; b < batch_size; ++b)
        {
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; ++x)
                {
                    scalar_t dy = gradient(b, c, y, x);
                    sum_dy += dy;
                    sum_dy_xhat += dy * bn_ctx->normalized(b, c, y, x);
                }
            }
        }

        grad_beta_(c, 0) = sum_dy;
        grad_gamma_(c, 0) = sum_dy_xhat;

        scalar_t gamma_c = gamma_(c, 0);
        scalar_t std_inv = 1.0f / std::sqrt(bn_ctx->saved_var(c, 0) + epsilon_);
        scalar_t coef = gamma_c * std_inv;

        for (size_t b = 0; b < batch_size; ++b)
        {
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; ++x)
                {
                    scalar_t dy = gradient(b, c, y, x);
                    scalar_t x_hat = bn_ctx->normalized(b, c, y, x);

                    scalar_t dx =
                        coef * (dy - (sum_dy / N) - x_hat * (sum_dy_xhat / N));
                    bn_ctx->grad_input(b, c, y, x) = dx;
                }
            }
        }
    }

    return bn_ctx->grad_input;
}

void BatchNormLayer::clear_gradients()
{
    if (grad_gamma_.size() > 0)
        grad_gamma_.fill(0.0f);
    if (grad_beta_.size() > 0)
        grad_beta_.fill(0.0f);
}

void BatchNormLayer::save(std::ostream &os) const
{
    uint32_t marker = make_marker("BNRM");
    os.write(reinterpret_cast<const char *>(&marker), sizeof(marker));

    uint64_t channels = channels_, spatial_size = spatial_size_;
    os.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
    os.write(reinterpret_cast<const char *>(&spatial_size),
             sizeof(spatial_size));

    gamma_.save(os);
    beta_.save(os);
    running_mean_.save(os);
    running_var_.save(os);
}

void BatchNormLayer::load(std::istream &is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    if (marker != make_marker("BNRM"))
        throw std::runtime_error("Arch mismatch in BatchNormLayer load");

    uint64_t channels, spatial_size;
    is.read(reinterpret_cast<char *>(&channels), sizeof(channels));
    is.read(reinterpret_cast<char *>(&spatial_size), sizeof(spatial_size));

    channels_ = channels;
    spatial_size_ = spatial_size;

    gamma_.load(is);
    beta_.load(is);
    running_mean_.load(is);
    running_var_.load(is);
}

nlohmann::json BatchNormLayer::get_config() const
{
    return { { "type", "BatchNorm" } };
}

Shape3D BatchNormLayer::get_output_shape(const Shape3D &input_shape) const
{
    return input_shape;
}