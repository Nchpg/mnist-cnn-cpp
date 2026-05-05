#include "layers/batchnorm_layer.hpp"

#include <cmath>
#include <fstream>
#include <omp.h>
#include <stdexcept>

BatchNormLayer::BatchNormLayer(size_t channels, size_t spatial_size)
    : channels_(channels)
    , spatial_size_(spatial_size)
    , gamma_(channels, 1, 1.0f)
    , beta_(channels, 1, 0.0f)
    , grad_gamma_(channels, 1)
    , grad_beta_(channels, 1)
    , running_mean_(channels, 1, 0.0f)
    , running_var_(channels, 1, 1.0f)
    , saved_mean_(channels, 1)
    , saved_var_(channels, 1)
    , normalized_()
    , output_()
{
    clear_gradients();
}

const Matrix &BatchNormLayer::forward(const Matrix &input)
{
    if (input.rows() != channels_ * spatial_size_)
    {
        throw std::invalid_argument("BatchNormLayer: input rows mismatch");
    }
    size_t batch_size = input.cols();
    input_ptr_ = &input;

    if (normalized_.cols() != batch_size || normalized_.rows() != input.rows())
    {
        normalized_.reshape(input.rows(), batch_size);
        output_.reshape(input.rows(), batch_size);
    }

    scalar_t N = static_cast<scalar_t>(spatial_size_ * batch_size);

#pragma omp parallel for if (channels_ > 1)
    for (size_t c = 0; c < channels_; ++c)
    {
        if (is_training_)
        {
            scalar_t sum = 0.0f;
            for (size_t s = 0; s < spatial_size_; ++s)
            {
                size_t row = c * spatial_size_ + s;
                for (size_t b = 0; b < batch_size; ++b)
                {
                    sum += input(row, b);
                }
            }
            saved_mean_(c, 0) = sum / N;

            scalar_t var_sum = 0.0f;
            for (size_t s = 0; s < spatial_size_; ++s)
            {
                size_t row = c * spatial_size_ + s;
                for (size_t b = 0; b < batch_size; ++b)
                {
                    scalar_t diff = input(row, b) - saved_mean_(c, 0);
                    var_sum += diff * diff;
                }
            }
            saved_var_(c, 0) = var_sum / N;

            scalar_t corrected_var = saved_var_(c, 0);
            if (N > 1)
            {
                corrected_var = saved_var_(c, 0) * N / (N - 1.0f);
            }

            running_mean_(c, 0) = momentum_ * running_mean_(c, 0)
                + (1.0f - momentum_) * saved_mean_(c, 0);
            running_var_(c, 0) = momentum_ * running_var_(c, 0)
                + (1.0f - momentum_) * corrected_var;
        }

        scalar_t mean_to_use =
            is_training_ ? saved_mean_(c, 0) : running_mean_(c, 0);
        scalar_t var_to_use =
            is_training_ ? saved_var_(c, 0) : running_var_(c, 0);
        scalar_t std_inv = 1.0f / std::sqrt(var_to_use + epsilon_);
        scalar_t gamma_c = gamma_(c, 0);
        scalar_t beta_c = beta_(c, 0);

        for (size_t s = 0; s < spatial_size_; ++s)
        {
            size_t row = c * spatial_size_ + s;
            for (size_t b = 0; b < batch_size; ++b)
            {
                scalar_t x_hat = (input(row, b) - mean_to_use) * std_inv;
                normalized_(row, b) = x_hat;
                output_(row, b) = x_hat * gamma_c + beta_c;
            }
        }
    }

    return output_;
}

const Matrix &BatchNormLayer::backward(const Matrix &gradient)
{
    size_t batch_size = normalized_.cols();
    scalar_t N = static_cast<scalar_t>(spatial_size_ * batch_size);

#pragma omp parallel for if (channels_ > 1)
    for (size_t c = 0; c < channels_; ++c)
    {
        scalar_t sum_dy = 0.0f;
        scalar_t sum_dy_xhat = 0.0f;

        for (size_t s = 0; s < spatial_size_; ++s)
        {
            size_t row = c * spatial_size_ + s;
            for (size_t b = 0; b < batch_size; ++b)
            {
                scalar_t dy = gradient(row, b);
                sum_dy += dy;
                sum_dy_xhat += dy * normalized_(row, b);
            }
        }

        grad_beta_(c, 0) = sum_dy;
        grad_gamma_(c, 0) = sum_dy_xhat;

        scalar_t gamma_c = gamma_(c, 0);
        scalar_t std_inv = 1.0f / std::sqrt(saved_var_(c, 0) + epsilon_);
        scalar_t coef = gamma_c * std_inv;

        for (size_t s = 0; s < spatial_size_; ++s)
        {
            size_t row = c * spatial_size_ + s;
            for (size_t b = 0; b < batch_size; ++b)
            {
                scalar_t dy = gradient(row, b);
                scalar_t x_hat = normalized_(row, b);

                scalar_t dx =
                    coef * (dy - (sum_dy / N) - x_hat * (sum_dy_xhat / N));
                output_(row, b) = dx;
            }
        }
    }

    return output_;
}

std::vector<Parameter> BatchNormLayer::get_parameters()
{
    return { { &gamma_, &grad_gamma_ }, { &beta_, &grad_beta_ } };
}

void BatchNormLayer::clear_gradients()
{
    grad_gamma_.fill(0.0f);
    grad_beta_.fill(0.0f);
}

void BatchNormLayer::set_training(bool training)
{
    is_training_ = training;
}

void BatchNormLayer::save(std::ostream &os) const
{
    os.write(reinterpret_cast<const char *>(&channels_), sizeof(channels_));
    os.write(reinterpret_cast<const char *>(&spatial_size_),
             sizeof(spatial_size_));
    os.write(reinterpret_cast<const char *>(gamma_.data()),
             channels_ * sizeof(scalar_t));
    os.write(reinterpret_cast<const char *>(beta_.data()),
             channels_ * sizeof(scalar_t));
    os.write(reinterpret_cast<const char *>(running_mean_.data()),
             channels_ * sizeof(scalar_t));
    os.write(reinterpret_cast<const char *>(running_var_.data()),
             channels_ * sizeof(scalar_t));
}

void BatchNormLayer::load(std::istream &is)
{
    size_t channels, spatial;
    is.read(reinterpret_cast<char *>(&channels), sizeof(channels));
    is.read(reinterpret_cast<char *>(&spatial), sizeof(spatial));
    if (channels != channels_ || spatial != spatial_size_)
    {
        throw std::runtime_error("Arch mismatch in BatchNormLayer load");
    }
    is.read(reinterpret_cast<char *>(gamma_.data()),
            channels_ * sizeof(scalar_t));
    is.read(reinterpret_cast<char *>(beta_.data()),
            channels_ * sizeof(scalar_t));
    is.read(reinterpret_cast<char *>(running_mean_.data()),
            channels_ * sizeof(scalar_t));
    is.read(reinterpret_cast<char *>(running_var_.data()),
            channels_ * sizeof(scalar_t));
}

nlohmann::json BatchNormLayer::get_config() const
{
    return { { "type", "BatchNorm" } };
}

Shape3D BatchNormLayer::get_output_shape(const Shape3D &input_shape) const
{
    return input_shape;
}
