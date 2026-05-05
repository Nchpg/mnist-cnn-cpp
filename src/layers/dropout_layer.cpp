#include "layers/dropout_layer.hpp"

#include <omp.h>

DropoutLayer::DropoutLayer(scalar_t ratio, std::mt19937 &gen)
    : ratio_(ratio)
{
    local_gen_ = gen;
}

void DropoutLayer::set_training(bool training)
{
    is_training_ = training;
}

const Matrix &DropoutLayer::forward(const Matrix &input)
{
    if (!is_training_)
    {
        return input;
    }

    if (mask_.rows() != input.rows() || mask_.cols() != input.cols())
    {
        mask_.reshape(input.rows(), input.cols());
        output_.reshape(input.rows(), input.cols());
    }

    scalar_t scale = 1.0f / (1.0f - ratio_);
    size_t n = input.size();
    const scalar_t *in_ptr = input.data();
    scalar_t *mask_ptr = mask_.data();
    scalar_t *out_ptr = output_.data();

#pragma omp parallel
    {
        std::mt19937 &local_gen = local_gen_;
        std::uniform_real_distribution<scalar_t> dist(0.0f, 1.0f);

#pragma omp for
        for (size_t i = 0; i < n; ++i)
        {
            if (dist(local_gen) > ratio_)
            {
                mask_ptr[i] = scale;
            }
            else
            {
                mask_ptr[i] = 0.0f;
            }
            out_ptr[i] = in_ptr[i] * mask_ptr[i];
        }
    }

    return output_;
}

const Matrix &DropoutLayer::backward(const Matrix &gradient)
{
    if (!is_training_)
    {
        return gradient;
    }

    if (grad_input_.rows() != gradient.rows()
        || grad_input_.cols() != gradient.cols())
    {
        grad_input_.reshape(gradient.rows(), gradient.cols());
    }

    size_t n = gradient.size();
    const scalar_t *grad_ptr = gradient.data();
    const scalar_t *mask_ptr = mask_.data();
    scalar_t *in_grad_ptr = grad_input_.data();

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i)
    {
        in_grad_ptr[i] = grad_ptr[i] * mask_ptr[i];
    }

    return grad_input_;
}

Shape3D DropoutLayer::get_output_shape(const Shape3D &input_shape) const
{
    return input_shape;
}

void DropoutLayer::save(std::ostream &os) const
{
    uint32_t marker = make_marker(LAYER_MARKER);
    os.write(reinterpret_cast<const char *>(&marker), sizeof(marker));
    os.write(reinterpret_cast<const char *>(&ratio_), sizeof(ratio_));
}

void DropoutLayer::load(std::istream &is)
{
    uint32_t marker;
    scalar_t r;
    is.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    is.read(reinterpret_cast<char *>(&r), sizeof(r));
    if (marker != make_marker(LAYER_MARKER))
        throw std::runtime_error("Arch mismatch in DropoutLayer binary load");
    ratio_ = r;
}

nlohmann::json DropoutLayer::get_config() const
{
    return { { "type", "Dropout" }, { "ratio", ratio_ } };
}