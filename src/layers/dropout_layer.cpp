#include "layers/dropout_layer.hpp"

DropoutLayer::DropoutLayer(scalar_t ratio, std::mt19937 &gen)
    : ratio_(ratio)
    , local_gen_(gen)
{}

const Tensor &DropoutLayer::forward(const Tensor &input)
{
    if (!is_training_)
    {
        return input;
    }

    if (output_.shape() != input.shape())
    {
        mask_.reshape(input.shape());
        output_.reshape(input.shape());
    }

    scalar_t scale = 1.0f / (1.0f - ratio_);
    size_t n = input.size();
    const scalar_t *in_ptr = input.data_ptr();
    scalar_t *mask_ptr = mask_.data_ptr();
    scalar_t *out_ptr = output_.data_ptr();

    static thread_local std::mt19937 gen(local_gen_());
    std::uniform_real_distribution<scalar_t> dist(0.0f, 1.0f);

#pragma omp parallel for if (n > 10000)
    for (size_t i = 0; i < n; ++i)
    {
        if (dist(gen) > ratio_)
        {
            mask_ptr[i] = scale;
        }
        else
        {
            mask_ptr[i] = 0.0f;
        }
        out_ptr[i] = in_ptr[i] * mask_ptr[i];
    }

    return output_;
}

const Tensor &DropoutLayer::backward(const Tensor &gradient)
{
    if (!is_training_)
    {
        return gradient;
    }

    if (grad_input_.shape() != gradient.shape())
    {
        grad_input_.reshape(gradient.shape());
    }

    size_t n = gradient.size();
    const scalar_t *grad_ptr = gradient.data_ptr();
    const scalar_t *mask_ptr = mask_.data_ptr();
    scalar_t *in_grad_ptr = grad_input_.data_ptr();

#pragma omp parallel for if (n > 10000)
    for (size_t i = 0; i < n; ++i)
    {
        in_grad_ptr[i] = grad_ptr[i] * mask_ptr[i];
    }

    return grad_input_;
}

void DropoutLayer::save(std::ostream &os) const
{
    uint32_t marker = make_marker("DROP");
    os.write(reinterpret_cast<const char *>(&marker), sizeof(marker));
    os.write(reinterpret_cast<const char *>(&ratio_), sizeof(ratio_));
}

void DropoutLayer::load(std::istream &is)
{
    uint32_t marker;
    scalar_t r;
    is.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    is.read(reinterpret_cast<char *>(&r), sizeof(r));
    if (marker != make_marker("DROP"))
        throw std::runtime_error("Arch mismatch in DropoutLayer binary load");
    ratio_ = r;
}

nlohmann::json DropoutLayer::get_config() const
{
    return { { "type", "Dropout" }, { "ratio", ratio_ } };
}

Shape3D DropoutLayer::get_output_shape(const Shape3D &input_shape) const
{
    return input_shape;
}