#include "layers/dropout_layer.hpp"

#include <cassert>
#include <stdexcept>

DropoutLayer::DropoutLayer(scalar_t ratio)
    : ratio_(ratio)
{}

const Tensor& DropoutLayer::forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool is_training) const
{
    if (input.rank() < 2) {
        throw std::invalid_argument("Runtime error: DropoutLayer requires at least a 2D tensor (Batch, Features).");
    }
    if (!ctx)
    {
        ctx = std::make_unique<DropoutContext>();
    }
    auto* dropout_ctx = static_cast<DropoutContext*>(ctx.get());

    if (!is_training)
    {
        return input;
    }

    if (dropout_ctx->output.shape() != input.shape())
    {
        dropout_ctx->mask.reshape(input.shape());
        dropout_ctx->output.reshape(input.shape());
    }

    scalar_t scale = 1.0f / (1.0f - ratio_);
    size_t n = input.size();
    const scalar_t* in_ptr = input.data_ptr();
    scalar_t* mask_ptr = dropout_ctx->mask.data_ptr();
    scalar_t* out_ptr = dropout_ctx->output.data_ptr();

#pragma omp parallel if (n > 10000)
    {
        thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<scalar_t> dist(0.0f, 1.0f);
#pragma omp for
        for (size_t i = 0; i < n; ++i)
        {
            mask_ptr[i] = (dist(gen) > ratio_) ? scale : 0.0f;
            out_ptr[i] = in_ptr[i] * mask_ptr[i];
        }
    }

    return dropout_ctx->output;
}

const Tensor& DropoutLayer::backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx, bool is_training)
{
    assert(is_training && "Backward doit uniquement etre appele durant l'entrainement !");
    (void)is_training;
    auto* dropout_ctx = static_cast<DropoutContext*>(ctx.get());

    if (dropout_ctx->grad_input.shape() != gradient.shape())
    {
        dropout_ctx->grad_input.reshape(gradient.shape());
    }

    size_t n = gradient.size();
    const scalar_t* grad_ptr = gradient.data_ptr();
    const scalar_t* mask_ptr = dropout_ctx->mask.data_ptr();
    scalar_t* in_grad_ptr = dropout_ctx->grad_input.data_ptr();

#pragma omp parallel for if (n > 10000)
    for (size_t i = 0; i < n; ++i)
    {
        in_grad_ptr[i] = grad_ptr[i] * mask_ptr[i];
    }

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