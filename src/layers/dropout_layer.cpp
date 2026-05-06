#include "layers/dropout_layer.hpp"

#include <cassert>

DropoutLayer::DropoutLayer(scalar_t ratio)
    : ratio_(ratio)
{}

const Tensor &DropoutLayer::forward(const Tensor &input,
                                    std::unique_ptr<LayerContext> &ctx,
                                    bool is_training) const
{
    if (!ctx)
    {
        ctx = std::make_unique<DropoutContext>();
    }
    auto *dropout_ctx = static_cast<DropoutContext *>(ctx.get());

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
    const scalar_t *in_ptr = input.data_ptr();
    scalar_t *mask_ptr = dropout_ctx->mask.data_ptr();
    scalar_t *out_ptr = dropout_ctx->output.data_ptr();

    static thread_local std::mt19937 gen([] {
        std::random_device rd;
        return rd();
    }());
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

    return dropout_ctx->output;
}

const Tensor &DropoutLayer::backward(const Tensor &gradient,
                                     std::unique_ptr<LayerContext> &ctx,
                                     bool is_training)
{
    assert(is_training
           && "Backward doit uniquement etre appele durant l'entrainement !");
    (void)is_training;
    auto *dropout_ctx = static_cast<DropoutContext *>(ctx.get());

    if (dropout_ctx->grad_input.shape() != gradient.shape())
    {
        dropout_ctx->grad_input.reshape(gradient.shape());
    }

    size_t n = gradient.size();
    const scalar_t *grad_ptr = gradient.data_ptr();
    const scalar_t *mask_ptr = dropout_ctx->mask.data_ptr();
    scalar_t *in_grad_ptr = dropout_ctx->grad_input.data_ptr();

#pragma omp parallel for if (n > 10000)
    for (size_t i = 0; i < n; ++i)
    {
        in_grad_ptr[i] = grad_ptr[i] * mask_ptr[i];
    }

    return dropout_ctx->grad_input;
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