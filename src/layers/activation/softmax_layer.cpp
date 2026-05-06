#include "layers/activation/softmax_layer.hpp"

#include "layers/activation/activation.hpp"

SoftmaxLayer::SoftmaxLayer()
{}

SoftmaxLayer::~SoftmaxLayer()
{}

const Tensor &SoftmaxLayer::forward(const Tensor &input,
                                    std::unique_ptr<LayerContext> &ctx,
                                    bool is_training) const
{
    if (!ctx)
    {
        ctx = std::make_unique<SoftmaxContext>();
    }
    auto *softmax_ctx = static_cast<SoftmaxContext *>(ctx.get());

    softmax_ctx->output.reshape(input.shape());
    Activation::softmax(input, softmax_ctx->output);

    (void)is_training;
    return softmax_ctx->output;
}

const Tensor &SoftmaxLayer::backward(const Tensor &gradient,
                                     std::unique_ptr<LayerContext> &ctx,
                                     bool is_training)
{
    auto *softmax_ctx = static_cast<SoftmaxContext *>(ctx.get());
    softmax_ctx->grad_input.reshape(gradient.shape());
    Activation::softmax_backward(softmax_ctx->output, gradient,
                                 softmax_ctx->grad_input);
    (void)is_training;
    return softmax_ctx->grad_input;
}

nlohmann::json SoftmaxLayer::get_config() const
{
    return { { "type", "Softmax" } };
}

Shape3D SoftmaxLayer::get_output_shape(const Shape3D &input_shape) const
{
    return input_shape;
}

void SoftmaxLayer::save(std::ostream &os) const
{
    uint32_t marker = make_marker(LAYER_MARKER);
    os.write(reinterpret_cast<const char *>(&marker), sizeof(marker));
}

void SoftmaxLayer::load(std::istream &is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    if (marker != make_marker(LAYER_MARKER))
        throw std::runtime_error("Architecture mismatch in SoftmaxLayer load");
}