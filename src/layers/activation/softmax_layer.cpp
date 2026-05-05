#include "layers/activation/softmax_layer.hpp"

#include "layers/activation/activation.hpp"

SoftmaxLayer::SoftmaxLayer()
{}

SoftmaxLayer::~SoftmaxLayer()
{}

const Tensor &SoftmaxLayer::forward(const Tensor &input)
{
    Activation::softmax(input, output_);
    return output_;
}

const Tensor &SoftmaxLayer::backward(const Tensor &gradient)
{
    Activation::softmax_backward(output_, gradient, grad_input_);
    return grad_input_;
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