#include "layers/activation/sigmoid_layer.hpp"

#include "layers/activation/activation.hpp"

SigmoidLayer::SigmoidLayer()
{}

const Tensor &SigmoidLayer::forward(const Tensor &input)
{
    Activation::sigmoid(input, output_);
    return output_;
}

const Tensor &SigmoidLayer::backward(const Tensor &gradient)
{
    Activation::sigmoid_backward(output_, gradient, grad_input_);
    return grad_input_;
}

void SigmoidLayer::save(std::ostream &os) const
{
    uint32_t marker = make_marker(LAYER_MARKER);
    os.write(reinterpret_cast<const char *>(&marker), sizeof(marker));
}

void SigmoidLayer::load(std::istream &is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    if (marker != make_marker(LAYER_MARKER))
        throw std::runtime_error("Architecture mismatch in SigmoidLayer load");
}

nlohmann::json SigmoidLayer::get_config() const
{
    return { { "type", "Sigmoid" } };
}

Shape3D SigmoidLayer::get_output_shape(const Shape3D &input_shape) const
{
    return input_shape;
}