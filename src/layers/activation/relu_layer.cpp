#include "layers/activation/relu_layer.hpp"

#include "layers/activation/activation.hpp"

ReluLayer::ReluLayer()
{}

const Tensor &ReluLayer::forward(const Tensor &input)
{
    input_ptr_ = &input;
    Activation::relu(input, output_);
    return output_;
}

const Tensor &ReluLayer::backward(const Tensor &gradient)
{
    Activation::relu_backward(*input_ptr_, gradient, grad_input_);
    return grad_input_;
}

void ReluLayer::save(std::ostream &os) const
{
    uint32_t marker = make_marker(LAYER_MARKER);
    os.write(reinterpret_cast<const char *>(&marker), sizeof(marker));
}

void ReluLayer::load(std::istream &is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    if (marker != make_marker(LAYER_MARKER))
        throw std::runtime_error("Architecture mismatch in ReluLayer load");
}

nlohmann::json ReluLayer::get_config() const
{
    return { { "type", "ReLU" } };
}

Shape3D ReluLayer::get_output_shape(const Shape3D &input_shape) const
{
    return input_shape;
}