#include "layers/dense_layer.hpp"

#include <fstream>
#include <stdexcept>

#include "layers/activation/activation.hpp"

DenseLayer::DenseLayer(size_t input_size, size_t output_size, std::mt19937 &gen)
    : input_size_(input_size)
    , output_size_(output_size)
    , weights_(output_size, input_size)
    , biases_(output_size, 1)
    , activations_(output_size, 1)
    , weights_grad_(output_size, input_size)
    , biases_grad_(output_size, 1)
    , weights_t_(input_size, output_size)
    , input_t_(1, input_size)
    , grad_input_(input_size, 1)
{
    scalar_t std_dev = std::sqrt(2.0f / static_cast<scalar_t>(input_size));
    weights_.random_normal(std_dev, gen);
    biases_.fill(0.0f);
    clear_gradients();
}

const Matrix &DenseLayer::forward(const Matrix &input)
{
    size_t batch_size = input.cols();
    input_ptr_ = &input;

    if (activations_.cols() != batch_size)
    {
        activations_.reshape(output_size_, batch_size);
        grad_input_.reshape(input_size_, batch_size);
    }

    Matrix::multiply(weights_, input, activations_);
    activations_.add_broadcast(biases_);
    return activations_;
}

const Matrix &DenseLayer::backward(const Matrix &gradient)
{
    Matrix::multiply_transA(weights_, gradient, grad_input_);
    Matrix::multiply_transB(gradient, *input_ptr_, weights_grad_);
    gradient.sum_columns(biases_grad_);
    return grad_input_;
}

std::vector<Parameter> DenseLayer::get_parameters()
{
    return { { &weights_, &weights_grad_ }, { &biases_, &biases_grad_ } };
}

void DenseLayer::clear_gradients()
{
    weights_grad_.fill(0.0f);
    biases_grad_.fill(0.0f);
}

void DenseLayer::save(std::ostream &os) const
{
    os.write(reinterpret_cast<const char *>(&input_size_), sizeof(input_size_));
    os.write(reinterpret_cast<const char *>(&output_size_),
             sizeof(output_size_));
    weights_.save(os);
    biases_.save(os);
}

void DenseLayer::load(std::istream &is)
{
    size_t in_size, out_size;
    is.read(reinterpret_cast<char *>(&in_size), sizeof(in_size));
    is.read(reinterpret_cast<char *>(&out_size), sizeof(out_size));
    if (in_size != input_size_ || out_size != output_size_)
    {
        throw std::runtime_error("Arch mismatch in DenseLayer load");
    }
    weights_.load(is);
    biases_.load(is);
}

const Matrix &DenseLayer::activations() const
{
    return activations_;
}

nlohmann::json DenseLayer::get_config() const
{
    return { { "type", "Dense" }, { "units", output_size_ } };
}

Shape3D DenseLayer::get_output_shape(const Shape3D &input_shape) const
{
    (void)input_shape;
    return { output_size_, 1, 1 };
}
