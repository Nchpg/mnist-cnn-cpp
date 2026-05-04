#include "dense_layer.hpp"
#include "activation.hpp"
#include <fstream>
#include <stdexcept>

DenseLayer::DenseLayer(size_t input_size, size_t output_size, std::mt19937& gen)
    : input_size_(input_size), output_size_(output_size),
      weights_(output_size, input_size),
      biases_(output_size, 1),
      activations_(output_size, 1),
      weights_grad_(output_size, input_size),
      biases_grad_(output_size, 1),
      weights_t_(input_size, output_size),
      input_t_(1, input_size),
      grad_input_(input_size, 1)
{
    scalar_t scale = std::sqrt(2.0f / static_cast<scalar_t>(input_size));
    weights_.random_uniform(scale, gen);
    biases_.fill(0.0f);
    clear_gradients();
}

const Matrix& DenseLayer::forward(const Matrix& input) {
    size_t batch_size = input.cols();
    input_ptr_ = &input;
    
    if (activations_.cols() != batch_size) {
        activations_.reshape(output_size_, batch_size);
        grad_input_.reshape(input_size_, batch_size);
    }

    Matrix::multiply(weights_, input, activations_);
    activations_.add_broadcast(biases_);
    return activations_;
}

const Matrix& DenseLayer::backward(const Matrix& gradient) {
    Matrix::multiply_transA(weights_, gradient, grad_input_);
    Matrix::multiply_transB(gradient, *input_ptr_, weights_grad_);
    gradient.sum_columns(biases_grad_);
    return grad_input_;
}

std::vector<Parameter> DenseLayer::get_parameters() {
    return {
        {&weights_, &weights_grad_},
        {&biases_, &biases_grad_}
    };
}

void DenseLayer::clear_gradients() {
    weights_grad_.fill(0.0f);
    biases_grad_.fill(0.0f);
}

void DenseLayer::save(std::ostream& os) const {
    os << LAYER_NAME << " " << input_size_ << " " << output_size_ << "\n";
    for (size_t output = 0; output < output_size_; output++) {
        os << biases_(output, 0);
        for (size_t input_index = 0; input_index < input_size_; input_index++) {
            os << " " << weights_(output, input_index);
        }
        os << "\n";
    }
}

void DenseLayer::load(std::istream& is) {
    std::string type;
    size_t in_size = 0, out_size = 0;
    is >> type >> in_size >> out_size;
    if (type != LAYER_NAME || in_size != input_size_ || out_size != output_size_) {
        throw std::runtime_error("Invalid DenseLayer data: expected '" + std::string(LAYER_NAME) + "'");
    }
    for (size_t output = 0; output < output_size_; output++) {
        is >> biases_(output, 0);
        for (size_t input_index = 0; input_index < input_size_; input_index++) {
            is >> weights_(output, input_index);
        }
    }
}
