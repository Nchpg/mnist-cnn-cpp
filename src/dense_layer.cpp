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

void DenseLayer::update_weights(scalar_t learning_rate) {
    weights_.subtract_scaled(weights_grad_, learning_rate);
    biases_.subtract_scaled(biases_grad_, learning_rate);
}

void DenseLayer::update_weights_adam(scalar_t learning_rate, scalar_t beta1, scalar_t beta2, scalar_t epsilon, scalar_t m_corr, scalar_t v_corr) {
    if (m_weights_.rows() == 0) {
        m_weights_.reshape(output_size_, input_size_);
        v_weights_.reshape(output_size_, input_size_);
        m_biases_.reshape(output_size_, 1);
        v_biases_.reshape(output_size_, 1);
        m_weights_.fill(0.0f);
        v_weights_.fill(0.0f);
        m_biases_.fill(0.0f);
        v_biases_.fill(0.0f);
    }

    size_t w_total = output_size_ * input_size_;
    scalar_t* w_ptr = weights_.data();
    scalar_t* wg_ptr = weights_grad_.data();
    scalar_t* mw_ptr = m_weights_.data();
    scalar_t* vw_ptr = v_weights_.data();

    #pragma omp parallel for if(w_total > 1000)
    for (size_t i = 0; i < w_total; ++i) {
        mw_ptr[i] = beta1 * mw_ptr[i] + (1.0f - beta1) * wg_ptr[i];
        vw_ptr[i] = beta2 * vw_ptr[i] + (1.0f - beta2) * wg_ptr[i] * wg_ptr[i];
        scalar_t m_hat = mw_ptr[i] * m_corr;
        scalar_t v_hat = vw_ptr[i] * v_corr;
        w_ptr[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }

    scalar_t* b_ptr = biases_.data();
    scalar_t* bg_ptr = biases_grad_.data();
    scalar_t* mb_ptr = m_biases_.data();
    scalar_t* vb_ptr = v_biases_.data();

    for (size_t i = 0; i < output_size_; ++i) {
        mb_ptr[i] = beta1 * mb_ptr[i] + (1.0f - beta1) * bg_ptr[i];
        vb_ptr[i] = beta2 * vb_ptr[i] + (1.0f - beta2) * bg_ptr[i] * bg_ptr[i];
        scalar_t m_hat = mb_ptr[i] * m_corr;
        scalar_t v_hat = vb_ptr[i] * v_corr;
        b_ptr[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}

void DenseLayer::clear_gradients() {
    weights_grad_.fill(0.0f);
    biases_grad_.fill(0.0f);
}

void DenseLayer::save(std::ostream& os) const {
    os << "DENSE " << input_size_ << " " << output_size_ << "\n";
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
    for (size_t output = 0; output < output_size_; output++) {
        is >> biases_(output, 0);
        for (size_t input_index = 0; input_index < input_size_; input_index++) {
            is >> weights_(output, input_index);
        }
    }
}
