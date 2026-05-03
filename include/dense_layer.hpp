#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "layer.hpp"
#include "matrix.hpp"
#include <cstddef>
#include <string>
#include <stdexcept>

#include <random>

class DenseLayer : public Layer {
private:
    size_t input_size_;
    size_t output_size_;
    Matrix weights_;
    Matrix biases_;
    Matrix activations_;
    const Matrix* input_ptr_ = nullptr; 
    Matrix weights_grad_;
    Matrix biases_grad_;
    
    // Adam states
    Matrix m_weights_, v_weights_;
    Matrix m_biases_, v_biases_;

    Matrix weights_t_;    
    Matrix input_t_;      
    Matrix grad_input_;   

public:
    DenseLayer(size_t input_size, size_t output_size, std::mt19937& gen);
    ~DenseLayer() override = default;
    
    const Matrix& forward(const Matrix& input) override;
    const Matrix& backward(const Matrix& gradient) override;
    
    void update_weights(scalar_t learning_rate) override;
    void update_weights_adam(scalar_t learning_rate, scalar_t beta1, scalar_t beta2, scalar_t epsilon, scalar_t m_corr, scalar_t v_corr) override;
    void clear_gradients() override;
    
    const Matrix& activations() const { return activations_; }
    
    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

    Shape3D get_output_shape(const Shape3D& /*input_shape*/) const override {
        return {output_size_, 1, 1};
    }
};

#endif 