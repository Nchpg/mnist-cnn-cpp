#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "layer.hpp"
#include "matrix.hpp"
#include <cstddef>
#include <string>
#include <stdexcept>

#include <random>

class DenseLayer : public Layer {
public:
    static constexpr const char* LAYER_NAME = "DENSE";

private:
    size_t input_size_;
    size_t output_size_;
    Matrix weights_;
    Matrix biases_;
    Matrix activations_;
    const Matrix* input_ptr_ = nullptr; 
    Matrix weights_grad_;
    Matrix biases_grad_;
    
    Matrix weights_t_;    
    Matrix input_t_;      
    Matrix grad_input_;   

public:
    DenseLayer(size_t input_size, size_t output_size, std::mt19937& gen);
    ~DenseLayer() override = default;
    
    const Matrix& forward(const Matrix& input) override;
    const Matrix& backward(const Matrix& gradient) override;
    
    void clear_gradients() override;
    
    std::vector<Parameter> get_parameters() override;
    
    const Matrix& activations() const { return activations_; }
    
    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

    Shape3D get_output_shape(const Shape3D& /*input_shape*/) const override {
        return {output_size_, 1, 1};
    }
};

#endif 