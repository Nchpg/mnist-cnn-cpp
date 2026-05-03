#ifndef CONV_LAYER_HPP
#define CONV_LAYER_HPP

#include "layer.hpp"
#include "matrix.hpp"
#include "tensor_batch.hpp"
#include <vector>
#include <random>
class ConvLayer : public Layer {
private:
    size_t kernel_size_;
    std::vector<scalar_t> biases_;  
    std::vector<scalar_t> biases_grad_;

    const Matrix* input_ptr_ = nullptr; 
    TensorBatch output_;
    TensorBatch grad_input_;

    Matrix filters_matrix_; 

    // Adam states
    Matrix m_filters_, v_filters_;
    Matrix m_biases_, v_biases_;
    Matrix filters_grad_matrix_;

    Matrix im2col_buffer_;  
    Matrix gemm_out_;       

    void im2col(const Matrix& input, Matrix& out_col);
    void col2im(const Matrix& grad_col, TensorBatch& grad_input);

public:
    ConvLayer(size_t input_rows, size_t input_cols, size_t input_channels, size_t kernel_size, size_t filter_count, std::mt19937& gen, scalar_t weight_scale = 0.01f);
    ~ConvLayer() override = default;

    const Matrix& forward(const Matrix& input) override;
    const Matrix& backward(const Matrix& gradient) override;

    void update_weights(scalar_t learning_rate) override;
    void update_weights_adam(scalar_t learning_rate, scalar_t beta1, scalar_t beta2, scalar_t epsilon, scalar_t m_corr, scalar_t v_corr) override;
    void clear_gradients() override;

    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

    Shape3D get_output_shape(const Shape3D& input_shape) const override {
        return {filters_matrix_.rows(), input_shape.height - kernel_size_ + 1, input_shape.width - kernel_size_ + 1};
    }
};

#endif 
