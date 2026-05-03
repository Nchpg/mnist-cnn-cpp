#ifndef CONV_LAYER_HPP
#define CONV_LAYER_HPP

#include "layer.hpp"
#include "matrix.hpp"
#include "tensor_batch.hpp"
#include <vector>
#include <random>

class ConvLayer : public Layer {
private:
    std::vector<TensorBatch> filters_; 
    std::vector<TensorBatch> filters_grad_;
    std::vector<scalar_t> biases_;  
    std::vector<scalar_t> biases_grad_;
    
    const Matrix* input_ptr_ = nullptr; 
    TensorBatch output_;
    TensorBatch grad_input_;

    Matrix filters_matrix_; 
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
    void clear_gradients() override;

    void save(std::ostream& os) const override;
    void load(std::istream& is) override;

    Shape3D get_output_shape(const Shape3D& input_shape) const override {
        size_t k_s = filters_[0].height();
        return {filters_.size(), input_shape.height - k_s + 1, input_shape.width - k_s + 1};
    }
};

#endif 
