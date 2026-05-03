#include "pooling_layer.hpp"
#include <stdexcept>
#include <omp.h>
#include <limits>

PoolingLayer::PoolingLayer(size_t input_rows, size_t input_cols, size_t filter_count, size_t pool_size, size_t stride)
    : pool_size_(pool_size), stride_(stride) {
    
    size_t out_rows = (input_rows - pool_size) / stride + 1;
    size_t out_cols = (input_cols - pool_size) / stride + 1;

    output_     = TensorBatch(filter_count, out_rows, out_cols, 1);
    grad_input_ = TensorBatch(filter_count, input_rows, input_cols, 1);

    argmax_indices_.resize(1 * filter_count * out_rows * out_cols, 0);
}

const Matrix& PoolingLayer::forward(const Matrix& input) {
    size_t batch_size = input.cols();
    input_ptr_ = &input;

    if (output_.batch_size() != batch_size) {
        output_.reshape4D(output_.channels(), output_.height(), output_.width(), batch_size);
        grad_input_.reshape4D(grad_input_.channels(), grad_input_.height(), grad_input_.width(), batch_size);
        argmax_indices_.resize(output_.rows() * batch_size, 0);
    }

    const size_t channels = output_.channels();
    const size_t out_h    = output_.height();
    const size_t out_w    = output_.width();
    const size_t in_h     = grad_input_.height();
    const size_t in_w     = grad_input_.width();

    #pragma omp parallel for collapse(2) if(channels * out_h > 4)
    for (size_t c = 0; c < channels; ++c) {
        for (size_t i = 0; i < out_h; ++i) {
            for (size_t j = 0; j < out_w; ++j) {
                
                size_t start_r = i * stride_;
                size_t start_c = j * stride_;
                size_t out_row = (c * out_h + i) * out_w + j;

                for (size_t b = 0; b < batch_size; ++b) {
                    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
                    size_t max_idx = 0;

                    for (size_t ki = 0; ki < pool_size_; ++ki) {
                        for (size_t kj = 0; kj < pool_size_; ++kj) {
                            size_t in_r = start_r + ki;
                            size_t in_c = start_c + kj;
                            
                            size_t in_row = (c * in_h + in_r) * in_w + in_c;
                            scalar_t val = input(in_row, b);

                            if (val > max_val) {
                                max_val = val;
                                max_idx = in_row;
                            }
                        }
                    }
                    
                    output_(b, c, i, j) = max_val;
                    argmax_indices_[out_row * batch_size + b] = max_idx;
                }
            }
        }
    }

    return output_;
}

const Matrix& PoolingLayer::backward(const Matrix& gradient) {
    grad_input_.fill(0.0f);
    size_t batch_size = output_.batch_size();
    size_t num_out_rows = output_.rows();

    #pragma omp parallel for if(num_out_rows > 100)
    for (size_t r = 0; r < num_out_rows; ++r) {
        for (size_t b = 0; b < batch_size; ++b) {
            size_t max_in_row = argmax_indices_[r * batch_size + b];
            scalar_t grad_val = gradient(r, b);
            grad_input_(max_in_row, b) += grad_val;
        }
    }

    return grad_input_;
}
