#include "conv_layer.hpp"
#include <stdexcept>
#include <omp.h>

ConvLayer::ConvLayer(size_t input_rows, size_t input_cols, size_t input_channels, size_t kernel_size, size_t filter_count, std::mt19937& gen, scalar_t /*weight_scale*/) 
    : kernel_size_(kernel_size) {
    size_t out_rows = input_rows - kernel_size + 1;
    size_t out_cols = input_cols - kernel_size + 1;

    output_     = TensorBatch(filter_count, out_rows, out_cols, 1);
    grad_input_ = TensorBatch(input_channels, input_rows, input_cols, 1);

    size_t weights_per_filter = input_channels * kernel_size * kernel_size;
    filters_matrix_ = Matrix(filter_count, weights_per_filter);

    scalar_t scale = std::sqrt(2.0f / static_cast<scalar_t>(weights_per_filter));
    filters_matrix_.random_uniform(scale, gen);
    
    biases_.resize(filter_count, 0.0f);
    biases_grad_.resize(filter_count, 0.0f);
}


static inline size_t get_im2col_col(size_t b, size_t y, size_t x, size_t out_h, size_t out_w) {
    return (b * out_h * out_w) + (y * out_w) + x;
}

static inline size_t get_kernel_row(size_t c, size_t ky, size_t kx, size_t k_s) {
    return (c * k_s * k_s) + (ky * k_s) + kx;
}

void ConvLayer::im2col(const Matrix& input, Matrix& out_col) {
    const size_t b_size = input.cols();
    const size_t in_c   = grad_input_.channels();
    const size_t in_h   = grad_input_.height();
    const size_t in_w   = grad_input_.width();
    const size_t k_s    = kernel_size_;
    const size_t out_h  = output_.height();
    const size_t out_w  = output_.width();

    #pragma omp parallel for collapse(2) if(b_size * in_c > 4)
    for (size_t b = 0; b < b_size; ++b) {
        for (size_t c = 0; c < in_c; ++c) {
            for (size_t ky = 0; ky < k_s; ++ky) {
                for (size_t kx = 0; kx < k_s; ++kx) {
                    size_t row = get_kernel_row(c, ky, kx, k_s);
                    for (size_t y = 0; y < out_h; ++y) {
                        for (size_t x = 0; x < out_w; ++x) {
                            size_t col = get_im2col_col(b, y, x, out_h, out_w);
                            out_col(row, col) = TensorBatch::read_mat(input, b, c, y + ky, x + kx, in_h, in_w);
                        }
                    }
                }
            }
        }
    }
}

void ConvLayer::col2im(const Matrix& grad_col, TensorBatch& grad_input) {
    const size_t b_size = grad_input.cols();
    const size_t in_c   = grad_input.channels();
    const size_t k_s    = kernel_size_;
    const size_t out_h  = output_.height();
    const size_t out_w  = output_.width();

    grad_input.fill(0.0f);
    #pragma omp parallel for collapse(2) if(b_size * in_c > 4)
    for (size_t b = 0; b < b_size; ++b) {
        for (size_t c = 0; c < in_c; ++c) {
            for (size_t ky = 0; ky < k_s; ++ky) {
                for (size_t kx = 0; kx < k_s; ++kx) {
                    size_t row = get_kernel_row(c, ky, kx, k_s);
                    for (size_t y = 0; y < out_h; ++y) {
                        for (size_t x = 0; x < out_w; ++x) {
                            size_t col = get_im2col_col(b, y, x, out_h, out_w);
                            grad_input(b, c, y + ky, x + kx) += grad_col(row, col);
                        }
                    }
                }
            }
        }
    }
}

const Matrix& ConvLayer::forward(const Matrix& input) {
    size_t b_size = input.cols();
    input_ptr_ = &input;
    
    if (output_.batch_size() != b_size || im2col_buffer_.rows() == 0) {
        output_.reshape4D(output_.channels(), output_.height(), output_.width(), b_size);
        grad_input_.reshape4D(grad_input_.channels(), grad_input_.height(), grad_input_.width(), b_size);
        
        size_t weights_per_filter = grad_input_.channels() * kernel_size_ * kernel_size_;
        size_t out_pixels_total = output_.height() * output_.width() * b_size;
        im2col_buffer_.reshape(weights_per_filter, out_pixels_total);
        gemm_out_.reshape(filters_matrix_.rows(), out_pixels_total);
    }
    
    im2col(*input_ptr_, im2col_buffer_);
    Matrix::multiply(filters_matrix_, im2col_buffer_, gemm_out_);

    const size_t f_count = filters_matrix_.rows();
    const size_t out_h = output_.height();
    const size_t out_w = output_.width();

    #pragma omp parallel for collapse(2) if(b_size * f_count > 4)
    for (size_t b = 0; b < b_size; ++b) {
        for (size_t f = 0; f < f_count; ++f) {
            scalar_t b_val = biases_[f];
            for (size_t y = 0; y < out_h; ++y) {
                for (size_t x = 0; x < out_w; ++x) {
                    size_t col = get_im2col_col(b, y, x, out_h, out_w);
                    output_(b, f, y, x) = gemm_out_(f, col) + b_val;
                }
            }
        }
    }

    return output_; 
}

const Matrix& ConvLayer::backward(const Matrix& gradient) {
    const size_t b_size = input_ptr_->cols();
    const size_t f_count = filters_matrix_.rows();
    const size_t out_h = output_.height();
    const size_t out_w = output_.width();
    const size_t weights_per_filter = filters_matrix_.cols();

    if (filters_grad_matrix_.rows() != f_count || filters_grad_matrix_.cols() != weights_per_filter) {
        filters_grad_matrix_.reshape(f_count, weights_per_filter);
    }

    #pragma omp parallel for collapse(2) if(f_count * b_size > 4)
    for (size_t b = 0; b < b_size; ++b) {
        for (size_t f = 0; f < f_count; ++f) {
            for (size_t y = 0; y < out_h; ++y) {
                for (size_t x = 0; x < out_w; ++x) {
                    size_t col = get_im2col_col(b, y, x, out_h, out_w);
                    gemm_out_(f, col) = TensorBatch::read_mat(gradient, b, f, y, x, out_h, out_w);
                }
            }
        }
    }

    for (size_t f = 0; f < f_count; ++f) {
        scalar_t sum = 0.0f;
        for (size_t col = 0; col < gemm_out_.cols(); ++col) {
            sum += gemm_out_(f, col);
        }
        biases_grad_[f] = sum;
    }

    Matrix::multiply_transB(gemm_out_, im2col_buffer_, filters_grad_matrix_);
    
    Matrix grad_col_buffer(weights_per_filter, gemm_out_.cols());
    Matrix::multiply_transA(filters_matrix_, gemm_out_, grad_col_buffer);
    col2im(grad_col_buffer, grad_input_);

    return grad_input_;
}

void ConvLayer::update_weights(scalar_t learning_rate) {
    const size_t f_count = filters_matrix_.rows();
    filters_matrix_.subtract_scaled(filters_grad_matrix_, learning_rate);
    for (size_t f = 0; f < f_count; ++f) {
        biases_[f] -= biases_grad_[f] * learning_rate;
    }
}

void ConvLayer::update_weights_adam(scalar_t learning_rate, scalar_t beta1, scalar_t beta2, scalar_t epsilon, scalar_t m_corr, scalar_t v_corr) {
    const size_t f_count = filters_matrix_.rows();
    const size_t weights_per_filter = filters_matrix_.cols();
    const size_t total_weights = f_count * weights_per_filter;

    if (m_filters_.rows() == 0) {
        m_filters_.reshape(f_count, weights_per_filter);
        v_filters_.reshape(f_count, weights_per_filter);
        m_biases_.reshape(f_count, 1);
        v_biases_.reshape(f_count, 1);
        m_filters_.fill(0.0f);
        v_filters_.fill(0.0f);
        m_biases_.fill(0.0f);
        v_biases_.fill(0.0f);
    }

    scalar_t* w_ptr = filters_matrix_.data();
    scalar_t* wg_ptr = filters_grad_matrix_.data();
    scalar_t* mw_ptr = m_filters_.data();
    scalar_t* vw_ptr = v_filters_.data();

    #pragma omp parallel for if(total_weights > 1000)
    for (size_t i = 0; i < total_weights; ++i) {
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

    for (size_t i = 0; i < f_count; ++i) {
        mb_ptr[i] = beta1 * mb_ptr[i] + (1.0f - beta1) * bg_ptr[i];
        vb_ptr[i] = beta2 * vb_ptr[i] + (1.0f - beta2) * bg_ptr[i] * bg_ptr[i];
        scalar_t m_hat = mb_ptr[i] * m_corr;
        scalar_t v_hat = vb_ptr[i] * v_corr;
        b_ptr[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}

void ConvLayer::clear_gradients() {
    filters_grad_matrix_.fill(0.0f);
    std::fill(biases_grad_.begin(), biases_grad_.end(), 0.0f);
}

void ConvLayer::save(std::ostream& os) const {
    const size_t f_count = filters_matrix_.rows();
    const size_t weights_per_filter = filters_matrix_.cols();
    os << "CONV " << f_count << " " << kernel_size_ << "\n";
    for (size_t f = 0; f < f_count; ++f) {
        os << biases_[f] << "\n";
        for (size_t i = 0; i < weights_per_filter; ++i) {
            os << filters_matrix_(f, i) << (i == weights_per_filter - 1 ? "" : " ");
        }
        os << "\n";
    }
}

void ConvLayer::load(std::istream& is) {
    const size_t expected_f_count = filters_matrix_.rows();
    const size_t expected_weights_per_filter = filters_matrix_.cols();
    std::string type;
    size_t f_count, k_size;
    is >> type >> f_count >> k_size;
    if (type != "CONV" || f_count != expected_f_count || k_size != kernel_size_) {
        throw std::runtime_error("Invalid ConvLayer data in model file");
    }
    for (size_t f = 0; f < f_count; ++f) {
        is >> biases_[f];
        for (size_t i = 0; i < expected_weights_per_filter; ++i) {
            is >> filters_matrix_(f, i);
        }
    }
}
