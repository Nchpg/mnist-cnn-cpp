#include "matrix.hpp"
#include <algorithm>
#include <stdexcept>
#include <omp.h>

Matrix::Matrix(size_t rows, size_t cols, scalar_t initial_val) 
    : rows_(rows), cols_(cols), data_(rows * cols, initial_val) {}

scalar_t& Matrix::at(size_t r, size_t r_idx) {
    return (*this)(r, r_idx);
}

const scalar_t& Matrix::at(size_t r, size_t c) const {
    return (*this)(r, c);
}

void Matrix::fill(scalar_t value) noexcept {
    std::fill(data_.begin(), data_.end(), value);
}

void Matrix::random_uniform(scalar_t scale, std::mt19937& gen) {
    std::uniform_real_distribution<scalar_t> dist(-scale, scale);
    for (scalar_t& val : data_) val = dist(gen);
}

void Matrix::reshape(size_t rows, size_t cols) {
    rows_ = rows;
    cols_ = cols;
    data_.resize(rows * cols, 0.0f);
}

void Matrix::multiply(const Matrix& A, const Matrix& B, Matrix& C) {
    if (A.cols_ != B.rows_ || C.rows_ != A.rows_ || C.cols_ != B.cols_) {
        throw std::invalid_argument("Dimension mismatch in multiply");
    }
    C.fill(0.0f);
    const size_t M = A.rows_, K = A.cols_, N = B.cols_;
    
    #pragma omp parallel for if(M * K * N > 50000)
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            scalar_t a_val = A(i, k);
            if (a_val == 0.0f) continue;
            for (size_t j = 0; j < N; ++j) {
                C(i, j) += a_val * B(k, j);
            }
        }
    }
}

void Matrix::multiply_transA(const Matrix& A, const Matrix& B, Matrix& C) {
    if (A.rows_ != B.rows_ || C.rows_ != A.cols_ || C.cols_ != B.cols_) {
        throw std::invalid_argument("Dimension mismatch in multiply_transA");
    }
    C.fill(0.0f);
    const size_t K = A.rows_, M = A.cols_, N = B.cols_;

    #pragma omp parallel for if(M * K * N > 50000)
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            scalar_t a_val = A(k, i); 
            if (a_val == 0.0f) continue;
            for (size_t j = 0; j < N; ++j) {
                C(i, j) += a_val * B(k, j);
            }
        }
    }
}

void Matrix::multiply_transB(const Matrix& A, const Matrix& B, Matrix& C) {
    if (A.cols_ != B.cols_ || C.rows_ != A.rows_ || C.cols_ != B.rows_) {
        throw std::invalid_argument("Dimension mismatch in multiply_transB");
    }
    C.fill(0.0f);
    const size_t M = A.rows_, K = A.cols_, N = B.rows_;

    #pragma omp parallel for if(M * K * N > 50000)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            scalar_t sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A(i, k) * B(j, k); 
            }
            C(i, j) = sum;
        }
    }
}

void Matrix::transpose(const Matrix& m, Matrix& out) {
    if (out.rows_ != m.cols_ || out.cols_ != m.rows_) {
        throw std::invalid_argument("Dimension mismatch in transpose");
    }
    for (size_t i = 0; i < m.rows_; ++i) {
        for (size_t j = 0; j < m.cols_; ++j) {
            out(j, i) = m(i, j);
        }
    }
}

Matrix& Matrix::operator+=(const Matrix& other) {
    for (size_t i = 0; i < data_.size(); ++i) data_[i] += other.data_[i];
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    for (size_t i = 0; i < data_.size(); ++i) data_[i] -= other.data_[i];
    return *this;
}

void Matrix::add_scaled(const Matrix& other, scalar_t scale) {
    for (size_t i = 0; i < data_.size(); ++i) data_[i] += other.data_[i] * scale;
}

void Matrix::add_outer_product(const Matrix& left, const Matrix& right) {
    bool use_omp = (rows_ * cols_) > 10000;
    #pragma omp parallel for if(use_omp)
    for (size_t i = 0; i < rows_; ++i) {
        scalar_t left_val = left(i, 0);
        for (size_t j = 0; j < cols_; ++j) {
            (*this)(i, j) += left_val * right(j, 0);
        }
    }
}

void Matrix::subtract_scaled(const Matrix& grad, scalar_t scale) {
    for (size_t i = 0; i < data_.size(); ++i) data_[i] -= grad.data_[i] * scale;
}

void Matrix::add_broadcast(const Matrix& vec) {
    if (vec.rows_ != rows_ || vec.cols_ != 1) {
        throw std::invalid_argument("Dimension mismatch in add_broadcast");
    }
    bool use_omp = (rows_ * cols_) > 10000;
    #pragma omp parallel for if(use_omp)
    for (size_t i = 0; i < rows_; ++i) {
        scalar_t val = vec(i, 0);
        for (size_t j = 0; j < cols_; ++j) {
            (*this)(i, j) += val;
        }
    }
}

void Matrix::sum_columns(Matrix& out) const {
    if (out.rows() != rows_ || out.cols() != 1) {
        throw std::invalid_argument("Dimension mismatch in sum_columns");
    }
    out.fill(0.0f);
    bool use_omp = (rows_ * cols_) > 10000;
    #pragma omp parallel for if(use_omp)
    for (size_t i = 0; i < rows_; ++i) {
        scalar_t sum = 0.0f;
        for (size_t j = 0; j < cols_; ++j) {
            sum += (*this)(i, j);
        }
        out(i, 0) = sum;
    }
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    for (size_t i = 0; i < m.rows_; ++i) {
        for (size_t j = 0; j < m.cols_; ++j) {
            os << m(i, j) << (j == m.cols_ - 1 ? "" : " ");
        }
        os << "\n";
    }
    return os;
}
