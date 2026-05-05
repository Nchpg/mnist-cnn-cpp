#include "utils/matrix.hpp"

#include <algorithm>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <string>

Matrix::Matrix(size_t rows, size_t cols, scalar_t initial_val)
    : rows_(rows)
    , cols_(cols)
    , capacity_(rows * cols)
    , data_(rows * cols, initial_val)
{}

scalar_t &Matrix::at(size_t r, size_t r_idx)
{
    return (*this)(r, r_idx);
}

const scalar_t &Matrix::at(size_t r, size_t c) const
{
    return (*this)(r, c);
}

void Matrix::fill(scalar_t value) noexcept
{
    size_t n = rows_ * cols_;
    std::fill(data_.begin(), data_.begin() + n, value);
}

void Matrix::random_uniform(scalar_t scale, std::mt19937 &gen)
{
    std::uniform_real_distribution<scalar_t> dist(-scale, scale);
    size_t n = rows_ * cols_;
    for (size_t i = 0; i < n; ++i)
        data_[i] = dist(gen);
}

void Matrix::random_normal(scalar_t std_dev, std::mt19937 &gen)
{
    std::normal_distribution<scalar_t> dist(0.0f, std_dev);
    size_t n = rows_ * cols_;
    for (size_t i = 0; i < n; ++i)
        data_[i] = dist(gen);
}

void Matrix::reshape(size_t rows, size_t cols)
{
    size_t new_size = rows * cols;
    rows_ = rows;
    cols_ = cols;
    if (new_size > capacity_)
    {
        capacity_ = new_size;
        data_.resize(new_size);
    }
}

void Matrix::multiply(const Matrix &A, const Matrix &B, Matrix &C)
{
    if (A.cols_ != B.rows_ || C.rows_ != A.rows_ || C.cols_ != B.cols_)
    {
        throw std::invalid_argument("Dimension mismatch in multiply");
    }
    C.fill(0.0f);
    const size_t M = A.rows_, K = A.cols_, N = B.cols_;

#pragma omp parallel for if (M * K * N > 50000)
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t k = 0; k < K; ++k)
        {
            scalar_t a_val = A(i, k);
            if (a_val == 0.0f)
                continue;
            for (size_t j = 0; j < N; ++j)
            {
                C(i, j) += a_val * B(k, j);
            }
        }
    }
}

void Matrix::multiply_transA(const Matrix &A, const Matrix &B, Matrix &C)
{
    if (A.rows_ != B.rows_ || C.rows_ != A.cols_ || C.cols_ != B.cols_)
    {
        throw std::invalid_argument("Dimension mismatch in multiply_transA");
    }
    C.fill(0.0f);
    const size_t K = A.rows_, M = A.cols_, N = B.cols_;

#pragma omp parallel for if (M * K * N > 50000)
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t k = 0; k < K; ++k)
        {
            scalar_t a_val = A(k, i);
            if (a_val == 0.0f)
                continue;
            for (size_t j = 0; j < N; ++j)
            {
                C(i, j) += a_val * B(k, j);
            }
        }
    }
}

void Matrix::multiply_transB(const Matrix &A, const Matrix &B, Matrix &C)
{
    if (A.cols_ != B.cols_ || C.rows_ != A.rows_ || C.cols_ != B.rows_)
    {
        throw std::invalid_argument("Dimension mismatch in multiply_transB");
    }
    C.fill(0.0f);
    const size_t M = A.rows_, K = A.cols_, N = B.rows_;

#pragma omp parallel for if (M * K * N > 50000)
    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            scalar_t sum = 0.0f;
            for (size_t k = 0; k < K; ++k)
            {
                sum += A(i, k) * B(j, k);
            }
            C(i, j) = sum;
        }
    }
}

void Matrix::transpose(const Matrix &m, Matrix &out)
{
    if (out.rows_ != m.cols_ || out.cols_ != m.rows_)
    {
        throw std::invalid_argument("Dimension mismatch in transpose");
    }
    for (size_t i = 0; i < m.rows_; ++i)
    {
        for (size_t j = 0; j < m.cols_; ++j)
        {
            out(j, i) = m(i, j);
        }
    }
}

Matrix &Matrix::operator+=(const Matrix &other)
{
    size_t n = rows_ * cols_;
#pragma omp simd
    for (size_t i = 0; i < n; ++i)
        data_[i] += other.data_[i];
    return *this;
}

Matrix &Matrix::operator-=(const Matrix &other)
{
    size_t n = rows_ * cols_;
#pragma omp simd
    for (size_t i = 0; i < n; ++i)
        data_[i] -= other.data_[i];
    return *this;
}

void Matrix::add_scaled(const Matrix &other, scalar_t scale)
{
    size_t n = rows_ * cols_;
#pragma omp simd
    for (size_t i = 0; i < n; ++i)
        data_[i] += other.data_[i] * scale;
}

void Matrix::subtract_scaled(const Matrix &grad, scalar_t scale)
{
    size_t n = rows_ * cols_;
#pragma omp simd
    for (size_t i = 0; i < n; ++i)
        data_[i] -= grad.data_[i] * scale;
}

void Matrix::add_broadcast(const Matrix &vec)
{
    if (vec.rows_ != rows_ || vec.cols_ != 1)
    {
        throw std::invalid_argument("Dimension mismatch in add_broadcast");
    }
    bool use_omp = (rows_ * cols_) > 10000;
#pragma omp parallel for if (use_omp)
    for (size_t i = 0; i < rows_; ++i)
    {
        scalar_t val = vec(i, 0);
        for (size_t j = 0; j < cols_; ++j)
        {
            (*this)(i, j) += val;
        }
    }
}

void Matrix::sum_columns(Matrix &out) const
{
    if (out.rows() != rows_ || out.cols() != 1)
    {
        throw std::invalid_argument("Dimension mismatch in sum_columns");
    }
    out.fill(0.0f);
    bool use_omp = (rows_ * cols_) > 10000;
#pragma omp parallel for if (use_omp)
    for (size_t i = 0; i < rows_; ++i)
    {
        scalar_t sum = 0.0f;
        for (size_t j = 0; j < cols_; ++j)
        {
            sum += (*this)(i, j);
        }
        out(i, 0) = sum;
    }
}

void Matrix::save(std::ostream &os) const
{
    uint64_t r = static_cast<uint64_t>(rows_);
    uint64_t c = static_cast<uint64_t>(cols_);
    os.write(reinterpret_cast<const char *>(&r), sizeof(r));
    os.write(reinterpret_cast<const char *>(&c), sizeof(c));
    os.write(reinterpret_cast<const char *>(data_.data()),
             r * c * sizeof(scalar_t));
}

void Matrix::load(std::istream &is)
{
    uint64_t r, c;
    is.read(reinterpret_cast<char *>(&r), sizeof(r));
    is.read(reinterpret_cast<char *>(&c), sizeof(c));
    if (r != rows_ || c != cols_)
    {
        throw std::runtime_error("Matrix dimension mismatch during load");
    }
    is.read(reinterpret_cast<char *>(data_.data()), r * c * sizeof(scalar_t));
}

std::ostream &operator<<(std::ostream &os, const Matrix &m)
{
    for (size_t i = 0; i < m.rows_; ++i)
    {
        for (size_t j = 0; j < m.cols_; ++j)
        {
            os << m(i, j) << (j == m.cols_ - 1 ? "" : " ");
        }
        os << "\n";
    }
    return os;
}
