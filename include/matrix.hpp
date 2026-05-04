#pragma once

#include <vector>
#include <iostream>
#include <random>



using scalar_t = float;

class Matrix {
private:
    size_t rows_ = 0;
    size_t cols_ = 0;
    size_t capacity_ = 0;
    std::vector<scalar_t> data_;

public:

    Matrix() = default;
    Matrix(size_t rows, size_t cols, scalar_t initial_val = 0.0f);

    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(Matrix&&) noexcept = default;


    inline scalar_t& operator()(size_t r, size_t c) { return data_[r * cols_ + c]; }
    inline const scalar_t& operator()(size_t r, size_t c) const { return data_[r * cols_ + c]; }


    scalar_t& at(size_t r, size_t c);
    const scalar_t& at(size_t r, size_t c) const;
    

    size_t rows() const noexcept { return rows_; }
    size_t cols() const noexcept { return cols_; }
    size_t size() const noexcept { return rows_ * cols_; }
    const std::vector<scalar_t>& get_raw_vector() const noexcept { return data_; }
    std::vector<scalar_t>& get_raw_vector() noexcept { return data_; }
    scalar_t* data() noexcept { return data_.data(); }
    const scalar_t* data() const noexcept { return data_.data(); }


    void fill(scalar_t value) noexcept;
    void random_uniform(scalar_t scale, std::mt19937& gen);
    void reshape(size_t rows, size_t cols);
    

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);


    static void multiply(const Matrix& A, const Matrix& B, Matrix& C);
    static void multiply_transA(const Matrix& A, const Matrix& B, Matrix& C); 
    static void multiply_transB(const Matrix& A, const Matrix& B, Matrix& C); 
    static void transpose(const Matrix& m, Matrix& out);
    void add_scaled(const Matrix& other, scalar_t scale);
    void add_outer_product(const Matrix& left, const Matrix& right);
    void subtract_scaled(const Matrix& grad, scalar_t scale);
    void add_broadcast(const Matrix& vec);
    void sum_columns(Matrix& out) const;

    void save(std::ostream& os) const;
    void load(std::istream& is);

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);
};
