#pragma once

#include <cmath>
#include <initializer_list>
#include <istream>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include "utils/types.hpp"

class Tensor
{
private:
    Shape shape_;
    std::vector<scalar_t> data_;
    std::vector<size_t> strides_;

    void compute_strides();

public:
    Tensor() = default;
    explicit Tensor(Shape shape, scalar_t init_val = 0.0f);
    explicit Tensor(std::initializer_list<size_t> dims, scalar_t init_val = 0.0f);
    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other) = default;

    const Shape& shape() const noexcept
    {
        return shape_;
    }
    size_t size() const noexcept
    {
        return data_.size();
    }
    size_t rank() const noexcept
    {
        return shape_.rank();
    }
    const std::vector<scalar_t>& data() const
    {
        return data_;
    }
    std::vector<scalar_t>& data()
    {
        return data_;
    }

    const scalar_t* data_ptr() const noexcept
    {
        return data_.data();
    }
    scalar_t* data_ptr() noexcept
    {
        return data_.data();
    }

    void reshape(Shape new_shape);
    void reshape(Shape new_shape, scalar_t init_val);
    void fill(scalar_t value);

    void random_uniform(scalar_t scale, std::mt19937& gen);
    void random_normal(scalar_t std_dev, std::mt19937& gen);

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    void add_scaled(const Tensor& other, scalar_t scale);
    static void sum_rows(const Tensor& input, Tensor& output);

    static void matmul(const Tensor& A, const Tensor& B, Tensor& C, bool transA = false, bool transB = false);

    scalar_t& operator()(size_t n, size_t c, size_t h, size_t w);
    const scalar_t& operator()(size_t n, size_t c, size_t h, size_t w) const;

    scalar_t& operator()(size_t i, size_t j);
    const scalar_t& operator()(size_t i, size_t j) const;

    scalar_t& operator()(size_t i);
    const scalar_t& operator()(size_t i) const;

    void save(std::ostream& os) const;
    void load(std::istream& is);
};