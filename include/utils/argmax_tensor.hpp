#pragma once

#include <cassert>
#include <stdexcept>
#include <vector>

#include "utils/types.hpp"

class ArgmaxTensor
{
private:
    Shape shape_;
    std::vector<size_t> data_;
    std::vector<size_t> strides_;

    void compute_strides()
    {
        strides_.resize(shape_.rank());
        size_t stride = 1;
        for (int i = static_cast<int>(shape_.rank()) - 1; i >= 0; --i)
        {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

public:
    ArgmaxTensor() = default;

    void reshape(Shape new_shape)
    {
        shape_ = new_shape;
        data_.resize(new_shape.size());
        compute_strides();
    }

    size_t size() const noexcept
    {
        return data_.size();
    }
    const Shape& shape() const noexcept
    {
        return shape_;
    }

    size_t& operator()(size_t b, size_t c, size_t i, size_t j)
    {
        assert(shape_.rank() == 4);
        assert(b < shape_[0] && c < shape_[1] && i < shape_[2] && j < shape_[3]);
        return data_[b * strides_[0] + c * strides_[1] + i * strides_[2] + j];
    }

    const size_t& operator()(size_t b, size_t c, size_t i, size_t j) const
    {
        assert(shape_.rank() == 4);
        assert(b < shape_[0] && c < shape_[1] && i < shape_[2] && j < shape_[3]);
        return data_[b * strides_[0] + c * strides_[1] + i * strides_[2] + j];
    }
};
