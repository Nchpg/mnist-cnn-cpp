#pragma once

#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <vector>

using scalar_t = float;

inline uint32_t make_marker(const char *s)
{
    uint32_t v;
    std::memcpy(&v, s, 4);
    return v;
}

struct Shape3D
{
    size_t channels;
    size_t height;
    size_t width;

    size_t size() const
    {
        return channels * height * width;
    }
};

class Shape
{
private:
    std::vector<size_t> dims_;

public:
    Shape()
        : dims_()
    {}
    Shape(std::vector<size_t> dims)
        : dims_(std::move(dims))
    {}
    Shape(std::initializer_list<size_t> dims)
        : dims_(dims)
    {}

    bool operator==(const Shape &other) const
    {
        return dims_ == other.dims_;
    }
    bool operator!=(const Shape &other) const
    {
        return dims_ != other.dims_;
    }

    size_t rank() const
    {
        return dims_.size();
    }
    size_t size() const
    {
        size_t s = 1;
        for (size_t d : dims_)
            s *= d;
        return s;
    }

    size_t operator[](size_t i) const
    {
        return dims_.at(i);
    }
    size_t &operator[](size_t i)
    {
        return dims_.at(i);
    }

    const std::vector<size_t> &dims() const
    {
        return dims_;
    }
};