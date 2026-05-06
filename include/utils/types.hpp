#pragma once

#include <array>
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

static constexpr size_t MAX_DIMS = 4;

class Shape
{
private:
    std::array<size_t, MAX_DIMS> dims_;
    size_t rank_;

public:
    Shape()
        : rank_(0)
    {
        dims_.fill(0);
    }
    Shape(std::initializer_list<size_t> dims)
    {
        if (dims.size() > MAX_DIMS)
            throw std::invalid_argument("Shape rank exceeds MAX_DIMS");
        rank_ = dims.size();
        dims_.fill(0);
        size_t i = 0;
        for (size_t d : dims)
            dims_[i++] = d;
    }
    Shape(const std::vector<size_t> &dims)
    {
        if (dims.size() > MAX_DIMS)
            throw std::invalid_argument("Shape rank exceeds MAX_DIMS");
        rank_ = dims.size();
        dims_.fill(0);
        for (size_t i = 0; i < rank_; ++i)
            dims_[i] = dims[i];
    }

    bool operator==(const Shape &other) const
    {
        if (rank_ != other.rank_)
            return false;
        for (size_t i = 0; i < rank_; ++i)
            if (dims_[i] != other.dims_[i])
                return false;
        return true;
    }
    bool operator!=(const Shape &other) const
    {
        return !(*this == other);
    }

    size_t rank() const
    {
        return rank_;
    }
    size_t size() const
    {
        if (rank_ == 0)
            return 0;
        size_t s = 1;
        for (size_t i = 0; i < rank_; ++i)
            s *= dims_[i];
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

    const std::array<size_t, MAX_DIMS> &dims() const
    {
        return dims_;
    }
};