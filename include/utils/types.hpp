#pragma once

#include <cstring>
#include <cstddef>

#include "utils/matrix.hpp"

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

struct Parameter
{
    Matrix *value;
    Matrix *gradient;
};