#pragma once

#include <algorithm>
#include <vector>

#include "utils/tensor.hpp"

namespace Utils
{
inline std::vector<size_t> argmax(const Tensor& m)
{
    std::vector<size_t> results(m.shape()[0]);
    for (size_t b = 0; b < m.shape()[0]; ++b)
    {
        size_t best = 0;
        for (size_t c = 1; c < m.shape()[1]; ++c)
        {
            if (m(b, c) > m(b, best))
                best = c;
        }
        results[b] = best;
    }
    return results;
}
} // namespace Utils