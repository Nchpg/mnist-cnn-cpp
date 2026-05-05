#pragma once

#include <vector>
#include <algorithm>

#include "utils/matrix.hpp"

namespace Utils
{
    inline std::vector<size_t> argmax(const Matrix &m)
    {
        std::vector<size_t> results(m.cols());
        for (size_t j = 0; j < m.cols(); ++j) {
            size_t best = 0;
            for (size_t i = 1; i < m.rows(); ++i) {
                if (m(i, j) > m(best, j))
                    best = i;
            }
            results[j] = best;
        }
        return results;
    }
}