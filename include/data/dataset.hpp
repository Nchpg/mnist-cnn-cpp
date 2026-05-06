#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "utils/tensor.hpp"

class Dataset
{
public:
    virtual ~Dataset() = default;

    virtual size_t count() const = 0;

    virtual void shuffle_indices() = 0;

    virtual void get_batch_images(size_t start_idx, size_t batch_size, Tensor& out_batch) const = 0;
    virtual void get_batch_labels(size_t start_idx, size_t batch_size, std::vector<size_t>& out_labels) const = 0;
};