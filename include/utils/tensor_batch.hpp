#pragma once

#include "utils/matrix.hpp"

class TensorBatch : public Matrix
{
private:
    size_t channels_;
    size_t height_;
    size_t width_;

public:
    TensorBatch();
    TensorBatch(size_t channels, size_t height, size_t width, size_t batch_size,
                scalar_t init_val = 0.0f);

    void copy_from(const Matrix &m);
    void reshape4D(size_t channels, size_t height, size_t width,
                   size_t batch_size);

    size_t channels() const;
    size_t height() const;
    size_t width() const;
    size_t batch_size() const;

    using Matrix::operator();

    scalar_t &operator()(size_t b, size_t c, size_t y, size_t x);
    const scalar_t &operator()(size_t b, size_t c, size_t y, size_t x) const;

    static const scalar_t &read_mat(const Matrix &m, size_t b, size_t c,
                                    size_t y, size_t x, size_t h, size_t w);
};
