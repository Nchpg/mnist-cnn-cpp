#include "utils/tensor_batch.hpp"

TensorBatch::TensorBatch()
    : Matrix()
    , channels_(0)
    , height_(0)
    , width_(0)
{
}

TensorBatch::TensorBatch(size_t channels, size_t height, size_t width, size_t batch_size,
                         scalar_t init_val)
    : Matrix(channels * height * width, batch_size, init_val)
    , channels_(channels)
    , height_(height)
    , width_(width)
{
}

void TensorBatch::copy_from(const Matrix &m)
{
    this->get_raw_vector() = m.get_raw_vector();
}

void TensorBatch::reshape4D(size_t channels, size_t height, size_t width,
                            size_t batch_size)
{
    channels_ = channels;
    height_ = height;
    width_ = width;
    Matrix::reshape(channels * height * width, batch_size);
}

size_t TensorBatch::channels() const
{
    return channels_;
}

size_t TensorBatch::height() const
{
    return height_;
}

size_t TensorBatch::width() const
{
    return width_;
}

size_t TensorBatch::batch_size() const
{
    return cols();
}

scalar_t &TensorBatch::operator()(size_t b, size_t c, size_t y, size_t x)
{
    return Matrix::operator()((c * height_ + y) * width_ + x, b);
}

const scalar_t &TensorBatch::operator()(size_t b, size_t c, size_t y,
                                        size_t x) const
{
    return Matrix::operator()((c * height_ + y) * width_ + x, b);
}

const scalar_t &TensorBatch::read_mat(const Matrix &m, size_t b, size_t c,
                                      size_t y, size_t x, size_t h,
                                      size_t w)
{
    return m((c * h + y) * w + x, b);
}