#pragma once
#include "matrix.hpp"

class TensorBatch : public Matrix {
private:
    size_t channels_;
    size_t height_;
    size_t width_;

public:

    TensorBatch() : Matrix(), channels_(0), height_(0), width_(0) {}



    TensorBatch(size_t channels, size_t height, size_t width, size_t batch_size, scalar_t init_val = 0.0f)
        : Matrix(channels * height * width, batch_size, init_val), 
          channels_(channels), height_(height), width_(width) {}


    void copy_from(const Matrix& m) {
        this->get_raw_vector() = m.get_raw_vector();
    }

    void reshape4D(size_t channels, size_t height, size_t width, size_t batch_size) {
        channels_ = channels;
        height_ = height;
        width_ = width;
        Matrix::reshape(channels * height * width, batch_size);
    }


    using Matrix::operator();




    inline scalar_t& operator()(size_t b, size_t c, size_t y, size_t x) {
        return Matrix::operator()((c * height_ + y) * width_ + x, b);
    }
    
    inline const scalar_t& operator()(size_t b, size_t c, size_t y, size_t x) const {
        return Matrix::operator()((c * height_ + y) * width_ + x, b);
    }


    size_t channels() const { return channels_; }
    size_t height() const { return height_; }
    size_t width() const { return width_; }
    size_t batch_size() const { return cols(); }


    static inline const scalar_t& read_mat(const Matrix& m, size_t b, size_t c, size_t y, size_t x, size_t h, size_t w) {
        return m((c * h + y) * w + x, b);
    }
};
