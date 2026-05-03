#pragma once

#include "layer.hpp"
#include "matrix.hpp"
#include <algorithm>

class ReluLayer : public Layer {
private:
    const Matrix* input_ptr_ = nullptr; 
    Matrix output_;
    Matrix grad_input_;

public:
    ReluLayer() = default;
    ~ReluLayer() override = default;

    const Matrix& forward(const Matrix& input) override {
        input_ptr_ = &input;
        if (output_.rows() != input.rows() || output_.cols() != input.cols()) {
            output_.reshape(input.rows(), input.cols());
        }
        
        size_t n = input.size();
        const scalar_t* in_ptr = input.data();
        scalar_t* out_ptr = output_.data();


        #pragma omp parallel for if(n > 10000)
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = in_ptr[i] > 0.0f ? in_ptr[i] : 0.0f;
        }
        return output_;
    }

    const Matrix& backward(const Matrix& gradient) override {
        if (grad_input_.rows() != input_ptr_->rows() || grad_input_.cols() != input_ptr_->cols()) {
            grad_input_.reshape(input_ptr_->rows(), input_ptr_->cols());
        }

        size_t n = input_ptr_->size();
        const scalar_t* in_ptr = input_ptr_->data();
        const scalar_t* grad_ptr = gradient.data();
        scalar_t* out_grad_ptr = grad_input_.data();

        #pragma omp parallel for if(n > 10000)
        for (size_t i = 0; i < n; ++i) {
            out_grad_ptr[i] = in_ptr[i] > 0.0f ? grad_ptr[i] : 0.0f;
        }
        return grad_input_;
    }

    void save(std::ostream& os) const override { os << "RELU\n"; }
    void load(std::istream& is) override {
        std::string type;
        is >> type;
        if (type != "RELU") throw std::runtime_error("Architecture mismatch in ReluLayer");
    }

    Shape3D get_output_shape(const Shape3D& input_shape) const override {
        return input_shape;
    }
};
