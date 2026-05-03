#ifndef DROPOUT_LAYER_HPP
#define DROPOUT_LAYER_HPP

#include "layer.hpp"
#include <random>
#include <string>
#include <stdexcept>

class DropoutLayer : public Layer {
private:
    scalar_t ratio_;
    bool is_training_ = true;
    Matrix mask_;
    Matrix output_;
    Matrix grad_input_;
    std::mt19937& gen_;

public:
    DropoutLayer(scalar_t ratio, std::mt19937& gen) : ratio_(ratio), gen_(gen) {}

    void set_training(bool training) override {
        is_training_ = training;
    }

    const Matrix& forward(const Matrix& input) override {
        if (!is_training_) {
            return input;
        }

        if (mask_.rows() != input.rows() || mask_.cols() != input.cols()) {
            mask_.reshape(input.rows(), input.cols());
            output_.reshape(input.rows(), input.cols());
        }

        std::uniform_real_distribution<scalar_t> dist(0.0f, 1.0f);
        scalar_t scale = 1.0f / (1.0f - ratio_);
        
        size_t n = input.size();
        const scalar_t* in_ptr = input.data();
        scalar_t* mask_ptr = mask_.data();
        scalar_t* out_ptr = output_.data();

        for (size_t i = 0; i < n; ++i) {
            if (dist(gen_) > ratio_) {
                mask_ptr[i] = scale;
            } else {
                mask_ptr[i] = 0.0f;
            }
            out_ptr[i] = in_ptr[i] * mask_ptr[i];
        }

        return output_;
    }

    const Matrix& backward(const Matrix& gradient) override {
        if (!is_training_) {
            return gradient;
        }

        if (grad_input_.rows() != gradient.rows() || grad_input_.cols() != gradient.cols()) {
            grad_input_.reshape(gradient.rows(), gradient.cols());
        }

        size_t n = gradient.size();
        const scalar_t* grad_ptr = gradient.data();
        const scalar_t* mask_ptr = mask_.data();
        scalar_t* in_grad_ptr = grad_input_.data();

        for (size_t i = 0; i < n; ++i) {
            in_grad_ptr[i] = grad_ptr[i] * mask_ptr[i];
        }

        return grad_input_;
    }

    Shape3D get_output_shape(const Shape3D& input_shape) const override {
        return input_shape;
    }

    void save(std::ostream& os) const override {
        os << "DROPOUT " << ratio_ << "\n";
    }

    void load(std::istream& is) override {
        std::string type;
        scalar_t r;
        is >> type >> r;
        if (type != "DROPOUT") throw std::runtime_error("Arch mismatch in DropoutLayer: expected 'DROPOUT'");
        ratio_ = r;
    }
};

#endif
