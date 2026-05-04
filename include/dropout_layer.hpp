#ifndef DROPOUT_LAYER_HPP
#define DROPOUT_LAYER_HPP

#include "layer.hpp"
#include <omp.h>
#include <random>
#include <string>
#include <stdexcept>

class DropoutLayer : public Layer {
public:
    static constexpr const char* LAYER_MARKER = "DROP";

private:
    scalar_t ratio_;
    bool is_training_ = true;
    Matrix mask_;
    Matrix output_;
    Matrix grad_input_;
    std::mt19937& gen_;
    std::vector<std::mt19937> thread_gens_;

public:
    DropoutLayer(scalar_t ratio, std::mt19937& gen) : ratio_(ratio), gen_(gen) {
        int max_threads = omp_get_max_threads();
        for (int i = 0; i < max_threads; ++i) {
            thread_gens_.emplace_back(gen());
        }
    }

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

        scalar_t scale = 1.0f / (1.0f - ratio_);
        size_t n = input.size();
        const scalar_t* in_ptr = input.data();
        scalar_t* mask_ptr = mask_.data();
        scalar_t* out_ptr = output_.data();

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::mt19937& local_gen = thread_gens_[tid];
            std::uniform_real_distribution<scalar_t> dist(0.0f, 1.0f);

            #pragma omp for
            for (size_t i = 0; i < n; ++i) {
                if (dist(local_gen) > ratio_) {
                    mask_ptr[i] = scale;
                } else {
                    mask_ptr[i] = 0.0f;
                }
                out_ptr[i] = in_ptr[i] * mask_ptr[i];
            }
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

        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            in_grad_ptr[i] = grad_ptr[i] * mask_ptr[i];
        }

        return grad_input_;
    }

    Shape3D get_output_shape(const Shape3D& input_shape) const override {
        return input_shape;
    }

    void save(std::ostream& os) const override {
        uint32_t marker = make_marker(LAYER_MARKER);
        os.write(reinterpret_cast<const char*>(&marker), sizeof(marker));
        os.write(reinterpret_cast<const char*>(&ratio_), sizeof(ratio_));
    }

    void load(std::istream& is) override {
        uint32_t marker;
        scalar_t r;
        is.read(reinterpret_cast<char*>(&marker), sizeof(marker));
        is.read(reinterpret_cast<char*>(&r), sizeof(r));
        if (marker != make_marker(LAYER_MARKER)) throw std::runtime_error("Arch mismatch in DropoutLayer binary load");
        ratio_ = r;
    }

    nlohmann::json get_config() const override { return {{"type", "Dropout"}, {"ratio", ratio_}}; }
};

#endif
