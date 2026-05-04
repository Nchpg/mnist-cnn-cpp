#ifndef LOSS_HPP
#define LOSS_HPP

#include "matrix.hpp"
#include "activation.hpp"
#include <vector>

class Loss {
public:
    virtual ~Loss() = default;
    virtual scalar_t forward(const Matrix& predictions, const std::vector<size_t>& targets) = 0;
    virtual void backward(const Matrix& predictions, const std::vector<size_t>& targets, Matrix& grad_output) = 0;
};

class CrossEntropyLoss : public Loss {
public:
    scalar_t forward(const Matrix& predictions, const std::vector<size_t>& targets) override {
        return Activation::cross_entropy_loss_stable(predictions, targets);
    }

    void backward(const Matrix& predictions, const std::vector<size_t>& targets, Matrix& grad_output) override {
        size_t num_classes = predictions.rows();
        size_t batch_size = predictions.cols();

        if (grad_output.rows() != num_classes || grad_output.cols() != batch_size) {
            grad_output.reshape(num_classes, batch_size);
        }

        scalar_t inv_batch_size = 1.0f / static_cast<scalar_t>(batch_size);

        #pragma omp parallel for collapse(2) if(num_classes * batch_size > 1000)
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < num_classes; ++i) {
                scalar_t target = (i == targets[b] ? 1.0f : 0.0f);
                grad_output(i, b) = (predictions(i, b) - target) * inv_batch_size;
            }
        }
    }
};

#endif
