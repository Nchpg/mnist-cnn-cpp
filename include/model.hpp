#ifndef MODEL_HPP
#define MODEL_HPP

#include "layer.hpp"
#include <vector>
#include <memory>

class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers_;

public:

    void add(std::unique_ptr<Layer> layer) {
        layers_.push_back(std::move(layer));
    }

    void set_training(bool training) {
        for (auto& layer : layers_) {
            layer->set_training(training);
        }
    }


    const Matrix& forward(const Matrix& input) {
        const Matrix* x = &input;
        for (auto& layer : layers_) {
            x = &(layer->forward(*x));
        }
        return *x;
    }


    void backward(const Matrix& gradient) {
        const Matrix* grad = &gradient;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            grad = &((*it)->backward(*grad));
        }
    }

    void update_weights(scalar_t learning_rate) {
        for (auto& layer : layers_) {
            layer->update_weights(learning_rate);
        }
    }

    void update_weights_adam(scalar_t learning_rate, scalar_t beta1, scalar_t beta2, scalar_t epsilon, scalar_t m_corr, scalar_t v_corr) {
        for (auto& layer : layers_) {
            layer->update_weights_adam(learning_rate, beta1, beta2, epsilon, m_corr, v_corr);
        }
    }

    void clear_gradients() {
        for (auto& layer : layers_) {
            layer->clear_gradients();
        }
    }


    const std::vector<std::unique_ptr<Layer>>& layers() const {
        return layers_;
    }


    std::vector<std::unique_ptr<Layer>>& layers() {
        return layers_;
    }
};

#endif 