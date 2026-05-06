#pragma once

#include <iostream>

#include "layers/layer.hpp"
#include "optimizers/optimizer.hpp"

class Model
{
private:
    std::vector<std::unique_ptr<Layer>> layers_;
    std::unique_ptr<Optimizer> optimizer_;

public:
    void add(std::unique_ptr<Layer> layer)
    {
        layers_.push_back(std::move(layer));
    }

    void set_optimizer(std::unique_ptr<Optimizer> optimizer)
    {
        std::vector<Tensor*> all_weights;
        std::vector<Tensor*> all_grads;
        for (auto& layer : layers_)
        {
            auto weights = layer->get_weights();
            auto grads = layer->get_gradients();
            all_weights.insert(all_weights.end(), weights.begin(), weights.end());
            all_grads.insert(all_grads.end(), grads.begin(), grads.end());
        }
        optimizer->set_parameters(all_weights, all_grads);
        optimizer_ = std::move(optimizer);
    }

    Optimizer* optimizer()
    {
        return optimizer_.get();
    }

    const Tensor& forward(const Tensor& input, std::vector<std::unique_ptr<LayerContext>>& contexts,
                          bool is_training) const
    {
        contexts.resize(layers_.size());
        const Tensor* x = &input;
        for (size_t i = 0; i < layers_.size(); ++i)
        {
            x = &(layers_[i]->forward(*x, contexts[i], is_training));
        }
        return *x;
    }

    void backward(const Tensor& gradient, std::vector<std::unique_ptr<LayerContext>>& contexts, bool is_training)
    {
        if (contexts.size() != layers_.size())
        {
            throw std::runtime_error("Context size mismatch! You must call "
                                     "forward() before backward().");
        }
        const Tensor* current_grad = &gradient;
        for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i)
        {
            current_grad = &(layers_[i]->backward(*current_grad, contexts[i], is_training));
        }
    }

    void backward_skip_last(const Tensor& gradient, std::vector<std::unique_ptr<LayerContext>>& contexts,
                            bool is_training)
    {
        if (layers_.empty())
            return;
        if (contexts.size() != layers_.size())
        {
            throw std::runtime_error("Context size mismatch! You must call "
                                     "forward() before backward().");
        }
        const Tensor* current_grad = &gradient;
        for (int i = static_cast<int>(layers_.size()) - 1; i > 0; --i)
        {
            current_grad = &(layers_[i]->backward(*current_grad, contexts[i], is_training));
        }
    }

    void step()
    {
        if (optimizer_)
        {
            optimizer_->step();
        }
    }

    void clear_gradients()
    {
        for (auto& layer : layers_)
        {
            layer->clear_gradients();
        }
    }

    const std::vector<std::unique_ptr<Layer>>& layers() const
    {
        return layers_;
    }

    std::vector<std::unique_ptr<Layer>>& layers()
    {
        return layers_;
    }
};