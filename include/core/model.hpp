#pragma once

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
        std::vector<Parameter> all_params;
        for (auto &layer : layers_)
        {
            auto params = layer->get_parameters();
            all_params.insert(all_params.end(), params.begin(), params.end());
        }
        optimizer->add_parameters(all_params);
        optimizer_ = std::move(optimizer);
    }

    Optimizer *optimizer()
    {
        return optimizer_.get();
    }

    void set_training(bool training)
    {
        for (auto &layer : layers_)
        {
            layer->set_training(training);
        }
    }

    const Matrix &forward(const Matrix &input)
    {
        const Matrix *x = &input;
        for (auto &layer : layers_)
        {
            x = &(layer->forward(*x));
        }
        return *x;
    }

    void backward(const Matrix &gradient)
    {
        const Matrix *grad = &gradient;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
        {
            grad = &((*it)->backward(*grad));
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
        for (auto &layer : layers_)
        {
            layer->clear_gradients();
        }
    }

    const std::vector<std::unique_ptr<Layer>> &layers() const
    {
        return layers_;
    }

    std::vector<std::unique_ptr<Layer>> &layers()
    {
        return layers_;
    }
};