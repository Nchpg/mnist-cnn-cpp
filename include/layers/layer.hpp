#pragma once

#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

#include "utils/tensor.hpp"

struct LayerContext
{
    virtual ~LayerContext() = default;
};

class Layer
{
public:
    virtual ~Layer() = default;

    virtual const Tensor& forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool is_training) const = 0;
    virtual const Tensor& backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx, bool is_training) = 0;

    virtual void clear_gradients()
    {}

    virtual std::vector<Tensor*> get_weights()
    {
        return {};
    }
    virtual std::vector<Tensor*> get_gradients()
    {
        return {};
    }

    virtual void save(std::ostream& /*os*/) const
    {}
    virtual void load(std::istream& /*is*/)
    {}

    virtual nlohmann::json get_config() const = 0;

    virtual Shape get_output_shape(const Shape& input_shape) const = 0;
    virtual Shape get_input_shape(const Shape& output_shape) const = 0;

protected:
    template <typename T>
    T* get_context(std::unique_ptr<LayerContext>& ctx) const
    {
        if (!ctx)
        {
            ctx = std::make_unique<T>();
        }
        return static_cast<T*>(ctx.get());
    }
};