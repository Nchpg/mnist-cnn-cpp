#pragma once

#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

#include "utils/matrix.hpp"
#include "utils/types.hpp"

class Layer
{
public:
    virtual ~Layer() = default;
    virtual const Matrix &forward(const Matrix &input) = 0;
    virtual const Matrix &backward(const Matrix &gradient) = 0;

    virtual void clear_gradients()
    {}
    virtual void set_training(bool /*training*/)
    {}

    virtual std::vector<Parameter> get_parameters()
    {
        return {};
    }

    virtual void save(std::ostream & /*os*/) const
    {}
    virtual void load(std::istream & /*is*/)
    {}

    virtual nlohmann::json get_config() const = 0;

    virtual Shape3D get_output_shape(const Shape3D &input_shape) const = 0;
};