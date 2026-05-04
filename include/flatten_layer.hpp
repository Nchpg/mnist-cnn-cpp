#ifndef FLATTEN_LAYER_HPP
#define FLATTEN_LAYER_HPP

#include "layer.hpp"
#include <string>
#include <stdexcept>
#include <algorithm>

class FlattenLayer : public Layer {
public:
    static constexpr const char* LAYER_NAME = "FLATTEN";

public:
    FlattenLayer() = default;
    ~FlattenLayer() override = default;

    const Matrix& forward(const Matrix& input) override {


        return input;
    }

    const Matrix& backward(const Matrix& gradient) override {

        return gradient;
    }

    void save(std::ostream& os) const override {
        os << LAYER_NAME << "\n";
    }

    void load(std::istream& is) override {
        std::string type;
        if (!(is >> type) || type != LAYER_NAME) {
            throw std::runtime_error("Invalid FlattenLayer data: expected '" + std::string(LAYER_NAME) + "'");
        }
    }

    Shape3D get_output_shape(const Shape3D& input_shape) const override {
        return {input_shape.size(), 1, 1};
    }
};

#endif 