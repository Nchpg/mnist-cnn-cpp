#ifndef FLATTEN_LAYER_HPP
#define FLATTEN_LAYER_HPP

#include "layer.hpp"
#include <string>
#include <stdexcept>
#include <algorithm>

class FlattenLayer : public Layer {
public:
    static constexpr const char* LAYER_MARKER = "FLAT";

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
        uint32_t marker = make_marker(LAYER_MARKER);
        os.write(reinterpret_cast<const char*>(&marker), sizeof(marker));
    }

    void load(std::istream& is) override {
        uint32_t marker;
        is.read(reinterpret_cast<char*>(&marker), sizeof(marker));
        if (marker != make_marker(LAYER_MARKER)) {
            throw std::runtime_error("Invalid FlattenLayer data in binary load");
        }
    }

    nlohmann::json get_config() const override { return {{"type", "Flatten"}}; }

    Shape3D get_output_shape(const Shape3D& input_shape) const override {
        return {input_shape.size(), 1, 1};
    }
};

#endif 