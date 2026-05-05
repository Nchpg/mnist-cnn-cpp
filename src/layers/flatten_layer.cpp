#include "layers/flatten_layer.hpp"

FlattenLayer::FlattenLayer()
{}

const Matrix &FlattenLayer::forward(const Matrix &input)
{
    return input;
}

const Matrix &FlattenLayer::backward(const Matrix &gradient)
{
    return gradient;
}

void FlattenLayer::save(std::ostream &os) const
{
    uint32_t marker = make_marker(LAYER_MARKER);
    os.write(reinterpret_cast<const char *>(&marker), sizeof(marker));
}

void FlattenLayer::load(std::istream &is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    if (marker != make_marker(LAYER_MARKER))
    {
        throw std::runtime_error("Invalid FlattenLayer data in binary load");
    }
}

nlohmann::json FlattenLayer::get_config() const
{
    return { { "type", "Flatten" } };
}

Shape3D FlattenLayer::get_output_shape(const Shape3D &input_shape) const
{
    return { input_shape.size(), 1, 1 };
}