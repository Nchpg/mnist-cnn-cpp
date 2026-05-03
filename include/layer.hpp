#ifndef LAYER_HPP
#define LAYER_HPP

#include "matrix.hpp"
#include <vector>
#include <memory>

struct Shape3D {
    size_t channels;
    size_t height;
    size_t width;

    size_t size() const { return channels * height * width; }
};

class Layer {
public:
    virtual ~Layer() = default;
    virtual const Matrix& forward(const Matrix& input) = 0;
    virtual const Matrix& backward(const Matrix& gradient) = 0;

    virtual void update_weights(scalar_t /*learning_rate*/) {}
    virtual void update_weights_adam(scalar_t /*learning_rate*/, scalar_t /*beta1*/, scalar_t /*beta2*/, scalar_t /*epsilon*/, scalar_t /*m_corr*/, scalar_t /*v_corr*/) {}
    virtual void clear_gradients() {}
    virtual void set_training(bool /*training*/) {}

    virtual void save(std::ostream& /*os*/) const {}
    virtual void load(std::istream& /*is*/) {}

    virtual Shape3D get_output_shape(const Shape3D& input_shape) const = 0;
};

#endif 