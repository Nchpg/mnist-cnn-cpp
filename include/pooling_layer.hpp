#ifndef POOLING_LAYER_HPP
#define POOLING_LAYER_HPP

#include "layer.hpp"
#include "matrix.hpp"
#include "tensor_batch.hpp"
#include <vector>

class PoolingLayer : public Layer {
public:
    static constexpr const char* LAYER_NAME = "POOL";

private:
    size_t pool_size_;
    size_t stride_;
    
    const Matrix* input_ptr_ = nullptr;
    TensorBatch output_;       
    TensorBatch grad_input_;   
    


    std::vector<size_t> argmax_indices_; 

public:
    PoolingLayer(size_t input_rows, size_t input_cols, size_t filter_count, size_t pool_size, size_t stride);
    ~PoolingLayer() override = default;

    const Matrix& forward(const Matrix& input) override;
    const Matrix& backward(const Matrix& gradient) override;

    void save(std::ostream& os) const override {
        os << LAYER_NAME << " " << pool_size_ << " " << stride_ << "\n";
    }

    void load(std::istream& is) override {
        std::string type;
        size_t s, st;
        is >> type >> s >> st;
        if (type != LAYER_NAME || s != pool_size_ || st != stride_) {
            throw std::runtime_error("Invalid PoolingLayer data");
        }
    }

    Shape3D get_output_shape(const Shape3D& input_shape) const override {
        return {input_shape.channels, (input_shape.height - pool_size_) / stride_ + 1, (input_shape.width - pool_size_) / stride_ + 1};
    }
};

#endif 