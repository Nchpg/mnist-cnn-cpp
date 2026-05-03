#pragma once

#include <cstddef>
#include <stdexcept>
#include <iostream>
#include "matrix.hpp"

class ModelConfig {
private:
    size_t image_size_;
    size_t class_count_;
    size_t conv_filter_count_;
    size_t conv_kernel_size_;
    size_t pool_size_;
    size_t pool_stride_;
    size_t dense_hidden_size_;
    scalar_t conv_weight_scale_;
    scalar_t conv_bias_scale_;

public:
    ModelConfig();
    
    size_t image_size() const { return image_size_; }
    size_t class_count() const { return class_count_; }
    size_t conv_filter_count() const { return conv_filter_count_; }
    size_t conv_kernel_size() const { return conv_kernel_size_; }
    size_t pool_size() const { return pool_size_; }
    size_t pool_stride() const { return pool_stride_; }
    size_t dense_hidden_size() const { return dense_hidden_size_; }
    scalar_t conv_weight_scale() const { return conv_weight_scale_; }
    scalar_t conv_bias_scale() const { return conv_bias_scale_; }
    
    struct Shape {
        size_t conv_output_size;
        size_t pool_output_size;
        size_t feature_count;
    };
    
    Shape shape_from_config() const;
    
    void validate_config() const;
    
    void print_config() const;
};

