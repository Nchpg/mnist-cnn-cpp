#include "model_config.hpp"
#include <iostream>
#include <cmath>

ModelConfig::ModelConfig() 
    : image_size_(28), class_count_(10), 
      conv_filter_count_(8), conv_kernel_size_(3),
      pool_size_(2), pool_stride_(2),
      dense_hidden_size_(128),
      conv_weight_scale_(0.47f), conv_bias_scale_(0.01f) {}

ModelConfig::Shape ModelConfig::shape_from_config() const {
    Shape shape;
    shape.conv_output_size = image_size_ - conv_kernel_size_ + 1;
    shape.pool_output_size = (shape.conv_output_size - pool_size_) / pool_stride_ + 1;
    shape.feature_count = conv_filter_count_ * shape.pool_output_size * shape.pool_output_size;
    return shape;
}

void ModelConfig::validate_config() const {
    if (image_size_ < conv_kernel_size_) {
        throw std::invalid_argument("Image size must be greater than kernel size");
    }
}

void ModelConfig::print_config() const {
    Shape shape = shape_from_config();
    std::cout << "Model architecture:\n"
              << "  Input         : " << image_size_ << "x" << image_size_ << " grayscale image\n"
              << "  Convolution   : " << conv_filter_count_ << " filters, " << conv_kernel_size_ << "x" << conv_kernel_size_ << " kernel, valid padding -> " << shape.conv_output_size << "x" << shape.conv_output_size << " maps\n"
              << "  Activation    : ReLU\n"
              << "  Pooling       : max pool " << pool_size_ << "x" << pool_size_ << ", stride " << pool_stride_ << " -> " << shape.pool_output_size << "x" << shape.pool_output_size << " maps\n"
              << "  Flatten       : " << conv_filter_count_ << " * " << shape.pool_output_size << " * " << shape.pool_output_size << " = " << shape.feature_count << " features\n"
              << "  Dense         : " << shape.feature_count << " -> " << dense_hidden_size_ << " -> " << class_count_ << "\n"
              << "  Output        : softmax probabilities" << std::endl;
}
