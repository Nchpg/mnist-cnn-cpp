#include "cnn.hpp"
#include "activation.hpp"
#include "conv_layer.hpp"
#include "relu_layer.hpp"
#include "pooling_layer.hpp"
#include "dense_layer.hpp"
#include "flatten_layer.hpp"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <memory>
#include <random>

CNN::CNN(const ModelConfig& config, unsigned int seed)
    : config_(config), model_(), gen_(seed) {
    config_.validate_config();
    
    Shape3D current_shape = {1, config_.image_size(), config_.image_size()};
    

    auto conv1 = std::make_unique<ConvLayer>(
        current_shape.height, current_shape.width, current_shape.channels,
        config_.conv_kernel_size(), config_.conv_filter_count(), gen_
    );
    current_shape = conv1->get_output_shape(current_shape);
    model_.add(std::move(conv1));
    model_.add(std::make_unique<ReluLayer>());

    auto conv2 = std::make_unique<ConvLayer>(
        current_shape.height, current_shape.width, current_shape.channels,
        config_.conv_kernel_size(), config_.conv_filter_count() * 2, gen_
    );
    current_shape = conv2->get_output_shape(current_shape);
    model_.add(std::move(conv2));
    model_.add(std::make_unique<ReluLayer>());

    auto pool1 = std::make_unique<PoolingLayer>(
        current_shape.height, current_shape.width, current_shape.channels,
        config_.pool_size(), config_.pool_stride()
    );
    current_shape = pool1->get_output_shape(current_shape);
    model_.add(std::move(pool1));



    auto conv3 = std::make_unique<ConvLayer>(
        current_shape.height, current_shape.width, current_shape.channels,
        config_.conv_kernel_size(), config_.conv_filter_count() * 4, gen_
    );
    current_shape = conv3->get_output_shape(current_shape);
    model_.add(std::move(conv3));
    model_.add(std::make_unique<ReluLayer>());

    auto conv4 = std::make_unique<ConvLayer>(
        current_shape.height, current_shape.width, current_shape.channels,
        config_.conv_kernel_size(), config_.conv_filter_count() * 8, gen_
    );
    current_shape = conv4->get_output_shape(current_shape);
    model_.add(std::move(conv4));
    model_.add(std::make_unique<ReluLayer>());

    auto pool2 = std::make_unique<PoolingLayer>(
        current_shape.height, current_shape.width, current_shape.channels,
        config_.pool_size(), config_.pool_stride()
    );
    current_shape = pool2->get_output_shape(current_shape);
    model_.add(std::move(pool2));



    auto flatten = std::make_unique<FlattenLayer>();
    current_shape = flatten->get_output_shape(current_shape);
    model_.add(std::move(flatten));

    auto dense1 = std::make_unique<DenseLayer>(current_shape.size(), config_.dense_hidden_size(), gen_);
    current_shape = dense1->get_output_shape(current_shape);
    model_.add(std::move(dense1));
    model_.add(std::make_unique<ReluLayer>());

    auto dense2 = std::make_unique<DenseLayer>(current_shape.size(), config_.class_count(), gen_);
    model_.add(std::move(dense2));
}

std::vector<scalar_t> CNN::predict(const Matrix& image) {
    const Matrix& logits = model_.forward(image);
    Matrix probs(logits.rows(), logits.cols());
    Activation::softmax(logits, probs);
    
    std::vector<scalar_t> result(probs.rows());
    for (size_t i = 0; i < probs.rows(); i++) result[i] = probs(i, 0);
    return result;
}

int CNN::predict_label(const Matrix& image) {
    const Matrix& logits = model_.forward(image);
    auto results = Activation::argmax(logits);
    return static_cast<int>(results[0]);
}

void CNN::train(MnistDataset& dataset, size_t epochs, scalar_t learning_rate) {
    config_.print_config();
    const size_t batch_size = 32;
    const size_t total_samples = dataset.count();
    const int bar_width = 30;
    
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        dataset.shuffle_indices(); 
        size_t correct = 0;
        scalar_t total_loss = 0.0f;
        
        size_t num_full_batches = total_samples / batch_size;
        size_t samples_to_process = num_full_batches * batch_size;

        for (size_t sample = 0; sample < samples_to_process; sample += batch_size) {
            size_t actual_batch_size = batch_size;
            
            Matrix batch_images = dataset.get_batch_images(sample, actual_batch_size);
            std::vector<size_t> batch_labels = dataset.get_batch_labels(sample, actual_batch_size);
            
            const Matrix& logits = model_.forward(batch_images);

            Matrix probs(logits.rows(), actual_batch_size);
            Activation::softmax(logits, probs);

            total_loss += Activation::cross_entropy_loss_stable(probs, batch_labels) * static_cast<scalar_t>(actual_batch_size);

            auto predictions = Activation::argmax(logits);
            for (size_t b = 0; b < actual_batch_size; ++b) {
                if (predictions[b] == batch_labels[b]) correct++;
            }

            Matrix grad_output(logits.rows(), actual_batch_size);
            scalar_t inv_batch_size = 1.0f / static_cast<scalar_t>(actual_batch_size);
            for (size_t b = 0; b < actual_batch_size; ++b) {
                for (size_t i = 0; i < logits.rows(); i++) {
                    scalar_t target = (i == batch_labels[b] ? 1.0f : 0.0f);
                    grad_output(i, b) = (probs(i, b) - target) * inv_batch_size;
                }
            }
            
            model_.backward(grad_output);
            model_.update_weights(learning_rate);
            model_.clear_gradients();
            
            scalar_t progress = static_cast<scalar_t>(sample + actual_batch_size) / static_cast<scalar_t>(samples_to_process);
            std::cout << "\rEpoch " << epoch + 1 << "/" << epochs << " [";
            int pos = static_cast<int>(static_cast<scalar_t>(bar_width) * progress);
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << static_cast<int>(progress * 100.0f) << "% " 
                    << (sample + actual_batch_size) << "/" << samples_to_process << " " << std::flush;
        }
        
        std::cout << "\n  -> loss: " << total_loss / static_cast<scalar_t>(samples_to_process)
                  << " - accuracy: " << 100.0f * static_cast<scalar_t>(correct) / static_cast<scalar_t>(samples_to_process) << "%" << std::endl;
    }
}

scalar_t CNN::accuracy(const MnistDataset& dataset) {
    size_t correct = 0;
    const size_t eval_batch_size = 64;
    for (size_t i = 0; i < dataset.count(); i += eval_batch_size) {
        size_t actual_batch_size = std::min(eval_batch_size, dataset.count() - i);
        Matrix batch_images = dataset.get_batch_images(i, actual_batch_size);
        std::vector<size_t> batch_labels = dataset.get_batch_labels(i, actual_batch_size);
        
        const Matrix& logits = model_.forward(batch_images);
        auto predictions = Activation::argmax(logits);
        
        for (size_t b = 0; b < actual_batch_size; ++b) {
            if (predictions[b] == batch_labels[b]) correct++;
        }
    }
    return static_cast<scalar_t>(correct) / static_cast<scalar_t>(dataset.count());
}

void CNN::save(const std::string& path) const {
    std::ofstream file(path, std::ios::out | std::ios::trunc);
    if (!file.is_open()) throw std::runtime_error("Save failed");
    file << MAGIC << "\n" << model_.layers().size() << "\n";
    for (const auto& layer : model_.layers()) layer->save(file);
    file.close();
}

void CNN::load(const std::string& path) {
    std::ifstream file(path, std::ios::in);
    if (!file.is_open()) throw std::runtime_error("Load failed");
    std::string magic; file >> magic;
    if (magic != MAGIC) throw std::runtime_error("Bad magic");
    size_t count; file >> count;
    if (count != model_.layers().size()) throw std::runtime_error("Arch mismatch");
    for (auto& layer : model_.layers()) layer->load(file);
    file.close();
}

void CNN::print_config() const { config_.print_config(); }
