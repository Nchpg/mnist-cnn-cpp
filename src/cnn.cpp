#include "cnn.hpp"
#include "activation.hpp"
#include "conv_layer.hpp"
#include "relu_layer.hpp"
#include "pooling_layer.hpp"
#include "dense_layer.hpp"
#include "flatten_layer.hpp"
#include "dropout_layer.hpp"
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
#include <nlohmann/json.hpp>

using json = nlohmann::json;

CNN::CNN(const std::string& config_path, unsigned int seed) : model_(), gen_(seed) {
    std::ifstream f(config_path);
    json config_data = json::parse(f);

    Shape3D current_shape = {
        config_data["input_shape"]["channels"],
        config_data["input_shape"]["height"],
        config_data["input_shape"]["width"]
    };

    for (const auto& layer_def : config_data["layers"]) {
        std::string type = layer_def["type"];

        if (type == "Conv") {
            size_t filters = layer_def["filters"];
            size_t kernel_size = layer_def["kernel_size"];
            auto layer = std::make_unique<ConvLayer>(
                current_shape.height, current_shape.width, current_shape.channels,
                kernel_size, filters, gen_
            );
            current_shape = layer->get_output_shape(current_shape);
            model_.add(std::move(layer));
        } 
        else if (type == "ReLU") {
            auto layer = std::make_unique<ReluLayer>();
            current_shape = layer->get_output_shape(current_shape);
            model_.add(std::move(layer));
        } 
        else if (type == "Pool") {
            size_t pool_size = layer_def["pool_size"];
            size_t stride = layer_def["stride"];
            auto layer = std::make_unique<PoolingLayer>(
                current_shape.height, current_shape.width, current_shape.channels,
                pool_size, stride
            );
            current_shape = layer->get_output_shape(current_shape);
            model_.add(std::move(layer));
        } 
        else if (type == "Flatten") {
            auto layer = std::make_unique<FlattenLayer>();
            current_shape = layer->get_output_shape(current_shape);
            model_.add(std::move(layer));
        } 
        else if (type == "Dense") {
            size_t units = layer_def["units"];
            auto layer = std::make_unique<DenseLayer>(current_shape.size(), units, gen_);
            current_shape = layer->get_output_shape(current_shape);
            model_.add(std::move(layer));
        }
        else if (type == "Dropout") {
            scalar_t ratio = layer_def["ratio"];
            auto layer = std::make_unique<DropoutLayer>(ratio, gen_);
            current_shape = layer->get_output_shape(current_shape);
            model_.add(std::move(layer));
        }
    }
}

std::vector<scalar_t> CNN::predict(const Matrix& image) {
    model_.set_training(false);
    const Matrix& logits = model_.forward(image);
    Matrix probs(logits.rows(), logits.cols());
    Activation::softmax(logits, probs);
    
    std::vector<scalar_t> result(probs.rows());
    for (size_t i = 0; i < probs.rows(); i++) result[i] = probs(i, 0);
    return result;
}

int CNN::predict_label(const Matrix& image) {
    model_.set_training(false);
    const Matrix& logits = model_.forward(image);
    auto results = Activation::argmax(logits);
    return static_cast<int>(results[0]);
}

void CNN::train(MnistDataset& dataset, size_t epochs, scalar_t learning_rate) {
    model_.set_training(true);
    const size_t batch_size = 64;
    const size_t total_samples = dataset.count();
    const int bar_width = 30;

    const scalar_t beta1 = 0.9f;
    const scalar_t beta2 = 0.999f;
    const scalar_t epsilon = 1e-8f;
    scalar_t beta1_t = 1.0f;
    scalar_t beta2_t = 1.0f;
    
    for (size_t epoch = 0; epoch < epochs; epoch++) {
        dataset.shuffle_indices(); 
        size_t correct = 0;
        scalar_t total_loss = 0.0f;
        
        size_t num_full_batches = total_samples / batch_size;
        size_t samples_to_process = num_full_batches * batch_size;

        for (size_t sample = 0; sample < samples_to_process; sample += batch_size) {
            beta1_t *= beta1;
            beta2_t *= beta2;
            scalar_t m_corr = 1.0f / (1.0f - beta1_t);
            scalar_t v_corr = 1.0f / (1.0f - beta2_t);

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
            model_.update_weights_adam(learning_rate, beta1, beta2, epsilon, m_corr, v_corr);
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
    model_.set_training(false);
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
