#pragma once

#include "model_config.hpp"
#include "matrix.hpp"
#include "model.hpp"
#include "mnist_dataset.hpp"
#include <vector>
#include <array>
#include <string>

#include <random>

class CNN {
private:
    ModelConfig config_;
    Model model_;
    std::mt19937 gen_;
    static constexpr const char* MAGIC = "mnist-cnn-v1";

public:
    CNN(const ModelConfig& config, unsigned int seed = 42);
    ~CNN() = default;
    
    std::vector<scalar_t> predict(const Matrix& image);
    int predict_label(const Matrix& image);
    
    void train(MnistDataset& dataset, size_t epochs, scalar_t learning_rate);
    scalar_t accuracy(const MnistDataset& dataset);
    void save(const std::string& path) const;
    void load(const std::string& path);
    void print_config() const;
};