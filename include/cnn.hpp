#pragma once

#include "matrix.hpp"
#include "model.hpp"
#include "mnist_dataset.hpp"
#include <vector>
#include <array>
#include <string>

#include <random>

enum class OptimizerType {
    SGD,
    Adam
};

class CNN {
private:
    Model model_;
    std::mt19937 gen_;
    static constexpr const char* MAGIC = "mnist-cnn-v1";
    OptimizerType optimizer_type_;
    scalar_t learning_rate_;

public:
    CNN(const std::string& config_path, unsigned int seed = 42, OptimizerType optimizer_type = OptimizerType::Adam, scalar_t learning_rate = 0.001f);
    ~CNN() = default;
    
    std::vector<scalar_t> predict(const Matrix& image);
    int predict_label(const Matrix& image);
    
    void train(MnistDataset& dataset, size_t epochs, scalar_t learning_rate);
    scalar_t accuracy(const MnistDataset& dataset);
    void save(const std::string& path) const;
    void load(const std::string& path);
};