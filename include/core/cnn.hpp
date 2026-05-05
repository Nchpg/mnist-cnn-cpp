#pragma once

#include <array>
#include <random>
#include <string>
#include <vector>

#include "utils/matrix.hpp"
#include "core/model.hpp"
#include "data/mnist_dataset.hpp"

enum class OptimizerType
{
    SGD,
    Adam
};

struct Hyperparameters
{
    size_t batch_size = 64;
    OptimizerType optimizer_type = OptimizerType::Adam;
    scalar_t learning_rate = 0.001f;
    scalar_t beta1 = 0.9f;
    scalar_t beta2 = 0.999f;
    scalar_t epsilon = 1e-8f;
};

class CNN
{
private:
    Model model_;
    std::mt19937 gen_;
    static constexpr const char *MAGIC = "mnist-cnn";
    Hyperparameters hp_;
    Shape3D input_shape_;
    std::vector<nlohmann::json> history_;

    void build_from_json(const nlohmann::json &config_data);

public:
    CNN(unsigned int seed = 42);
    ~CNN() = default;

    void load_from_json(const std::string &config_path);
    void load_from_model(const std::string &path);

    std::vector<scalar_t> predict(const Matrix &image);
    int predict_label(const Matrix &image);

    void train(MnistDataset &dataset, size_t epochs);
    scalar_t accuracy(const MnistDataset &dataset);
    void save(const std::string &path) const;

    void print_architecture() const;
    const Hyperparameters &hyperparameters() const
    {
        return hp_;
    }
    void set_hyperparameters(const Hyperparameters &hp);
};