#pragma once

#include <array>
#include <random>
#include <string>
#include <vector>

#include "core/hyperparameters.hpp"
#include "core/model.hpp"
#include "data/dataset.hpp"
#include "utils/tensor.hpp"

class CNN
{
private:
    Model model_;
    std::mt19937 gen_;
    static constexpr const char* MAGIC = "mnist-cnn";
    Hyperparameters hp_;
    Shape3D input_shape_;
    std::vector<nlohmann::json> history_;

    scalar_t data_mean_ = 0.0f;
    scalar_t data_std_ = 1.0f;
    bool normalize_input_ = false;

    void build_from_json(const nlohmann::json& config_data);

public:
    CNN(unsigned int seed = 42);
    ~CNN() = default;

    void load_from_json(const std::string& config_path);
    void load_from_model(const std::string& path);

    std::vector<scalar_t> predict(const Tensor& image) const;
    int predict_label(const Tensor& image) const;

    void train(Dataset& dataset, size_t epochs);
    scalar_t accuracy(const Dataset& dataset);
    void save(const std::string& path) const;

    void print_architecture() const;
    const Hyperparameters& hyperparameters() const
    {
        return hp_;
    }
    void set_hyperparameters(const Hyperparameters& hp);

    scalar_t mean() const
    {
        return data_mean_;
    }
    scalar_t std() const
    {
        return data_std_;
    }
    bool normalize() const
    {
        return normalize_input_;
    }
    void set_normalization(scalar_t m, scalar_t s, bool norm)
    {
        data_mean_ = m;
        data_std_ = s;
        normalize_input_ = norm;
    }
};