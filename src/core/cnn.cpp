#include "core/cnn.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>

#include "data/mnist_dataset.hpp"
#include "layers/activation/activation.hpp"
#include "layers/activation/relu_layer.hpp"
#include "layers/activation/sigmoid_layer.hpp"
#include "layers/activation/softmax_layer.hpp"
#include "layers/batchnorm_layer.hpp"
#include "layers/conv_layer.hpp"
#include "layers/dense_layer.hpp"
#include "layers/dropout_layer.hpp"
#include "layers/flatten_layer.hpp"
#include "layers/pooling_layer.hpp"
#include "loss/cross_entropy_loss.hpp"
#include "loss/loss.hpp"
#include "loss/mse_loss.hpp"
#include "optimizers/adam_optimizer.hpp"
#include "optimizers/optimizer.hpp"
#include "optimizers/sgd_optimizer.hpp"
#include "utils/utils.hpp"

using json = nlohmann::json;

CNN::CNN(unsigned int seed)
    : model_()
    , gen_(seed)
{}

void CNN::load_from_json(const std::string &config_path)
{
    std::ifstream f(config_path);
    if (!f.is_open())
        throw std::runtime_error("Could not open config file: " + config_path);
    json config_data = json::parse(f);
    build_from_json(config_data);
}

void CNN::build_from_json(const nlohmann::json &config_data)
{
    input_shape_ = { config_data["input_shape"]["channels"],
                     config_data["input_shape"]["height"],
                     config_data["input_shape"]["width"] };

    if (config_data.contains("hyperparameters"))
    {
        const auto &hp = config_data["hyperparameters"];
        hp_.batch_size = hp.value("batch_size", 64);

        std::string loss_str = hp.value("loss", "CrossEntropy");
        if (loss_str == "MSE")
        {
            hp_.loss_type = LossType::MSE;
        }
        else
        {
            hp_.loss_type = LossType::CrossEntropy;
        }

        if (hp.contains("optimizer"))
        {
            const auto &opt = hp["optimizer"];
            std::string opt_type = opt.value("type", "Adam");
            hp_.optimizer_type =
                (opt_type == "SGD") ? OptimizerType::SGD : OptimizerType::Adam;
            hp_.learning_rate = opt.value("learning_rate", 0.001f);
            hp_.beta1 = opt.value("beta1", 0.9f);
            hp_.beta2 = opt.value("beta2", 0.999f);
            hp_.epsilon = opt.value("epsilon", 1e-8f);
        }
    }

    if (config_data.contains("history"))
    {
        history_ = config_data["history"].get<std::vector<json>>();
    }

    Shape3D current_shape = input_shape_;
    model_.layers().clear();

    for (const auto &layer_def : config_data["layers"])
    {
        std::string type = layer_def["type"];

        if (type == "Conv")
        {
            size_t filters = layer_def["filters"];
            size_t kernel_size = layer_def["kernel_size"];
            auto layer = std::make_unique<ConvLayer>(
                current_shape.height, current_shape.width,
                current_shape.channels, kernel_size, filters, gen_);
            current_shape = layer->get_output_shape(current_shape);
            model_.add(std::move(layer));
        }
        else if (type == "ReLU")
        {
            auto layer = std::make_unique<ReluLayer>();
            current_shape = layer->get_output_shape(current_shape);
            model_.add(std::move(layer));
        }
        else if (type == "Sigmoid")
        {
            auto layer = std::make_unique<SigmoidLayer>();
            current_shape = layer->get_output_shape(current_shape);
            model_.add(std::move(layer));
        }
        else if (type == "Pool")
        {
            size_t pool_size = layer_def["pool_size"];
            size_t stride = layer_def["stride"];
            auto layer = std::make_unique<PoolingLayer>(
                current_shape.height, current_shape.width,
                current_shape.channels, pool_size, stride);
            current_shape = layer->get_output_shape(current_shape);
            model_.add(std::move(layer));
        }
        else if (type == "Flatten")
        {
            auto layer = std::make_unique<FlattenLayer>(current_shape.channels,
                                                        current_shape.height,
                                                        current_shape.width);
            current_shape = layer->get_output_shape(current_shape);
            model_.add(std::move(layer));
        }
        else if (type == "Dense")
        {
            size_t units = layer_def["units"];
            auto layer =
                std::make_unique<DenseLayer>(current_shape.size(), units, gen_);
            current_shape = layer->get_output_shape(current_shape);
            model_.add(std::move(layer));
        }
        else if (type == "Dropout")
        {
            scalar_t ratio = layer_def["ratio"];
            auto layer = std::make_unique<DropoutLayer>(ratio, gen_);
            current_shape = layer->get_output_shape(current_shape);
            model_.add(std::move(layer));
        }
        else if (type == "BatchNorm")
        {
            auto layer = std::make_unique<BatchNormLayer>(
                current_shape.channels,
                current_shape.height * current_shape.width);
            current_shape = layer->get_output_shape(current_shape);
            model_.add(std::move(layer));
        }
        else if (type == "Softmax")
        {
            auto layer = std::make_unique<SoftmaxLayer>();
            current_shape = layer->get_output_shape(current_shape);
            model_.add(std::move(layer));
        }
    }
}

std::vector<scalar_t> CNN::predict(const Tensor &image) const
{
    Tensor img_batch = image;
    img_batch.reshape(Shape(
        { 1, input_shape_.channels, input_shape_.height, input_shape_.width }));

    thread_local std::vector<std::unique_ptr<LayerContext>> contexts;
    const Tensor &probs = model_.forward(img_batch, contexts, false);

    std::vector<scalar_t> result(probs.shape()[1]);
    for (size_t i = 0; i < probs.shape()[1]; i++)
        result[i] = probs(0, i);
    return result;
}

int CNN::predict_label(const Tensor &image) const
{
    Tensor img_batch = image;
    img_batch.reshape(Shape(
        { 1, input_shape_.channels, input_shape_.height, input_shape_.width }));

    thread_local std::vector<std::unique_ptr<LayerContext>> contexts;
    const Tensor &probs = model_.forward(img_batch, contexts, false);
    auto results = Utils::argmax(probs);
    return static_cast<int>(results[0]);
}

void CNN::train(Dataset &dataset, size_t epochs)
{
    if (!model_.optimizer())
    {
        set_hyperparameters(hp_);
    }

    if (model_.optimizer())
    {
        model_.optimizer()->reset();
        model_.optimizer()->set_learning_rate(hp_.learning_rate);
    }
    const size_t batch_size = hp_.batch_size;
    const size_t total_samples = dataset.count();
    const int bar_width = 30;

    Tensor batch_images;
    std::vector<size_t> batch_labels;
    Tensor grad_output;

    std::unique_ptr<Loss> loss_fn;
    if (hp_.loss_type == LossType::CrossEntropy)
    {
        loss_fn = std::make_unique<CrossEntropyLoss>();
    }
    else if (hp_.loss_type == LossType::MSE)
    {
        loss_fn = std::make_unique<MSELoss>();
    }
    else
    {
        throw std::runtime_error("Unsupported loss type");
    }

    scalar_t final_loss = 0.0f;
    scalar_t final_acc = 0.0f;

    bool use_fused_optimization = false;
    std::string last_layer_type = "";
    if (!model_.layers().empty())
    {
        last_layer_type =
            model_.layers().back()->get_config().value("type", "");
        use_fused_optimization = loss_fn->supports_fusion_with(last_layer_type);
    }

    size_t samples_to_process = (total_samples / batch_size) * batch_size;

    std::vector<std::unique_ptr<LayerContext>> contexts;

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        dataset.shuffle_indices();
        size_t correct = 0;
        scalar_t total_loss = 0.0f;

        for (size_t sample = 0; sample < samples_to_process;
             sample += batch_size)
        {
            dataset.get_batch_images(sample, batch_size, batch_images);
            dataset.get_batch_labels(sample, batch_size, batch_labels);

            batch_images.reshape(
                Shape({ batch_size, input_shape_.channels, input_shape_.height,
                        input_shape_.width }));

            const Tensor &output = model_.forward(batch_images, contexts, true);

            total_loss += loss_fn->forward(output, batch_labels)
                * static_cast<scalar_t>(batch_size);

            auto predictions = Utils::argmax(output);
            for (size_t b = 0; b < batch_size; ++b)
            {
                if (predictions[b] == batch_labels[b])
                    correct++;
            }

            if (use_fused_optimization)
            {
                loss_fn->backward_fused(last_layer_type, output, batch_labels,
                                        grad_output);
                model_.backward_skip_last(grad_output, contexts, true);
            }
            else
            {
                loss_fn->backward(output, batch_labels, grad_output);
                model_.backward(grad_output, contexts, true);
            }

            model_.step();
            model_.clear_gradients();

            scalar_t progress = static_cast<scalar_t>(sample + batch_size)
                / static_cast<scalar_t>(samples_to_process);
            std::cout << "\rEpoch " << epoch + 1 << "/" << epochs << " [";
            int pos =
                static_cast<int>(static_cast<scalar_t>(bar_width) * progress);
            for (int i = 0; i < bar_width; ++i)
            {
                if (i < pos)
                    std::cout << "=";
                else if (i == pos)
                    std::cout << ">";
                else
                    std::cout << " ";
            }
            std::cout << "] " << static_cast<int>(progress * 100.0f) << "% "
                      << (sample + batch_size) << "/" << samples_to_process
                      << " " << std::flush;
        }

        final_loss = total_loss / static_cast<scalar_t>(samples_to_process);
        final_acc = 100.0f * static_cast<scalar_t>(correct)
            / static_cast<scalar_t>(samples_to_process);
        std::cout << "\n  -> loss: " << final_loss
                  << " - accuracy: " << final_acc << "%" << std::endl;
    }

    json session;
    session["samples"] = total_samples;
    session["epochs"] = epochs;
    session["final_loss"] = final_loss;
    session["final_accuracy"] = final_acc;
    session["hyperparameters"] = {
        { "batch_size", hp_.batch_size },
        { "loss",
          (hp_.loss_type == LossType::CrossEntropy) ? "CrossEntropy" : "MSE" },
        { "optimizer",
          { { "type",
              (hp_.optimizer_type == OptimizerType::SGD) ? "SGD" : "Adam" },
            { "learning_rate", hp_.learning_rate },
            { "beta1", hp_.beta1 },
            { "beta2", hp_.beta2 },
            { "epsilon", hp_.epsilon } } }
    };
    history_.push_back(session);
}

scalar_t CNN::accuracy(const Dataset &dataset)
{
    size_t correct = 0;
    const size_t eval_batch_size = 64;

    Tensor batch_images;
    std::vector<size_t> batch_labels;
    std::vector<std::unique_ptr<LayerContext>> contexts;

    for (size_t i = 0; i < dataset.count(); i += eval_batch_size)
    {
        size_t actual_batch_size =
            std::min(eval_batch_size, dataset.count() - i);
        dataset.get_batch_images(i, actual_batch_size, batch_images);
        dataset.get_batch_labels(i, actual_batch_size, batch_labels);

        batch_images.reshape(
            Shape({ actual_batch_size, input_shape_.channels,
                    input_shape_.height, input_shape_.width }));

        const Tensor &logits = model_.forward(batch_images, contexts, false);
        auto predictions = Utils::argmax(logits);

        for (size_t b = 0; b < actual_batch_size; ++b)
        {
            if (predictions[b] == batch_labels[b])
                correct++;
        }
    }
    return static_cast<scalar_t>(correct)
        / static_cast<scalar_t>(dataset.count());
}

void CNN::save(const std::string &path) const
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open())
        throw std::runtime_error("Save failed");

    json header;
    header["input_shape"] = { { "channels", input_shape_.channels },
                              { "height", input_shape_.height },
                              { "width", input_shape_.width } };

    header["hyperparameters"] = {
        { "batch_size", hp_.batch_size },
        { "loss",
          (hp_.loss_type == LossType::CrossEntropy) ? "CrossEntropy" : "MSE" },
        { "optimizer",
          { { "type",
              (hp_.optimizer_type == OptimizerType::SGD) ? "SGD" : "Adam" },
            { "learning_rate", hp_.learning_rate },
            { "beta1", hp_.beta1 },
            { "beta2", hp_.beta2 },
            { "epsilon", hp_.epsilon } } }
    };

    json layers_configs = json::array();
    for (const auto &layer : model_.layers())
    {
        layers_configs.push_back(layer->get_config());
    }
    header["layers"] = layers_configs;
    header["history"] = history_;

    std::string header_str = header.dump();
    size_t header_len = header_str.length();

    file.write(MAGIC, 4);
    file.write(reinterpret_cast<const char *>(&header_len), sizeof(header_len));
    file.write(header_str.c_str(), header_len);

    file.write(reinterpret_cast<const char *>(&data_mean_), sizeof(scalar_t));
    file.write(reinterpret_cast<const char *>(&data_std_), sizeof(scalar_t));
    file.write(reinterpret_cast<const char *>(&normalize_input_), sizeof(bool));

    for (const auto &layer : model_.layers())
        layer->save(file);
    file.close();
}

void CNN::load_from_model(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Load failed: " + path);

    char magic[4];
    file.read(magic, 4);
    if (std::strncmp(magic, MAGIC, 4) != 0)
    {
        throw std::runtime_error("Unsupported model format or bad magic");
    }

    size_t header_len;
    file.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));
    std::string header_str(header_len, '\0');
    file.read(&header_str[0], header_len);

    json header = json::parse(header_str);
    build_from_json(header);

    file.read(reinterpret_cast<char *>(&data_mean_), sizeof(scalar_t));
    file.read(reinterpret_cast<char *>(&data_std_), sizeof(scalar_t));
    file.read(reinterpret_cast<char *>(&normalize_input_), sizeof(bool));

    for (auto &layer : model_.layers())
        layer->load(file);
    file.close();
}

void CNN::print_architecture() const
{
    std::cout << "CNN Architecture:\n";
    std::cout << "  Input Shape: " << input_shape_.channels << "x"
              << input_shape_.height << "x" << input_shape_.width << "\n";
    std::cout << "  Hyperparameters:\n";
    std::cout << "    Batch Size: " << hp_.batch_size << "\n";
    std::cout << "    Optimizer:\n";
    std::cout << "      Type: "
              << (hp_.optimizer_type == OptimizerType::SGD ? "SGD" : "Adam")
              << "\n";
    std::cout << "      Learning Rate: " << hp_.learning_rate << "\n";
    if (hp_.optimizer_type == OptimizerType::Adam)
    {
        std::cout << "      Adam Beta1: " << hp_.beta1 << "\n";
        std::cout << "      Adam Beta2: " << hp_.beta2 << "\n";
        std::cout << "      Adam Epsilon: " << hp_.epsilon << "\n";
    }
    std::cout << "  Layers:\n";
    for (size_t i = 0; i < model_.layers().size(); ++i)
    {
        std::cout << "    " << i + 1 << ". "
                  << model_.layers()[i]->get_config().dump() << "\n";
    }

    if (!history_.empty())
    {
        std::cout << "  Training History:\n";
        for (size_t i = 0; i < history_.size(); ++i)
        {
            const auto &h = history_[i];
            auto get_f = [](const json &j, const char *k) -> std::string {
                if (j.contains(k) && j[k].is_number())
                {
                    std::ostringstream oss;
                    oss << j[k].get<float>();
                    return oss.str();
                }
                return "nan";
            };

            std::cout << "    Session " << i + 1 << ": "
                      << h.value("samples", 0UL) << " samples, "
                      << h.value("epochs", 0UL) << " epochs, "
                      << "loss: " << get_f(h, "final_loss") << ", "
                      << "acc: " << get_f(h, "final_accuracy") << "%\n";

            const auto &hp = h["hyperparameters"];
            const auto &opt = hp["optimizer"];

            std::cout << "      Batch Size: " << hp.value("batch_size", 0UL)
                      << "\n"
                      << "      Optimizer: " << opt.value("type", "Unknown")
                      << " (lr=" << get_f(opt, "learning_rate");

            if (opt.value("type", "Adam") == "Adam")
            {
                std::cout << ", beta1=" << get_f(opt, "beta1")
                          << ", beta2=" << get_f(opt, "beta2")
                          << ", eps=" << get_f(opt, "epsilon");
            }
            std::cout << ")\n";
        }
    }
}

void CNN::set_hyperparameters(const Hyperparameters &hp)
{
    hp_ = hp;
    if (hp_.optimizer_type == OptimizerType::SGD)
    {
        model_.set_optimizer(std::make_unique<SGDOptimizer>(hp_.learning_rate));
    }
    else
    {
        model_.set_optimizer(std::make_unique<AdamOptimizer>(
            hp_.learning_rate, hp_.beta1, hp_.beta2, hp_.epsilon));
    }
}
