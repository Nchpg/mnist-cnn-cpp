#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string_view>

#include "core/cnn.hpp"
#include "data/mnist_dataset.hpp"
#include "layers/activation/activation.hpp"

static void usage(const char *program)
{
    std::cerr << "usage:\n"
              << "  " << program << " info [model_in or architecture.json]\n"
              << "  " << program
              << " train <mnist.csv> <arch.json> <model_out> [limit=... "
                 "epochs=... lr=...]\n"
              << "  " << program
              << " resume <mnist.csv> <model_in> <model_out> [limit=... "
                 "epochs=... lr=...]\n"
              << "  " << program << " test <mnist.csv> <model_in> [limit=...]\n"
              << "  " << program
              << " predict <model_in> <mnist.csv> [index=...]\n";
}

static void parse_hp_overrides(int start_idx, int argc, char **argv,
                               Hyperparameters &hp, size_t &epochs,
                               size_t &limit)
{
    for (int i = start_idx; i < argc; ++i)
    {
        std::string arg = argv[i];
        size_t pos = arg.find('=');
        if (pos == std::string::npos)
            continue;
        std::string key = arg.substr(0, pos);
        std::string value = arg.substr(pos + 1);

        if (key == "lr")
            hp.learning_rate = std::stof(value);
        else if (key == "batch_size")
            hp.batch_size = std::stoul(value);
        else if (key == "beta1")
            hp.beta1 = std::stof(value);
        else if (key == "beta2")
            hp.beta2 = std::stof(value);
        else if (key == "epsilon")
            hp.epsilon = std::stof(value);
        else if (key == "optimizer")
            hp.optimizer_type =
                (value == "SGD" ? OptimizerType::SGD : OptimizerType::Adam);
        else if (key == "epochs")
            epochs = std::stoul(value);
        else if (key == "limit")
            limit = std::stoul(value);
    }
}

static void print_training_params(const Hyperparameters &hp, size_t epochs,
                                  size_t limit)
{
    std::cout << "Effective Training Parameters:\n";
    std::cout << "  Dataset Limit: "
              << (limit == 0 ? "All" : std::to_string(limit)) << "\n";
    std::cout << "  Epochs: " << epochs << "\n";
    std::cout << "  Batch Size: " << hp.batch_size << "\n";
    std::cout << "  Optimizer: "
              << (hp.optimizer_type == OptimizerType::SGD ? "SGD" : "Adam")
              << "\n";
    std::cout << "  Learning Rate: " << hp.learning_rate << "\n";
    std::cout << "  Loss: "
              << (hp.loss_type == LossType::CrossEntropy ? "CrossEntropy"
                                                         : "MSE")
              << "\n";
    if (hp.optimizer_type == OptimizerType::Adam)
    {
        std::cout << "  Adam Params: beta1=" << hp.beta1
                  << ", beta2=" << hp.beta2 << ", eps=" << hp.epsilon << "\n";
    }
    std::cout << "-------------------------------------------" << std::endl;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::string_view command(argv[1]);

    if (command == "info")
    {
        std::string path = argc > 2 ? argv[2] : "architecture.json";
        CNN cnn;
        try
        {
            if (path.find('.') != std::string::npos
                && path.substr(path.find_last_of(".") + 1) == "json")
            {
                cnn.load_from_json(path);
            }
            else
            {
                cnn.load_from_model(path);
            }
            cnn.print_architecture();
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    if (command == "train")
    {
        if (argc < 5)
        {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
        std::string csv_path = argv[2];
        std::string arch_json = argv[3];
        std::string model_out = argv[4];

        CNN cnn;
        cnn.load_from_json(arch_json);

        Hyperparameters hp = cnn.hyperparameters();
        size_t epochs = 3;
        size_t limit = 0;
        parse_hp_overrides(5, argc, argv, hp, epochs, limit);
        cnn.set_hyperparameters(hp);

        MnistDataset train = MnistDataset::load(csv_path, limit);
        MnistDataset::compute_normalization(train.images_data());
        train.apply_normalization();

        std::cout << "Starting training from architecture: " << arch_json
                  << " using " << csv_path << std::endl;
        print_training_params(cnn.hyperparameters(), epochs, limit);
        cnn.train(train, epochs);
        cnn.save(model_out);
        std::cout << "Model saved to: " << model_out << std::endl;
    }
    else if (command == "resume")
    {
        if (argc < 5)
        {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
        std::string csv_path = argv[2];
        std::string model_in = argv[3];
        std::string model_out = argv[4];

        CNN cnn;
        cnn.load_from_model(model_in);

        Hyperparameters hp = cnn.hyperparameters();
        size_t epochs = 3;
        size_t limit = 0;
        parse_hp_overrides(5, argc, argv, hp, epochs, limit);
        cnn.set_hyperparameters(hp);

        MnistDataset train = MnistDataset::load(csv_path, limit);
        train.apply_normalization();

        std::cout << "Resuming training from model: " << model_in << " using "
                  << csv_path << std::endl;
        print_training_params(cnn.hyperparameters(), epochs, limit);
        cnn.train(train, epochs);
        cnn.save(model_out);
        std::cout << "Model saved to: " << model_out << std::endl;
    }
    else if (command == "test")
    {
        if (argc < 4)
        {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
        std::string csv_path = argv[2];
        std::string model_in = argv[3];

        size_t limit = 0;
        for (int i = 4; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg.substr(0, 6) == "limit=")
                limit = std::stoul(arg.substr(6));
        }

        CNN cnn;
        cnn.load_from_model(model_in);
        MnistDataset test = MnistDataset::load(csv_path, limit);
        test.apply_normalization();

        std::cout << "accuracy " << 100.0f * cnn.accuracy(test) << "% on "
                  << test.count() << " images\n";
    }
    else if (command == "predict")
    {
        if (argc < 4)
        {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
        std::string model_in = argv[2];
        std::string csv_path = argv[3];

        size_t index = 0;
        for (int i = 4; i < argc; ++i)
        {
            std::string arg = argv[i];
            if (arg.substr(0, 6) == "index=")
                index = std::stoul(arg.substr(6));
        }

        CNN cnn;
        cnn.load_from_model(model_in);
        MnistDataset dataset = MnistDataset::load(csv_path, index + 1);
        Tensor image = dataset.image(index);

        std::vector<scalar_t> probabilities = cnn.predict(image);
        int prediction = cnn.predict_label(image);

        std::cout << "prediction: " << prediction << "\n";
        for (size_t digit = 0; digit < probabilities.size(); digit++)
        {
            std::cout << digit << ": " << probabilities[digit] << "\n";
        }
    }
    else
    {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
