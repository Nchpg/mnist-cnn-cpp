#include "cnn.hpp"
#include "mnist_dataset.hpp"
#include "activation.hpp"
#include <iostream>
#include <cstdlib>
#include <string_view>
#include <array>
#include <fstream>

static void usage(const char *program) {
    std::cerr << "usage:\n"
              << "  " << program << " architecture\n"
              << "  " << program << " train <mnist_train.csv> <model_out> [limit] [epochs] [learning_rate]\n"
              << "  " << program << " test <mnist_test.csv> <model_in> [limit]\n"
              << "  " << program << " predict <model_in> '<label,pixel0,...,pixel783>'\n";
}

int main(int argc, char **argv) {
    const std::string config_path = "architecture.json";

    if (argc >= 2 && std::string_view(argv[1]) == "architecture") {
        std::ifstream f(config_path);
        if (f.is_open()) {
            std::cout << f.rdbuf() << std::endl;
        } else {
            std::cerr << "Could not open " << config_path << std::endl;
        }
        return EXIT_SUCCESS;
    }

    if (argc < 4) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    CNN cnn(config_path, 42);
    std::string_view command(argv[1]);

    if (command == "train") {
        size_t limit = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 0;
        size_t epochs = argc > 5 ? std::strtoul(argv[5], nullptr, 10) : 3;
        scalar_t learning_rate = argc > 6 ? static_cast<scalar_t>(std::strtod(argv[6], nullptr)) : 0.01f;

        MnistDataset train = MnistDataset::load(argv[2], limit);
        MnistDataset::compute_normalization(train.images());
        train.apply_normalization();
        cnn.train(train, epochs, learning_rate);
        cnn.save(argv[3]);
        
    } else if (command == "test") {
        size_t limit = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 0;
        cnn.load(argv[3]);
        MnistDataset test = MnistDataset::load(argv[2], limit);
        
        std::cout << "accuracy " << 100.0f * cnn.accuracy(test) << "% on " 
                  << test.count() << " images\n";
                  
    } else if (command == "predict") {
        cnn.load(argv[2]);
        MnistDataset dataset = MnistDataset::load(argv[3], 1);
        Matrix image = dataset.image(0);
        
        std::vector<scalar_t> probabilities = cnn.predict(image);
        int prediction = cnn.predict_label(image);

        std::cout << "prediction: " << prediction << "\n";
        for (size_t digit = 0; digit < probabilities.size(); digit++) {
            std::cout << digit << ": " << probabilities[digit] << "\n";
        }
    } else {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
