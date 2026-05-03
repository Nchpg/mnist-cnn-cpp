#include "mnist_dataset.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <cstdlib>
#include <algorithm>
#include <random>

MnistDataset MnistDataset::load(const std::string& path, size_t limit) {

    if (path.length() >= 4 && path.substr(path.length() - 4) == ".bin") {
        return load_bin(path, limit);
    }

    return load_csv(path, limit);
}


MnistDataset MnistDataset::load_bin(const std::string& path, size_t limit) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open binary file: " + path);
    }

    MnistDataset dataset;
    size_t capacity = limit > 0 ? limit : 1024;
    dataset.images_.reserve(capacity);
    dataset.labels_.reserve(capacity);
    dataset.indices_.reserve(capacity);

    unsigned char label;
    unsigned char pixels[PIXELS];


    while (limit == 0 || dataset.count_ < limit) {

        if (!file.read(reinterpret_cast<char*>(&label), 1)) {
            break; 
        }
        

        if (!file.read(reinterpret_cast<char*>(pixels), PIXELS)) {
            throw std::runtime_error("Incomplete record in binary file: " + path);
        }


        Matrix image(PIXELS, 1);
        for (size_t i = 0; i < PIXELS; ++i) {
            image(i, 0) = static_cast<scalar_t>(pixels[i]) / 255.0f;
        }

        dataset.images_.push_back(image);
        dataset.labels_.push_back(label);
        dataset.indices_.push_back(dataset.count_);
        dataset.count_++;
    }

    if (dataset.count_ == 0) {
        throw std::runtime_error("Binary file is empty or invalid: " + path);
    }

    dataset.gen_ = std::mt19937(42); 
    dataset.shuffle_indices();

    return dataset;
}

MnistDataset MnistDataset::load_csv(const std::string& path, size_t limit) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    MnistDataset dataset;
    size_t capacity = limit > 0 ? limit : 1024;
    dataset.images_.reserve(capacity);
    dataset.labels_.reserve(capacity);
    dataset.indices_.reserve(capacity);

    std::string line;

    if (!std::getline(file, line)) {
        throw std::runtime_error("File is empty: " + path);
    }


    if (!line_looks_like_header(line)) {
        unsigned char label;
        dataset.images_.push_back(parse_pixels(line, label));
        dataset.labels_.push_back(label);
        dataset.indices_.push_back(dataset.count_); 
        dataset.count_++;
    }


    while ((limit == 0 || dataset.count_ < limit) && std::getline(file, line)) {
        if (line.length() <= 1) continue; 
        
        unsigned char label;
        dataset.images_.push_back(parse_pixels(line, label));
        dataset.labels_.push_back(label);
        dataset.indices_.push_back(dataset.count_); 
        dataset.count_++;
    }

    if (dataset.count_ == 0) {
        throw std::runtime_error("File contains no MNIST rows: " + path);
    }

    dataset.gen_ = std::mt19937(42); 

    dataset.shuffle_indices();

    return dataset;
}

Matrix MnistDataset::parse_pixels(const std::string& line, unsigned char& label) {
    std::stringstream ss(line);
    std::string token;
    
    if (!std::getline(ss, token, ',')) {
        throw std::runtime_error("Empty MNIST CSV row: Failed to extract label.");
    }
    
    char* end = nullptr;
    long parsed_label = std::strtol(token.c_str(), &end, 10);
    if (end == token.c_str() || parsed_label < 0 || parsed_label >= static_cast<long>(CLASSES)) {
        throw std::runtime_error("Invalid MNIST label: " + token);
    }
    label = static_cast<unsigned char>(parsed_label);
    
    Matrix image(PIXELS, 1);
    for (size_t i = 0; i < PIXELS; ++i) {
        if (!std::getline(ss, token, ',')) {
            throw std::runtime_error("MNIST row has fewer than " + std::to_string(PIXELS) + " pixels. Expected " + std::to_string(PIXELS) + ", got " + std::to_string(i) + " pixels.");
        }
        
        long pixel = std::strtol(token.c_str(), &end, 10);
        if (end == token.c_str() || pixel < 0 || pixel > 255) {
            throw std::runtime_error("Invalid MNIST pixel value: " + token + " at pixel index " + std::to_string(i));
        }
        image(i, 0) = static_cast<scalar_t>(pixel) / 255.0f;
    }
    
    return image;
}

void MnistDataset::shuffle_indices() {
    std::shuffle(indices_.begin(), indices_.end(), gen_);
}


Matrix MnistDataset::get_batch_images(size_t start_idx, size_t batch_size) const {
    if (start_idx + batch_size > count_) {
        batch_size = count_ - start_idx; 
    }
    Matrix batch(PIXELS, batch_size);
    for (size_t b = 0; b < batch_size; ++b) {

        const Matrix& img = images_[indices_[start_idx + b]]; 
        for (size_t i = 0; i < PIXELS; ++i) {
            batch(i, b) = img(i, 0);
        }
    }
    return batch;
}


std::vector<size_t> MnistDataset::get_batch_labels(size_t start_idx, size_t batch_size) const {
    if (start_idx + batch_size > count_) {
        batch_size = count_ - start_idx; 
    }
    std::vector<size_t> batch(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {

        batch[b] = static_cast<size_t>(labels_[indices_[start_idx + b]]); 
    }
    return batch;
}

