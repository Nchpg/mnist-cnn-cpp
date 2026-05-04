#include "mnist_dataset.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <cmath>
#include <omp.h>

scalar_t MnistDataset::mean_ = 0.0f;
scalar_t MnistDataset::std_ = 1.0f;
bool MnistDataset::normalize_ = false;

void MnistDataset::compute_normalization(const std::vector<Matrix>& images) {
    if (images.empty()) return;

    size_t total = images.size();
    scalar_t sum = 0.0f;

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < total; ++i) {
        const Matrix& img = images[i];
        for (size_t j = 0; j < PIXELS; ++j) {
            sum += img(j, 0);
        }
    }
    mean_ = sum / static_cast<scalar_t>(total * PIXELS);

    scalar_t var_sum = 0.0f;
    #pragma omp parallel for reduction(+:var_sum)
    for (size_t i = 0; i < total; ++i) {
        const Matrix& img = images[i];
        for (size_t j = 0; j < PIXELS; ++j) {
            scalar_t diff = img(j, 0) - mean_;
            var_sum += diff * diff;
        }
    }
    std_ = std::sqrt(var_sum / static_cast<scalar_t>(total * PIXELS));
    normalize_ = true;
}

void MnistDataset::apply_normalization() {
    if (!normalize_) return;

    #pragma omp parallel for
    for (size_t i = 0; i < count_; ++i) {
        for (size_t j = 0; j < PIXELS; ++j) {
            images_[i](j, 0) = (images_[i](j, 0) - mean_) / std_;
        }
    }
}

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
            scalar_t raw = static_cast<scalar_t>(pixels[i]) / 255.0f;
            image(i, 0) = normalize_ ? (raw - mean_) / std_ : raw;
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
        scalar_t raw = static_cast<scalar_t>(pixel) / 255.0f;
        image(i, 0) = normalize_ ? (raw - mean_) / std_ : raw;
    }
    
    return image;
}

void MnistDataset::shuffle_indices() {
    std::shuffle(indices_.begin(), indices_.end(), gen_);
}


void MnistDataset::get_batch_images(size_t start_idx, size_t batch_size, Matrix& out_batch) const {
    if (start_idx + batch_size > count_) {
        batch_size = count_ - start_idx;
    }
    if (out_batch.rows() != PIXELS || out_batch.cols() != batch_size) {
        out_batch.reshape(PIXELS, batch_size);
    }
    for (size_t b = 0; b < batch_size; ++b) {
        const Matrix& img = images_[indices_[start_idx + b]];
        for (size_t i = 0; i < PIXELS; ++i) {
            out_batch(i, b) = img(i, 0);
        }
    }
}


void MnistDataset::get_batch_labels(size_t start_idx, size_t batch_size, std::vector<size_t>& out_labels) const {
    if (start_idx + batch_size > count_) {
        batch_size = count_ - start_idx;
    }
    out_labels.resize(batch_size);
    for (size_t b = 0; b < batch_size; ++b) {
        out_labels[b] = static_cast<size_t>(labels_[indices_[start_idx + b]]);
    }
}

