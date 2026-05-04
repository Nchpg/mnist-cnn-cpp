#ifndef MNIST_DATASET_HPP
#define MNIST_DATASET_HPP

#include "matrix.hpp"
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstdlib>
#include <algorithm>
#include <random> 

class MnistDataset {
private:
    static constexpr size_t IMAGE_SIZE = 28;
    static constexpr size_t PIXELS = IMAGE_SIZE * IMAGE_SIZE;
    static constexpr size_t CLASSES = 10;

    size_t count_ = 0;
    std::vector<scalar_t> all_images_data_; // Single continuous buffer
    std::vector<unsigned char> labels_;
    std::vector<size_t> indices_;
    std::mt19937 gen_;

    static bool line_looks_like_header(const std::string& line) {
        size_t pos = 0;
        while (pos < line.length() && std::isspace(line[pos])) pos++;
        return pos >= line.length() || !std::isdigit(line[pos]);
    }

public:
    static scalar_t mean_;
    static scalar_t std_;
    static bool normalize_;

    static void compute_normalization(const std::vector<scalar_t>& images_data);
    void apply_normalization();

    const std::vector<scalar_t>& images_data() const { return all_images_data_; }

    static scalar_t mean() { return mean_; }
    static scalar_t std() { return std_; }
    static bool normalize() { return normalize_; }
    
public:
    MnistDataset() = default;

    static MnistDataset load(const std::string& path, size_t limit = 0);
    static MnistDataset load_csv(const std::string& path, size_t limit = 0);
    static MnistDataset load_bin(const std::string& path, size_t limit = 0);
    
    size_t count() const { return count_; }
    Matrix image(size_t index) const { 
        if (index >= count_) throw std::out_of_range("Image index out of range");
        Matrix img(PIXELS, 1);
        std::copy(all_images_data_.begin() + index * PIXELS, all_images_data_.begin() + (index + 1) * PIXELS, img.data());
        return img;
    }
    unsigned char label(size_t index) const { 
        if (index >= count_) throw std::out_of_range("Label index out of range");
        return labels_[index];
    }
    
    void shuffle_indices();

    void get_batch_images(size_t start_idx, size_t batch_size, Matrix& out_batch) const;
    void get_batch_labels(size_t start_idx, size_t batch_size, std::vector<size_t>& out_labels) const;
};

#endif 