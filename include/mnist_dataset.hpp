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
    std::vector<Matrix> images_;
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

    static void compute_normalization(const std::vector<Matrix>& images);
    void apply_normalization();

    const std::vector<Matrix>& images() const { return images_; }

    static scalar_t mean() { return mean_; }
    static scalar_t std() { return std_; }
    static bool normalize() { return normalize_; }
    
public:
    MnistDataset() = default;

    static MnistDataset load(const std::string& path, size_t limit = 0);
    static MnistDataset load_csv(const std::string& path, size_t limit = 0);
    static MnistDataset load_bin(const std::string& path, size_t limit = 0);
    static Matrix parse_pixels(const std::string& line, unsigned char& label);
    
    size_t count() const { return count_; }
    const Matrix& image(size_t index) const { 
        if (index >= count_) throw std::out_of_range("Image index out of range");
        return images_[index];
    }
    unsigned char label(size_t index) const { 
        if (index >= count_) throw std::out_of_range("Label index out of range");
        return labels_[index];
    }
    
    void shuffle_indices();
    
    Matrix get_batch_images(size_t start_idx, size_t batch_size) const;
    std::vector<size_t> get_batch_labels(size_t start_idx, size_t batch_size) const;
    

};

#endif 