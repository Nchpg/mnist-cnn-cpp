#pragma once

#include <stdexcept>

#include "data/dataset.hpp"
#include "utils/tensor.hpp"

class MnistDataset : public Dataset
{
private:
    size_t count_ = 0;
    std::vector<scalar_t> all_images_data_;
    std::vector<unsigned char> labels_;
    std::vector<size_t> indices_;
    std::mt19937 gen_;

    scalar_t mean_ = 0.0f;
    scalar_t std_ = 1.0f;
    bool normalize_ = false;

    static bool line_looks_like_header(const std::string& line)
    {
        size_t pos = 0;
        while (pos < line.length() && std::isspace(line[pos]))
        {
            pos++;
        }
        return pos >= line.length() || !std::isdigit(line[pos]);
    }

public:
    static constexpr size_t IMG_WIDTH = 28;
    static constexpr size_t IMG_HEIGHT = 28;
    static constexpr size_t PIXELS = IMG_WIDTH * IMG_HEIGHT;
    static constexpr size_t NUM_CLASSES = 10;
    static constexpr float NORMALIZE_DIVISOR = 255.0f;

    void compute_normalization(const std::vector<scalar_t>& images_data);
    void apply_normalization();
    void set_normalization(scalar_t mean, scalar_t std, bool normalize)
    {
        mean_ = mean;
        std_ = std;
        normalize_ = normalize;
    }

    const std::vector<scalar_t>& images_data() const
    {
        return all_images_data_;
    }

    scalar_t mean() const
    {
        return mean_;
    }
    scalar_t std() const
    {
        return std_;
    }
    bool normalize() const
    {
        return normalize_;
    }

public:
    MnistDataset() = default;

    static MnistDataset load(const std::string& path, size_t limit = 0);
    static MnistDataset load_csv(const std::string& path, size_t limit = 0);
    static MnistDataset load_bin(const std::string& path, size_t limit = 0);

    size_t count() const override
    {
        return count_;
    }
    Tensor image(size_t index) const
    {
        if (index >= count_)
            throw std::out_of_range("Image index out of range");
        Tensor img(Shape({ PIXELS, 1 }), 0.0f);
        std::copy(all_images_data_.begin() + index * PIXELS, all_images_data_.begin() + (index + 1) * PIXELS,
                  img.data_ptr());
        return img;
    }
    unsigned char label(size_t index) const
    {
        if (index >= count_)
            throw std::out_of_range("Label index out of range");
        return labels_[index];
    }

    void shuffle_indices();

    void get_batch_images(size_t start_idx, size_t batch_size, Tensor& out_batch) const;
    void get_batch_labels(size_t start_idx, size_t batch_size, std::vector<size_t>& out_labels) const;
};