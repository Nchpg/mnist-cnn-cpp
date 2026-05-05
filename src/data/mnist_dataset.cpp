#include "data/mnist_dataset.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <omp.h>
#include <random>
#include <sstream>
#include <stdexcept>

#include "data/constants.hpp"

scalar_t MnistDataset::mean_ = 0.0f;
scalar_t MnistDataset::std_ = 1.0f;
bool MnistDataset::normalize_ = false;

void MnistDataset::compute_normalization(
    const std::vector<scalar_t> &images_data)
{
    if (images_data.empty())
        return;

    size_t total_pixels = images_data.size();
    scalar_t sum = 0.0f;

#pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < total_pixels; ++i)
    {
        sum += images_data[i];
    }
    mean_ = sum / static_cast<scalar_t>(total_pixels);

    scalar_t var_sum = 0.0f;
#pragma omp parallel for reduction(+ : var_sum)
    for (size_t i = 0; i < total_pixels; ++i)
    {
        scalar_t diff = images_data[i] - mean_;
        var_sum += diff * diff;
    }
    std_ = std::sqrt(var_sum / static_cast<scalar_t>(total_pixels));
    normalize_ = true;
}

void MnistDataset::apply_normalization()
{
    if (!normalize_)
        return;

    size_t total_pixels = all_images_data_.size();
#pragma omp parallel for
    for (size_t i = 0; i < total_pixels; ++i)
    {
        all_images_data_[i] = (all_images_data_[i] - mean_) / std_;
    }
}

MnistDataset MnistDataset::load(const std::string &path, size_t limit)
{
    if (path.length() >= 4 && path.substr(path.length() - 4) == ".bin")
    {
        return load_bin(path, limit);
    }
    return load_csv(path, limit);
}

MnistDataset MnistDataset::load_bin(const std::string &path, size_t limit)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Failed to open binary file: " + path);

    MnistDataset dataset;
    size_t capacity = limit > 0 ? limit : 10000;
    dataset.all_images_data_.reserve(capacity * PIXELS);
    dataset.labels_.reserve(capacity);
    dataset.indices_.reserve(capacity);

    unsigned char label;
    std::vector<unsigned char> pixels(PIXELS);

    while (limit == 0 || dataset.count_ < limit)
    {
        if (!file.read(reinterpret_cast<char *>(&label), 1))
            break;
        if (!file.read(reinterpret_cast<char *>(pixels.data()), PIXELS))
        {
            throw std::runtime_error("Incomplete record in binary file: "
                                     + path);
        }

        for (size_t i = 0; i < PIXELS; ++i)
        {
            scalar_t val = static_cast<scalar_t>(pixels[i]) / NORMALIZE_DIVISOR;
            dataset.all_images_data_.push_back(val);
        }
        dataset.labels_.push_back(label);
        dataset.indices_.push_back(dataset.count_);
        dataset.count_++;
    }

    if (dataset.count_ == 0)
        throw std::runtime_error("Binary file is empty or invalid: " + path);
    dataset.gen_ = std::mt19937(42);
    dataset.shuffle_indices();
    return dataset;
}

MnistDataset MnistDataset::load_csv(const std::string &path, size_t limit)
{
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + path);

    MnistDataset dataset;
    size_t capacity = limit > 0 ? limit : 10000;
    dataset.all_images_data_.reserve(capacity * PIXELS);
    dataset.labels_.reserve(capacity);
    dataset.indices_.reserve(capacity);

    std::string line;
    if (!std::getline(file, line))
        throw std::runtime_error("File is empty: " + path);

    auto parse_line = [&](const std::string &l) {
        std::stringstream ss(l);
        std::string token;
        if (!std::getline(ss, token, ','))
            return;

        dataset.labels_.push_back(static_cast<unsigned char>(std::stoi(token)));
        for (size_t i = 0; i < PIXELS; ++i)
        {
            if (!std::getline(ss, token, ','))
                throw std::runtime_error("Invalid CSV row");
            dataset.all_images_data_.push_back(
                static_cast<scalar_t>(std::stoi(token)) / NORMALIZE_DIVISOR);
        }
        dataset.indices_.push_back(dataset.count_);
        dataset.count_++;
    };

    if (!line_looks_like_header(line))
        parse_line(line);

    while ((limit == 0 || dataset.count_ < limit) && std::getline(file, line))
    {
        if (line.length() <= 1)
            continue;
        parse_line(line);
    }

    if (dataset.count_ == 0)
        throw std::runtime_error("File contains no MNIST rows: " + path);
    dataset.gen_ = std::mt19937(42);
    dataset.shuffle_indices();
    return dataset;
}

void MnistDataset::shuffle_indices()
{
    std::shuffle(indices_.begin(), indices_.end(), gen_);
}

void MnistDataset::get_batch_images(size_t start_idx, size_t batch_size,
                                    Matrix &out_batch) const
{
    if (start_idx + batch_size > count_)
        batch_size = count_ - start_idx;
    if (out_batch.rows() != PIXELS || out_batch.cols() != batch_size)
    {
        out_batch.reshape(PIXELS, batch_size);
    }

#pragma omp parallel for
    for (size_t i = 0; i < PIXELS; ++i)
    {
        for (size_t b = 0; b < batch_size; ++b)
        {
            size_t img_idx = indices_[start_idx + b];
            out_batch(i, b) = all_images_data_[img_idx * PIXELS + i];
        }
    }
}

void MnistDataset::get_batch_labels(size_t start_idx, size_t batch_size,
                                    std::vector<size_t> &out_labels) const
{
    if (start_idx + batch_size > count_)
        batch_size = count_ - start_idx;
    out_labels.resize(batch_size);
    for (size_t b = 0; b < batch_size; ++b)
    {
        out_labels[b] = static_cast<size_t>(labels_[indices_[start_idx + b]]);
    }
}
