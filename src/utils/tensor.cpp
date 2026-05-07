#include "utils/tensor.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>

void Tensor::compute_strides()
{
    strides_.resize(shape_.rank());
    size_t stride = 1;
    for (int i = static_cast<int>(shape_.rank()) - 1; i >= 0; --i)
    {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

Tensor::Tensor(Shape shape, scalar_t init_val)
    : shape_(shape)
    , data_(shape.size(), init_val)
{
    compute_strides();
}

Tensor::Tensor(std::initializer_list<size_t> dims, scalar_t init_val)
    : shape_(Shape(std::vector<size_t>(dims)))
    , data_(shape_.size(), init_val)
{
    compute_strides();
}

void Tensor::reshape(Shape new_shape)
{
    size_t new_size = new_shape.size();
    size_t old_size = data_.size();
    if (new_size != old_size)
    {
        data_.resize(new_size, 0.0f);
        if (new_size > old_size)
        {
            std::fill(data_.begin() + old_size, data_.end(), 0.0f);
        }
    }
    shape_ = new_shape;
    compute_strides();
}

void Tensor::reshape(Shape new_shape, scalar_t init_val)
{
    size_t new_size = new_shape.size();
    size_t old_size = data_.size();
    if (new_size != old_size)
    {
        data_.resize(new_size, init_val);
        if (new_size > old_size)
        {
            std::fill(data_.begin() + old_size, data_.end(), init_val);
        }
    }
    shape_ = new_shape;
    compute_strides();
}

void Tensor::fill(scalar_t value)
{
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::random_uniform(scalar_t scale, std::mt19937& gen)
{
    std::uniform_real_distribution<scalar_t> dist(-scale, scale);
    for (auto& val : data_)
        val = dist(gen);
}

void Tensor::random_normal(scalar_t std_dev, std::mt19937& gen)
{
    std::normal_distribution<scalar_t> dist(0.0f, std_dev);
    for (auto& val : data_)
        val = dist(gen);
}

Tensor& Tensor::operator+=(const Tensor& other)
{
    if (shape_ != other.shape_)
    {
        throw std::invalid_argument("Shapes mismatch in operator+=");
    }
    size_t n = data_.size();
#pragma omp parallel for if (n > 10000)
    for (size_t i = 0; i < n; ++i)
        data_[i] += other.data_[i];
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other)
{
    if (shape_ != other.shape_)
    {
        throw std::invalid_argument("Shapes mismatch in operator-=");
    }
    size_t n = data_.size();
#pragma omp parallel for if (n > 10000)
    for (size_t i = 0; i < n; ++i)
        data_[i] -= other.data_[i];
    return *this;
}

void Tensor::add_scaled(const Tensor& other, scalar_t scale)
{
    size_t n = data_.size();
    #pragma omp parallel for if (n > 10000)
    for (size_t i = 0; i < n; ++i)
        data_[i] += other.data_[i] * scale;
}

void Tensor::add_bias(const Tensor& bias)
{
    assert(rank() == 2 && "add_bias: Current tensor must be 2D (matrix)");
    assert(bias.size() == shape_[1] && "add_bias: Bias size must match output dimension");

    size_t batch_size = shape_.batch();
    size_t output_size = shape_.features();

    scalar_t* act_ptr = this->data_ptr();
    const scalar_t* bias_ptr = bias.data_ptr();

    #pragma omp parallel for if (batch_size * output_size > 10000)
    for (size_t b = 0; b < batch_size; ++b)
    {
        size_t row_offset = b * output_size;
        for (size_t i = 0; i < output_size; ++i)
        {
            act_ptr[row_offset + i] += bias_ptr[i];
        }
    }
}


void Tensor::sum_rows(const Tensor& input, Tensor& output)
{
    assert(input.rank() == 2 && "sum_rows requires a 2D input tensor");
    
    size_t rows = input.shape()[0];
    size_t cols = input.shape()[1];

    // Redimensionne la sortie si nécessaire (matrice colonne)
    if (output.rank() != 2 || output.shape()[0] != rows || output.shape()[1] != 1)
    {
        output.reshape(Shape({rows, 1}));
    }

    const scalar_t* in_ptr = input.data_ptr();
    scalar_t* out_ptr = output.data_ptr();

    #pragma omp parallel for
    for (size_t r = 0; r < rows; ++r)
    {
        scalar_t sum = 0.0f;
        for (size_t c = 0; c < cols; ++c)
        {
            sum += in_ptr[r * cols + c];
        }
        out_ptr[r] = sum;
    }
}

void Tensor::matmul(const Tensor& A, const Tensor& B, Tensor& C, bool transA, bool transB)
{
    assert(A.rank() == 2 && "Matmul: A must be 2D");
    assert(B.rank() == 2 && "Matmul: B must be 2D");

    size_t M = transA ? A.shape()[1] : A.shape()[0];
    size_t K_A = transA ? A.shape()[0] : A.shape()[1];
    size_t K_B = transB ? B.shape()[1] : B.shape()[0];
    size_t N = transB ? B.shape()[0] : B.shape()[1];

    if (K_A != K_B)
    {
        throw std::invalid_argument("Matmul: dimension mismatch");
    }
    size_t K = K_A;

    if (C.rank() != 2 || C.shape()[0] != M || C.shape()[1] != N)
        C.reshape(Shape({ M, N }));
    C.fill(0.0f);

    scalar_t* C_ptr = C.data_ptr();
    const scalar_t* A_ptr = A.data_ptr();
    const scalar_t* B_ptr = B.data_ptr();
    size_t lda = A.shape()[1];
    size_t ldb = B.shape()[1];
    size_t ldc = C.shape()[1];

    if (!transB)
    {
#pragma omp parallel for if (M * N * K > 5000)
        for (size_t i = 0; i < M; ++i)
        {
            for (size_t k = 0; k < K; ++k)
            {
                size_t a_idx = transA ? k * lda + i : i * lda + k;
                scalar_t a_val = A_ptr[a_idx];
                if (a_val == 0.0f)
                    continue;
                for (size_t j = 0; j < N; ++j)
                {
                    C_ptr[i * ldc + j] += a_val * B_ptr[k * ldb + j];
                }
            }
        }
    }
    else
    {
#pragma omp parallel for if (M * N * K > 5000)
        for (size_t i = 0; i < M; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                scalar_t sum = 0.0f;
                for (size_t k = 0; k < K; ++k)
                {
                    size_t a_idx = transA ? k * lda + i : i * lda + k;
                    sum += A_ptr[a_idx] * B_ptr[j * ldb + k];
                }
                C_ptr[i * ldc + j] = sum;
            }
        }
    }
}

scalar_t& Tensor::operator()(size_t n, size_t c, size_t h, size_t w)
{
    assert(rank() == 4);
    assert(n < shape_[0] && c < shape_[1] && h < shape_[2] && w < shape_[3]);
    return data_[n * strides_[0] + c * strides_[1] + h * strides_[2] + w];
}

const scalar_t& Tensor::operator()(size_t n, size_t c, size_t h, size_t w) const
{
    assert(rank() == 4);
    assert(n < shape_[0] && c < shape_[1] && h < shape_[2] && w < shape_[3]);
    return data_[n * strides_[0] + c * strides_[1] + h * strides_[2] + w];
}

scalar_t& Tensor::operator()(size_t i, size_t j)
{
    assert(rank() == 2);
    assert(i < shape_[0] && j < shape_[1]);
    return data_[i * strides_[0] + j];
}

const scalar_t& Tensor::operator()(size_t i, size_t j) const
{
    assert(rank() == 2);
    assert(i < shape_[0] && j < shape_[1]);
    return data_[i * strides_[0] + j];
}

scalar_t& Tensor::operator()(size_t i)
{
    assert(rank() == 1);
    assert(i < shape_[0]);
    return data_[i];
}

const scalar_t& Tensor::operator()(size_t i) const
{
    assert(rank() == 1);
    assert(i < shape_[0]);
    return data_[i];
}

void Tensor::save(std::ostream& os) const
{
    uint64_t r = rank();
    os.write(reinterpret_cast<const char*>(&r), sizeof(r));
    for (size_t i = 0; i < r; ++i)
    {
        uint64_t dim = shape_[i];
        os.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    os.write(reinterpret_cast<const char*>(data_.data()), data_.size() * sizeof(scalar_t));
}

void Tensor::load(std::istream& is)
{
    uint64_t r;
    if (!is.read(reinterpret_cast<char*>(&r), sizeof(r)))
        throw std::runtime_error("Tensor load failed: cannot read rank");

    if (r > MAX_DIMS)
        throw std::runtime_error("Tensor load failed: rank exceeds MAX_DIMS");

    std::vector<size_t> dims(r);
    for (size_t i = 0; i < r; ++i)
    {
        uint64_t d;
        if (!is.read(reinterpret_cast<char*>(&d), sizeof(d)))
            throw std::runtime_error("Tensor load failed: cannot read dimension");
        dims[i] = d;
    }

    reshape(Shape(dims));
    if (!is.read(reinterpret_cast<char*>(data_.data()), data_.size() * sizeof(scalar_t)))
        throw std::runtime_error("Tensor load failed: truncated data");
}