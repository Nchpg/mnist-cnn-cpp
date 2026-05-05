#include "layers/dense_layer.hpp"

#include <fstream>
#include <stdexcept>

DenseLayer::DenseLayer(size_t input_size, size_t output_size, std::mt19937 &gen)
    : input_size_(input_size)
    , output_size_(output_size)
{
    weights_.reshape(Shape({ output_size_, input_size_ }));
    biases_.reshape(Shape({ output_size_, 1 }));
    weights_grad_.reshape(Shape({ output_size_, input_size_ }));
    biases_grad_.reshape(Shape({ output_size_, 1 }));

    scalar_t std_dev = std::sqrt(2.0f / static_cast<scalar_t>(input_size));
    weights_.random_normal(std_dev, gen);
    biases_.fill(0.0f);
    clear_gradients();
}

const Tensor &DenseLayer::forward(const Tensor &input)
{
    size_t batch_size = input.shape()[0];
    input_ptr_ = &input;

    if (activations_.rank() != 2 || activations_.shape()[0] != batch_size
        || activations_.shape()[1] != output_size_)
    {
        activations_.reshape(Shape({ batch_size, output_size_ }));
        grad_input_.reshape(Shape({ batch_size, input_size_ }));
    }

    Tensor::matmul(input, weights_, activations_, false, true);

    scalar_t *act_ptr = activations_.data_ptr();
    const scalar_t *bias_ptr = biases_.data_ptr();
#pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t i = 0; i < output_size_; ++i)
        {
            act_ptr[b * output_size_ + i] += bias_ptr[i];
        }
    }

    return activations_;
}

const Tensor &DenseLayer::backward(const Tensor &gradient)
{
    size_t batch_size = gradient.shape()[0];

    Tensor::matmul(gradient, weights_, grad_input_, false, false);
    Tensor::matmul(gradient, *input_ptr_, weights_grad_, true, false);

    biases_grad_.fill(0.0f);
    const scalar_t *grad_ptr = gradient.data_ptr();
    scalar_t *bias_grad_ptr = biases_grad_.data_ptr();
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t i = 0; i < output_size_; ++i)
        {
            bias_grad_ptr[i] += grad_ptr[b * output_size_ + i];
        }
    }

    return grad_input_;
}

void DenseLayer::clear_gradients()
{
    weights_grad_.fill(0.0f);
    biases_grad_.fill(0.0f);
}

void DenseLayer::save(std::ostream &os) const
{
    uint32_t marker = make_marker("DNSL");
    os.write(reinterpret_cast<const char *>(&marker), sizeof(marker));

    uint64_t in_size = input_size_, out_size = output_size_;
    os.write(reinterpret_cast<const char *>(&in_size), sizeof(in_size));
    os.write(reinterpret_cast<const char *>(&out_size), sizeof(out_size));

    weights_.save(os);
    biases_.save(os);
}

void DenseLayer::load(std::istream &is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    if (marker != make_marker("DNSL"))
        throw std::runtime_error("Architecture mismatch in DenseLayer load");

    uint64_t in_size, out_size;
    is.read(reinterpret_cast<char *>(&in_size), sizeof(in_size));
    is.read(reinterpret_cast<char *>(&out_size), sizeof(out_size));

    input_size_ = in_size;
    output_size_ = out_size;

    weights_.load(is);
    biases_.load(is);
}

nlohmann::json DenseLayer::get_config() const
{
    return { { "type", "Dense" }, { "units", output_size_ } };
}

Shape3D DenseLayer::get_output_shape(const Shape3D &input_shape) const
{
    (void)input_shape;
    return { output_size_, 1, 1 };
}