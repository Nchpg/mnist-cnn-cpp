#include "layers/pooling_layer.hpp"

#include <cassert>
#include <limits>

PoolingLayer::PoolingLayer(size_t input_h, size_t input_w, size_t input_c,
                           size_t pool_size, size_t stride)
    : in_h_(input_h)
    , in_w_(input_w)
    , in_c_(input_c)
    , pool_size_(pool_size)
    , stride_(stride)
{
    out_h_ = (input_h - pool_size) / stride + 1;
    out_w_ = (input_w - pool_size) / stride + 1;
}

const Tensor &PoolingLayer::forward(const Tensor &input,
                                    std::unique_ptr<LayerContext> &ctx,
                                    bool is_training) const
{
    if (!ctx)
    {
        ctx = std::make_unique<PoolingContext>();
    }
    auto *pool_ctx = static_cast<PoolingContext *>(ctx.get());

    size_t batch_size = input.shape()[0];

    if (pool_ctx->output.shape().rank() == 0
        || pool_ctx->output.shape()[0] != batch_size)
    {
        pool_ctx->output.reshape(Shape({ batch_size, in_c_, out_h_, out_w_ }));
        if (is_training)
        {
            pool_ctx->argmax_indices.resize(pool_ctx->output.size());
        }
    }

#pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t c = 0; c < in_c_; ++c)
        {
            for (size_t i = 0; i < out_h_; ++i)
            {
                for (size_t j = 0; j < out_w_; ++j)
                {
                    size_t start_r = i * stride_;
                    size_t start_c = j * stride_;

                    scalar_t max_val =
                        -std::numeric_limits<scalar_t>::infinity();
                    size_t max_idx = 0;

                    for (size_t ki = 0; ki < pool_size_; ++ki)
                    {
                        for (size_t kj = 0; kj < pool_size_; ++kj)
                        {
                            size_t in_r = start_r + ki;
                            size_t in_c = start_c + kj;

                            scalar_t val = input(b, c, in_r, in_c);
                            if (val > max_val)
                            {
                                max_val = val;
                                max_idx = in_r * in_w_ + in_c;
                            }
                        }
                    }
                    pool_ctx->output(b, c, i, j) = max_val;
                    if (is_training)
                    {
                        size_t out_idx =
                            ((b * in_c_ + c) * out_h_ + i) * out_w_ + j;
                        pool_ctx->argmax_indices[out_idx] = max_idx;
                    }
                }
            }
        }
    }

    (void)is_training;
    return pool_ctx->output;
}

const Tensor &PoolingLayer::backward(const Tensor &gradient,
                                     std::unique_ptr<LayerContext> &ctx,
                                     bool is_training)
{
    assert(is_training
           && "Backward doit uniquement etre appele durant l'entrainement !");
    auto *pool_ctx = static_cast<PoolingContext *>(ctx.get());

    size_t batch_size = gradient.shape()[0];
    if (pool_ctx->grad_input.shape().rank() == 0
        || pool_ctx->grad_input.shape()[0] != batch_size)
    {
        pool_ctx->grad_input.reshape(
            Shape({ batch_size, in_c_, in_h_, in_w_ }));
    }
    pool_ctx->grad_input.fill(0.0f);

#pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t c = 0; c < in_c_; ++c)
        {
            for (size_t i = 0; i < out_h_; ++i)
            {
                for (size_t j = 0; j < out_w_; ++j)
                {
                    size_t out_idx =
                        ((b * in_c_ + c) * out_h_ + i) * out_w_ + j;
                    size_t max_in_idx = pool_ctx->argmax_indices[out_idx];
                    size_t in_r = max_in_idx / in_w_;
                    size_t in_c = max_in_idx % in_w_;

                    pool_ctx->grad_input(b, c, in_r, in_c) +=
                        gradient(b, c, i, j);
                }
            }
        }
    }

    return pool_ctx->grad_input;
}

void PoolingLayer::save(std::ostream &os) const
{
    uint32_t marker = make_marker("POOL");
    os.write(reinterpret_cast<const char *>(&marker), sizeof(marker));

    uint64_t in_h = in_h_, in_w = in_w_, in_c = in_c_, pool_size = pool_size_,
             stride = stride_;
    os.write(reinterpret_cast<const char *>(&in_h), sizeof(in_h));
    os.write(reinterpret_cast<const char *>(&in_w), sizeof(in_w));
    os.write(reinterpret_cast<const char *>(&in_c), sizeof(in_c));
    os.write(reinterpret_cast<const char *>(&pool_size), sizeof(pool_size));
    os.write(reinterpret_cast<const char *>(&stride), sizeof(stride));
}

void PoolingLayer::load(std::istream &is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    if (marker != make_marker("POOL"))
        throw std::runtime_error("Architecture mismatch in PoolingLayer load");

    uint64_t in_h, in_w, in_c, pool_size, stride;
    is.read(reinterpret_cast<char *>(&in_h), sizeof(in_h));
    is.read(reinterpret_cast<char *>(&in_w), sizeof(in_w));
    is.read(reinterpret_cast<char *>(&in_c), sizeof(in_c));
    is.read(reinterpret_cast<char *>(&pool_size), sizeof(pool_size));
    is.read(reinterpret_cast<char *>(&stride), sizeof(stride));

    in_h_ = in_h;
    in_w_ = in_w;
    in_c_ = in_c;
    pool_size_ = pool_size;
    stride_ = stride;
    out_h_ = (in_h_ - pool_size_) / stride_ + 1;
    out_w_ = (in_w_ - pool_size_) / stride_ + 1;
}

nlohmann::json PoolingLayer::get_config() const
{
    return { { "type", "Pool" },
             { "pool_size", pool_size_ },
             { "stride", stride_ } };
}

Shape3D PoolingLayer::get_output_shape(const Shape3D &input_shape) const
{
    return { input_shape.channels,
             (input_shape.height - pool_size_) / stride_ + 1,
             (input_shape.width - pool_size_) / stride_ + 1 };
}