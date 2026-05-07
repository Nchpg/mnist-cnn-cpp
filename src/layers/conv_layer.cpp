#include "layers/conv_layer.hpp"

#include <cassert>
#include <iostream>
#include <omp.h>
#include <stdexcept>

struct ConvIndexingView {
    size_t k_size, out_h, out_w;

    inline size_t row_idx(size_t c, size_t ky, size_t kx) const {
        return c * (k_size * k_size) + ky * k_size + kx;
    }

    inline size_t col_idx(size_t b, size_t y, size_t x) const {
        return b * (out_h * out_w) + y * out_w + x;
    }
};

ConvLayer::ConvLayer(size_t input_h, size_t input_w, size_t input_c, size_t kernel_size, size_t filter_count,
                     std::mt19937& gen)
    : in_h_(input_h)
    , in_w_(input_w)
    , in_c_(input_c)
    , k_size_(kernel_size)
    , out_c_(filter_count)
{
    out_h_ = input_h - kernel_size + 1;
    out_w_ = input_w - kernel_size + 1;

    biases_.reshape(Shape({ out_c_}));
    biases_.fill(0.0f);

    // He Initialization
    filters_.reshape(Shape({ out_c_, in_c_ * k_size_ * k_size_ }));
    scalar_t std_dev = std::sqrt(2.0f / static_cast<scalar_t>(in_c_ * kernel_size * kernel_size));
    filters_.random_normal(std_dev, gen);
}

/*
 * CONVOLUTIONAL MAPPING & TERMINOLOGY:
 * 
 * 1. PATCH (P_n): Local Receptive Field.
 *    A spatial sub-volume [In_Channels, K_Size, K_Size] of the input tensor.
 *    - Represented as a COLUMN in the 'input_patches_2d' matrix.
 *    - Index n ∈ [0, N-1], where N = Batch * Out_H * Out_W.
 *    - Elements: x_{k,n} (pixel k of patch n).
 *
 * 2. FILTER (F_m): Weight Volume.
 *    A kernel [In_Channels, K_Size, K_Size] that produces one output channel.
 *    - Represented as a ROW in the 'filters_' matrix.
 *    - Index m ∈ [0, M-1], where M = Out_Channels.
 *    - Elements: w_{m,k} (weight k of filter m).
 * 
 * 3. VOLUME (K): Depth of Dot Product.
 *    Total elements in a patch or filter (In_Channels * K_Size * K_Size).
 *    - Index k ∈ [0, K-1].
 *
 * MATHEMATICAL NOTATION SUMMARY:
 * ------------------------------
 *  M        : Number of Filters (Output Channels).
 *  N        : Number of Patches (Total spatial elements).
 *  K        : Volume (Inner dimension of GEMM).
 *  F_m      : The m-th Filter (m ∈ [0, M-1]).
 *  P_n      : The n-th Input Patch (n ∈ [0, N-1]).
 *  w_{m,k}  : Weight k of Filter m.
 *  x_{k,n}  : Pixel k of Patch n.
 *  dL/dOut  : Loss gradient with respect to the layer output.
 */

const Tensor& ConvLayer::forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool /*is_training*/) const
{
    if (input.rank() != 4)
        throw std::invalid_argument("Runtime error: ConvLayer expected a 4D tensor [Batch, Channels, Height, Width].");

    ConvContext* conv_ctx = get_context<ConvContext>(ctx);

    const Shape input_shape = input.shape();
    const size_t batch_size = input_shape.batch();

    conv_ctx->output.reshape(get_output_shape(input.shape()));
    conv_ctx->input_patches_2d.reshape(Shape({ in_c_ * k_size_ * k_size_, batch_size * out_h_ * out_w_ }));
    conv_ctx->raw_output_2d.reshape(Shape({ out_c_, batch_size * out_h_ * out_w_ }));
    ConvIndexingView view{k_size_, out_h_, out_w_};

    /*
     * STEP 1: IM2COL TRANSFORMATION (4D Tensor -> 2D Patches Matrix)
     * -----------------------------------------------------------------
     * We slide the kernel over the 4D input volume to extract local 
     * neighborhoods (patches) and rearrange them into columns of a 2D matrix.
     * 
     * Mapping: Input[b, c, y + ky, x + kx] -> Patches_2D[k, n]
     * 
     *  INPUT (4D Volume) [B, C, H, W]           PATCHES_2D [K x N]
     *  (Conceptual Extraction)                  <--------------- N (Patches) --------------->
     *       _______                             +-------------------------------------------+
     *      /      /|   -- Extract P_0 -->       |  P_0      |  P_1      | ... |  P_{N-1}    | ^
     *     /______/ |   -- Extract P_1 -->       |   |       |   |       |     |    |        | | K (Volume)
     *    |       | |      ...                   | x_{0,0}   | x_{0,1}   |     | x_{0,N-1}   | |
     *    |       |/    -- Extract P_{N-1} -->   |  ...      |  ...      |     |  ...        | |
     *    |_______/                              | x_{K-1,0} | x_{K-1,1} |     | x_{K-1,N-1} | v
     *                                           +-------------------------------------------+
     */
    #pragma omp parallel for collapse(2) if (batch_size * in_c_ > 4)
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t c = 0; c < in_c_; ++c)
        {
            for (size_t ky = 0; ky < k_size_; ++ky)
            {
                for (size_t kx = 0; kx < k_size_; ++kx)
                {
                    size_t row_idx = view.row_idx(c, ky, kx);
                    for (size_t y = 0; y < out_h_; ++y)
                    {
                        for (size_t x = 0; x < out_w_; ++x)
                        {
                            size_t col_idx = view.col_idx(b, y, x);
                            conv_ctx->input_patches_2d(row_idx, col_idx) = input(b, c, y + ky, x + kx);
                        }
                    }
                }
            }
        }
    }

     /*
     * STEP 2: GEMM-BASED CONVOLUTION (Filters * Patches)
     * -----------------------------------------------------------------
     * Operation: Raw_Output_2D = Filters_Matrix * Patches_Matrix.
     *
     * 1. FILTERS (Matrix A): [M x K]
     *    <---------------- K (Volume) ------------->
     *    +-------------------------------------------+
     *    | F_0: [ w_{0,0}, ..., w_{0,K-1} ]          | ^
     *    | F_1: [ w_{1,0}, ..., w_{1,K-1} ]          | | M (Filters)
     *    |   ...                                     | |
     *    | F_{M-1}: [ w_{M-1,0}, ..., w_{M-1,K-1} ]  | v
     *    +-------------------------------------------+
     *
     *                    X (Matrix Multiplication)
     *
     * 2. PATCHES (Matrix B): [K x N]
     *    <--------------- N (Patches) --------------->
     *    +-------------------------------------------+
     *    |  P_0      |  P_1      | ... |  P_{N-1}    | ^
     *    |   |       |   |       |     |    |        | | K (Volume)
     *    | x_{0,0}   | x_{0,1}   |     | x_{0,N-1}   | |
     *    |  ...      |  ...      |     |  ...        | |
     *    | x_{K-1,0} | x_{K-1,1} |     | x_{K-1,N-1} | v
     *    +-------------------------------------------+
     *
     *                    ||
     *
     * 3. RAW_OUTPUT_2D (Matrix C): [M x N]
     *    <----------------- N (Patches) ----------------->
     *    +-----------------------------------------------+
     *    | F_0·P_0     | F_0·P_1 | ... | F_0·P_{N-1}     | ^
     *    |-------------+---------+-----+-----------------| | M (Filters)
     *    | F_{M-1}·P_0 |   ...   |     | F_{M-1}·P_{N-1} | v
     *    +-----------------------------------------------+
     */
    Tensor::matmul(filters_, conv_ctx->input_patches_2d, conv_ctx->raw_output_2d);

    /*
     * STEP 3: BIAS ADDITION & OUTPUT RESHAPING (2D -> 4D)
     * -----------------------------------------------------------------
     * Add the bias term to each dot product result and reshape into the 4D output tensor.
     */
    #pragma omp parallel for collapse(4) if (batch_size * out_c_ * out_h_ * out_w_ > 100)
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t c = 0; c < out_c_; ++c)
        {
            for (size_t y = 0; y < out_h_; ++y)
            {
                for (size_t x = 0; x < out_w_; ++x)
                {
                    size_t col_idx = view.col_idx(b, y, x);
                    conv_ctx->output(b, c, y, x) = conv_ctx->raw_output_2d(c, col_idx) + biases_(c);
                }
            }
        }
    }

    return conv_ctx->output;
}

const Tensor& ConvLayer::backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx, [[maybe_unused]] bool is_training)
{
    /*
     * BACKPROPAGATION OVERVIEW:
     * Given the loss gradient with respect to the output (dL/dOut), we compute:
     * 1. BIAS GRADIENTS (dL/db): Sum of dL/dOut over patches (N).
     * 2. FILTER GRADIENTS (dL/dW): dL/dOut [M x N] * Input^T [N x K].
     * 3. INPUT GRADIENTS (dL/dX): Filters^T [K x M] * dL/dOut [M x N].
     */

    assert(is_training && "Backward must only be called during training!");

    ConvContext* conv_ctx = get_context<ConvContext>(ctx);

    const Shape output_shape = gradient.shape();
    const size_t batch_size = output_shape.batch();

    filters_grad_.reshape(filters_.shape());
    biases_grad_.reshape(biases_.shape());

    conv_ctx->grad_input.reshape(get_input_shape(output_shape));
    conv_ctx->grad_output_2d.reshape(Shape({ out_c_, batch_size * out_h_ * out_w_ }));
    conv_ctx->grad_input_patches_2d.reshape(Shape({ in_c_ * k_size_ * k_size_, batch_size * out_h_ * out_w_ }));
    
    ConvIndexingView view{k_size_, out_h_, out_w_};

    /*
     * STEP 1: GRADIENT REARRANGEMENT (4D Tensor -> 2D Matrix)
     * -----------------------------------------------------------------
     * Flatten dL/dOut [Batch, M, OH, OW] into a 2D matrix [M x N].
     * 
     *  dL/dOut (4D Volume) [B, M, OH, OW]             dL/dOut_2D [M x N]
     *  (Conceptual Flattening)               <--------------- N (Patches) --------------->
     *       _______                          +-------------------------------------------+
     *      /      /|   -- Flatten F_0 -->    | Row 0: [dL/dOut_{0,0} ... dL/dOut_{0,N-1}]| ^
     *     /______/ |   -- Flatten F_1 -->    | Row 1: [dL/dOut_{1,0} ... dL/dOut_{1,N-1}]| | M (Filters)
     *    |       | |      ...                |  ...                                      | |
     *    |       |/    -- Flatten F_{M-1} -> | Row M-1: [ ... ]                          | v
     *    |_______/                           +-------------------------------------------+
     */
    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t f = 0; f < out_c_; ++f)
        {
            for (size_t y = 0; y < out_h_; ++y)
            {
                for (size_t x = 0; x < out_w_; ++x)
                {
                    size_t col_idx = view.col_idx(b, y, x);
                    conv_ctx->grad_output_2d(f, col_idx) = gradient(b, f, y, x);
                }
            }
        }
    }

    /*
     * STEP 2: BIAS GRADIENT CALCULATION (dL/db)
     * -----------------------------------------------------------------
     * Derivation: Out = (W * X) + b  =>  dOut/db = 1.
     * Chain Rule: dL/db = sum( dL/dOut * dOut/db ) = sum( dL/dOut ).
     * 
     *   grad_output_2d (dL/dOut) [M x N]           biases_grad (dL/db)
     *   <----------- N (Patches) ---------->
     *   +------------------------------------+       +-------------+
     *   | dL/dOut_{0,0} ... dL/dOut_{0,N-1}  | sum ->| dL/db[0]    | ^
     *   | dL/dOut_{1,0} ...                  | sum ->| dL/db[1]    | | M (Filters)
     *   | dL/dOut_{M-1,0}...dL/dOut_{M-1,N-1}| sum ->| dL/db[M-1]  | v
     *   +------------------------------------+       +-------------+
     */
    Tensor::sum_rows(conv_ctx->grad_output_2d, biases_grad_);

    /*
     * STEP 3: FILTER GRADIENT CALCULATION (dL/dW)
     * -----------------------------------------------------------------
     * Derivation: Out = W * X + b  =>  dOut/dW = X.
     * Chain Rule: dL/dW = dL/dOut * X^T.
     *
     * 1. dL/dOut: [M x N]
     *    <--------------- N (Patches) --------------->
     *    +-------------------------------------------+
     *    | dL/dOut_{0,0} ... dL/dOut_{0,N-1}         | ^
     *    |  ...                                      | | M (Filters)
     *    | dL/dOut_{M-1,0}...dL/dOut_{M-1,N-1}       | v
     *    +-------------------------------------------+
     *
     *                   X (Matrix Multiplication)
     *
     * 2. (PATCHES)^T: [N x K]
     *    <----------------- K (Volume) ---------------->
     *    +---------------------------------------------+
     *    | P_0^T: [ x_{0,0}, x_{1,0}, ..., x_{K-1,0} ] | ^
     *    | ...                                         | | N (Patches)
     *    | P_{N-1}^T: [x_{0,N-1}, ..., x_{K-1,N-1}]    | v
     *    +---------------------------------------------+
     *
     *                   ||
     *
     * 3. FILTERS_GRAD (dL/dW): [M x K]
     *    <----------- K (Volume) ----------->
     *    +----------------------------------+
     *    | dL/dW_{0,0} ... dL/dW_{0,K-1}    | ^
     *    | ...                              | | M (Filters)
     *    | dL/dW_{M-1,0}... dL/dW_{M-1,K-1} | v
     *    +----------------------------------+
     */
    Tensor::matmul(conv_ctx->grad_output_2d, conv_ctx->input_patches_2d, filters_grad_, false, true);

    /*
     * STEP 4: INPUT GRADIENT CALCULATION (dL/dX_patches)
     * -----------------------------------------------------------------
     * Derivation: Out = W * X + b  =>  dOut/dX = W.
     * Chain Rule: dL/dX_patches = W^T * dL/dOut.
     *
     * 1. (FILTERS)^T: [K x M]
     *    <--------------- M (Filters) --------------->
     *    +-------------------------------------------+
     *    | Row k=0: [w_{0,0}, w_{1,0}, ..., w_{M-1,0}] | ^
     *    | ...                                       | | K (Volume)
     *    | Row k=K-1: [w_{0,K-1}, ..., w_{M-1,K-1}]  | v
     *    +-------------------------------------------+
     *
     *                   X (Matrix Multiplication)
     *
     * 2. dL/dOut: [M x N]
     *    <--------------- N (Patches) --------------->
     *    +-------------------------------------------+
     *    | dL/dOut_{0,0} ... dL/dOut_{0,N-1}         | ^
     *    |  ...                                      | | M (Filters)
     *    | dL/dOut_{M-1,0}...dL/dOut_{M-1,N-1}       | v
     *    +-------------------------------------------+
     *
     *                   ||
     *
     * 3. GRAD_INPUT_PATCHES_2D (dL/dX_patches): [K x N]
     *    <-------------------- N (Patches) ------------------->
     *    +----------------------------------------------------+
     *    | dL/dP_0      | dL/dP_1      | ... | dL/dP_{N-1}    | ^
     *    |   |          |   |          |     |    |           | | K (Volume)
     *    | dL/dx_{0,0}  | dL/dx_{0,1}  |     | dL/dx_{0,N-1}  | |
     *    |  ...         |  ...         |     |  ...           | |
     *    | dL/dx_{K-1,0}| dL/dx_{K-1,1}|     | dL/dx_{K-1,N-1}| v
     *    +----------------------------------------------------+
     */
    Tensor::matmul(filters_, conv_ctx->grad_output_2d, conv_ctx->grad_input_patches_2d, true, false);

    /*
     * STEP 5: GRADIENT SCATTER-ACCUMULATION (2D -> 4D)
     * -----------------------------------------------------------------
     * Accumulate elements from the 2D gradient matrix dL/dX_patches [K x N]
     * into the final 4D input gradient tensor dL/dX [Batch, In_C, In_H, In_W].
     * 
     * Since input pixels are shared between overlapping patches, we must 
     * accumulate (sum) all gradients that reach the same pixel.
     */
    conv_ctx->grad_input.fill(0.0f);
    #pragma omp parallel for collapse(2)
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t c = 0; c < in_c_; ++c)
        {
            for (size_t ky = 0; ky < k_size_; ++ky)
            {
                for (size_t kx = 0; kx < k_size_; ++kx)
                {
                    size_t row_idx = view.row_idx(c, ky, kx);
                    for (size_t y = 0; y < out_h_; ++y)
                    {
                        for (size_t x = 0; x < out_w_; ++x)
                        {
                            size_t col_idx = view.col_idx(b, y, x);
                            conv_ctx->grad_input(b, c, y + ky, x + kx) += conv_ctx->grad_input_patches_2d(row_idx, col_idx);
                        }
                    }
                }
            }
        }
    }

    return conv_ctx->grad_input;
}

void ConvLayer::clear_gradients()
{
    filters_grad_.fill(0.0f);
    biases_grad_.fill(0.0f);
}

Shape ConvLayer::get_output_shape(const Shape& input_shape) const
{
    if (input_shape.rank() != 4) {
        throw std::invalid_argument("Architecture error: ConvLayer requires a 4D input (Batch, Channels, Height, Width).");
    }
    if (input_shape.channels() != in_c_ || input_shape.height() != in_h_ || input_shape.width() != in_w_) {
        throw std::invalid_argument("Architecture error: ConvLayer input dimensions do not match layer configuration.");
    }
    return {input_shape.batch(), out_c_, out_h_, out_w_};
}

Shape ConvLayer::get_input_shape(const Shape& output_shape) const
{
    return {output_shape.batch(), in_c_, in_h_, in_w_};
}

void ConvLayer::save(std::ostream& os) const
{
    uint32_t marker = make_marker("CONV");
    os.write(reinterpret_cast<const char*>(&marker), sizeof(marker));

    uint64_t in_h = in_h_, in_w = in_w_, in_c = in_c_, k_size = k_size_, out_c = out_c_;
    os.write(reinterpret_cast<const char*>(&in_h), sizeof(in_h));
    os.write(reinterpret_cast<const char*>(&in_w), sizeof(in_w));
    os.write(reinterpret_cast<const char*>(&in_c), sizeof(in_c));
    os.write(reinterpret_cast<const char*>(&k_size), sizeof(k_size));
    os.write(reinterpret_cast<const char*>(&out_c), sizeof(out_c));

    filters_.save(os);
    biases_.save(os);
}

void ConvLayer::load(std::istream& is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char*>(&marker), sizeof(marker));
    if (marker != make_marker("CONV"))
        throw std::runtime_error("Arch mismatch in ConvLayer load");

    uint64_t in_h, in_w, in_c, k_size, out_c;
    is.read(reinterpret_cast<char*>(&in_h), sizeof(in_h));
    is.read(reinterpret_cast<char*>(&in_w), sizeof(in_w));
    is.read(reinterpret_cast<char*>(&in_c), sizeof(in_c));
    is.read(reinterpret_cast<char*>(&k_size), sizeof(k_size));
    is.read(reinterpret_cast<char*>(&out_c), sizeof(out_c));

    in_h_ = in_h;
    in_w_ = in_w;
    in_c_ = in_c;
    k_size_ = k_size;
    out_c_ = out_c;
    out_h_ = in_h_ - k_size_ + 1;
    out_w_ = in_w_ - k_size_ + 1;

    filters_.load(is);
    biases_.load(is);
}

nlohmann::json ConvLayer::get_config() const
{
    return { { "type", "Conv" },
             { "filters", static_cast<int>(out_c_) },
             { "kernel_size", static_cast<int>(k_size_) } };
}