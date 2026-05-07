#include "layers/dense_layer.hpp"

#include <cassert>
#include <fstream>
#include <stdexcept>

DenseLayer::DenseLayer(size_t input_size, size_t output_size, std::mt19937& gen)
    : input_size_(input_size)
    , output_size_(output_size)
{
    biases_.reshape(Shape({ output_size_, 1 }));
    biases_.fill(0.0f);

    // He Initialization
    const scalar_t std_dev = std::sqrt(2.0f / static_cast<scalar_t>(input_size));
    weights_.reshape(Shape({ output_size_, input_size_ }));
    weights_.random_normal(std_dev, gen);
}

/*
 * DENSE LAYER MAPPING & TERMINOLOGY:
 * 
 * 1. INPUT (X): [Batch x In_Features]
 *    Matrix of feature vectors. Elements: x_{b,i}
 * 
 * 2. WEIGHTS (W): [Out_Features x In_Features]
 *    Weights per neuron. Elements: w_{o,i}
 * 
 * 3. BIASES (b): [Out_Features x 1]
 *    One bias per output neuron. Elements: b_{o}
 * 
 * 4. OUTPUT (Y): [Batch x Out_Features]
 *    Linear result: Y = X * W^T + b. Elements: y_{b,o}
 */

const Tensor& DenseLayer::forward(const Tensor& input, std::unique_ptr<LayerContext>& ctx, bool /*is_training*/) const
{
    if (input.rank() != 2) {
        throw std::invalid_argument("Runtime error: DenseLayer expected a 2D tensor.");
    }
    DenseContext* dense_ctx = get_context<DenseContext>(ctx);

    const Shape input_shape = input.shape();

    dense_ctx->input_ptr = &input;
    dense_ctx->activations.reshape(get_output_shape(input_shape));

    /*
     * STEP 1: LINEAR TRANSFORMATION (X * W^T)
     * ---------------------------------------
     * Operation: MatMul(X, W, Transpose_B=True)
     * 
     *    INPUT (X) [B x I]          WEIGHTS^T (W^T) [I x O]       OUTPUT (Y) [B x O]
     *    +-------------------+      +-------------------+         +-------------------+
     *    | x_{0,0}   x_{0,1} |      | w_{0,0}   w_{1,0} |         | y_{0,0}   y_{0,1} | ^
     *    | x_{1,0}   x_{1,1} |  X   | w_{0,1}   w_{1,1} |    =    | y_{1,0}   y_{1,1} | | B (Batch)
     *    |  ...       ...    |      |  ...       ...    |         |  ...       ...    | v
     *    +-------------------+      +-------------------+         +-------------------+
     *      <---- I (In) ---->         <---- O (Out) ---->           <---- O (Out) ---->
     */
    Tensor::matmul(input, weights_, dense_ctx->activations, false, true);

    /*
     * STEP 2: BIAS ADDITION
     * ---------------------
     * Add the bias vector [O x 1] to each row of the output matrix.
     * 
     *    ACTIVATIONS (Y) [B x O]       BIASES (b) [O x 1]          RESULT
     *    +-------------------+         +-------+           +-----------------------+
     *    | y_{0,0}   y_{0,1} |         | b_{0} |           | y_{0,0}+b_0  y_{0,1}+b_1|
     *    | y_{1,0}   y_{1,1} |    +    | b_{1} |     =     | y_{1,0}+b_0  y_{1,1}+b_1|
     *    +-------------------+         +-------+           +-----------------------+
     */
    dense_ctx->activations.add_bias(biases_);

    return dense_ctx->activations;
}

const Tensor& DenseLayer::backward(const Tensor& gradient, std::unique_ptr<LayerContext>& ctx, [[maybe_unused]] bool is_training)
{
    /*
     * BACKPROPAGATION OVERVIEW:
     * -------------------------
     * Given the loss gradient w.r.t output dL/dY [B x O], we compute:
     * 1. dL/dX (Grad Input): dL/dY * W           -> [B x O] * [O x I] = [B x I]
     * 2. dL/dW (Grad Weights): (dL/dY)^T * X     -> [O x B] * [B x I] = [O x I]
     * 3. dL/db (Grad Biases): sum_rows(dL/dY)    -> [O x 1]
     */

    assert(is_training && "Backward doit uniquement etre appele durant l'entrainement !");

    DenseContext* dense_ctx = get_context<DenseContext>(ctx);

    const Shape output_shape = gradient.shape();

    dense_ctx->grad_input.reshape(get_input_shape(output_shape));

    weights_grad_.reshape(weights_.shape());
    biases_grad_.reshape(biases_.shape());

    /*
     * STEP 1: dL/dX
     * ------------------------------------
     * Derivation: Y = X * W^T + b  =>  dY/dX = W
     * Chain Rule: dL/dX = dL/dY * (dY/dX) = dL/dY * W
     * 
     *  dL/dY [B x O]                 WEIGHTS (W) [O x I]         dL/dX [B x I]
     *  +------------------------+     +-------------------+       +------------------------+
     *  | dL/dy_{0,0} dL/dy_{0,1}|     | w_{0,0}   w_{0,1} |       | dL/dx_{0,0} dL/dx_{0,1}|
     *  | dL/dy_{1,0} dL/dy_{1,1}|  X  | w_{1,0}   w_{1,1} |   =   | dL/dx_{1,0} dL/dx_{1,1}|
     *  +------------------------+     +-------------------+       +------------------------+
     */
    Tensor::matmul(gradient, weights_, dense_ctx->grad_input, false, false);

    /*
     * STEP 2: dL/dW
     * --------------------------------------
     * Derivation: Y = X * W^T + b =>  dY/dW = X
     * Chain Rule: dL/dW = (dL/dY)^T * X
     * (Transposed to match [O x I] weight shape)
     * 
     *  (dL/dY)^T [O x B]             INPUT (X) [B x I]           dL/dW [O x I]
     *  +------------------------+     +-------------------+       +------------------------+
     *  | dL/dy_{0,0} dL/dy_{1,0}|     | x_{0,0}   x_{0,1} |       | dL/dw_{0,0} dL/dw_{0,1}|
     *  | dL/dy_{0,1} dL/dy_{1,1}|  X  | x_{1,0}   x_{1,1} |   =   | dL/dw_{1,0} dL/dw_{1,1}|
     *  +------------------------+     +-------------------+       +------------------------+
     */
    Tensor::matmul(gradient, *(dense_ctx->input_ptr), weights_grad_, true, false);

    /*
     * STEP 3: dL/db
     * -------------------------------------
     * Derivation: Y = X * W^T + b  =>  dY/db = 1
     * Chain Rule: dL/db = sum(dL/dY * 1)
     * 
     *    dL/dY [B x O]                                           dL/db [O x 1]
     *    +------------------------+                               +----------------------------+
     *    | dL/dy_{0,0} dL/dy_{0,1}| -- (Sum col 0 elements) -->   | dL/dy_{0,0} + dL/dy_{1,0} +..|
     *    | dL/dy_{1,0} dL/dy_{1,1}| -- (Sum col 1 elements) -->   | dL/dy_{0,1} + dL/dy_{1,1} +..|
     *    +------------------------+                               +----------------------------+
     */
    biases_grad_.fill(0.0f);
    biases_grad_.add_bias(gradient);

    return dense_ctx->grad_input;
}

void DenseLayer::clear_gradients()
{
    weights_grad_.fill(0.0f);
    biases_grad_.fill(0.0f);
}

void DenseLayer::save(std::ostream& os) const
{
    uint32_t marker = make_marker("DNSL");
    os.write(reinterpret_cast<const char*>(&marker), sizeof(marker));

    uint64_t in_size = input_size_, out_size = output_size_;
    os.write(reinterpret_cast<const char*>(&in_size), sizeof(in_size));
    os.write(reinterpret_cast<const char*>(&out_size), sizeof(out_size));

    weights_.save(os);
    biases_.save(os);
}

void DenseLayer::load(std::istream& is)
{
    uint32_t marker;
    is.read(reinterpret_cast<char*>(&marker), sizeof(marker));
    if (marker != make_marker("DNSL"))
        throw std::runtime_error("Architecture mismatch in DenseLayer load");

    uint64_t in_size, out_size;
    is.read(reinterpret_cast<char*>(&in_size), sizeof(in_size));
    is.read(reinterpret_cast<char*>(&out_size), sizeof(out_size));

    input_size_ = in_size;
    output_size_ = out_size;

    weights_.load(is);
    biases_.load(is);
}

nlohmann::json DenseLayer::get_config() const
{
    return { { "type", "Dense" }, { "units", output_size_ } };
}

Shape DenseLayer::get_output_shape(const Shape& input_shape) const
{
    if (input_shape.rank() != 2) {
        throw std::invalid_argument("Architecture error: DenseLayer requires a 2D input (Batch, Features). Did you forget a FlattenLayer?");
    }
    if (input_shape.features() != input_size_) {
        throw std::invalid_argument("Architecture error: DenseLayer input feature size mismatch.");
    }
    return {input_shape.batch(), output_size_};
}

Shape DenseLayer::get_input_shape(const Shape& output_shape) const
{
    return {output_shape.batch(), input_size_};
}