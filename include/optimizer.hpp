#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "layer.hpp"
#include <vector>
#include <memory>
#include <cmath>

class Optimizer {
private:
    scalar_t learning_rate_;

protected:
    std::vector<Parameter> parameters_;

public:
    Optimizer(scalar_t learning_rate = 0.0f) : learning_rate_(learning_rate) {}
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    virtual void reset() {}
    virtual void set_learning_rate(scalar_t lr) { learning_rate_ = lr; }
    scalar_t learning_rate() const { return learning_rate_; }

    virtual void add_parameters(const std::vector<Parameter>& params) {
        parameters_.insert(parameters_.end(), params.begin(), params.end());
    }
};

class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(scalar_t learning_rate) : Optimizer(learning_rate) {}

    void step() override {
        for (auto& param : parameters_) {
            if (param.value && param.gradient) {
                param.value->subtract_scaled(*param.gradient, learning_rate());
            }
        }
    }
};

class AdamOptimizer : public Optimizer {
private:
    scalar_t beta1_;
    scalar_t beta2_;
    scalar_t epsilon_;
    size_t t_;
    std::vector<Matrix> m_;
    std::vector<Matrix> v_;

public:
    AdamOptimizer(scalar_t learning_rate, scalar_t beta1 = 0.9f, scalar_t beta2 = 0.999f, scalar_t epsilon = 1e-8f)
        : Optimizer(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {}

    void add_parameters(const std::vector<Parameter>& params) override {
        size_t start_idx = parameters_.size();
        Optimizer::add_parameters(params);
        for (size_t i = start_idx; i < parameters_.size(); ++i) {
            auto& param = parameters_[i];
            if (param.value && param.gradient) {
                m_.emplace_back(param.value->rows(), param.value->cols());
                v_.emplace_back(param.value->rows(), param.value->cols());
            } else {
                m_.emplace_back();
                v_.emplace_back();
            }
        }
    }

    void step() override {
        t_++;
        scalar_t beta1_t = std::pow(beta1_, static_cast<scalar_t>(t_));
        scalar_t beta2_t = std::pow(beta2_, static_cast<scalar_t>(t_));
        scalar_t m_corr = 1.0f / (1.0f - beta1_t);
        scalar_t v_corr = 1.0f / (1.0f - beta2_t);

        for (size_t i = 0; i < parameters_.size(); ++i) {
            auto& param = parameters_[i];
            if (!param.value || !param.gradient) continue;

            size_t total = param.value->size();
            scalar_t* w_ptr = param.value->data();
            scalar_t* wg_ptr = param.gradient->data();
            scalar_t* mw_ptr = m_[i].data();
            scalar_t* vw_ptr = v_[i].data();

            #pragma omp parallel for if(total > 1000)
            for (size_t j = 0; j < total; ++j) {
                mw_ptr[j] = beta1_ * mw_ptr[j] + (1.0f - beta1_) * wg_ptr[j];
                vw_ptr[j] = beta2_ * vw_ptr[j] + (1.0f - beta2_) * wg_ptr[j] * wg_ptr[j];
                scalar_t m_hat = mw_ptr[j] * m_corr;
                scalar_t v_hat = vw_ptr[j] * v_corr;
                w_ptr[j] -= learning_rate() * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    }

    void reset() {
        t_ = 0;
        for (auto& m_mat : m_) m_mat.fill(0.0f);
        for (auto& v_mat : v_) v_mat.fill(0.0f);
    }
};

#endif