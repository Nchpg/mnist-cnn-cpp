#include "optimizers/adam_optimizer.hpp"

AdamOptimizer::AdamOptimizer(scalar_t learning_rate, scalar_t beta1, scalar_t beta2, scalar_t epsilon)
    : Optimizer(learning_rate)
    , beta1_(beta1)
    , beta2_(beta2)
    , epsilon_(epsilon)
    , t_(0)
    , beta1_t_(beta1)
    , beta2_t_(beta2)
{}

void AdamOptimizer::set_parameters(const std::vector<Tensor*>& weights, const std::vector<Tensor*>& grads)
{
    Optimizer::set_parameters(weights, grads);
    m_.clear();
    v_.clear();
    for (auto* w : weights)
    {
        if (w)
        {
            m_.emplace_back(w->shape(), 0.0f);
            v_.emplace_back(w->shape(), 0.0f);
        }
        else
        {
            m_.emplace_back();
            v_.emplace_back();
        }
    }
}

void AdamOptimizer::step()
{
    t_++;
    beta1_t_ *= beta1_;
    beta2_t_ *= beta2_;
    scalar_t m_corr = 1.0f / (1.0f - beta1_t_);
    scalar_t v_corr = 1.0f / (1.0f - beta2_t_);

    for (size_t i = 0; i < weights_.size(); ++i)
    {
        if (!weights_[i] || !gradients_[i])
            continue;
        if (weights_[i]->size() == 0 || gradients_[i]->size() == 0)
            continue;

        size_t total = weights_[i]->size();
        scalar_t* w_ptr = weights_[i]->data().data();
        scalar_t* wg_ptr = gradients_[i]->data().data();
        scalar_t* mw_ptr = m_[i].data().data();
        scalar_t* vw_ptr = v_[i].data().data();

#pragma omp parallel for if (total > 1000)
        for (size_t j = 0; j < total; ++j)
        {
            mw_ptr[j] = beta1_ * mw_ptr[j] + (1.0f - beta1_) * wg_ptr[j];
            vw_ptr[j] = beta2_ * vw_ptr[j] + (1.0f - beta2_) * wg_ptr[j] * wg_ptr[j];
            scalar_t m_hat = mw_ptr[j] * m_corr;
            scalar_t v_hat = std::max(0.0f, vw_ptr[j] * v_corr);
            w_ptr[j] -= learning_rate() * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
}

void AdamOptimizer::reset()
{
    t_ = 0;
    beta1_t_ = beta1_;
    beta2_t_ = beta2_;
    for (auto& m_mat : m_)
        m_mat.fill(0.0f);
    for (auto& v_mat : v_)
        v_mat.fill(0.0f);
}