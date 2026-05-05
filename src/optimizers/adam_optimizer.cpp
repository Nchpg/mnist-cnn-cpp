#include "optimizers/adam_optimizer.hpp"

AdamOptimizer::AdamOptimizer(scalar_t learning_rate, scalar_t beta1,
                             scalar_t beta2, scalar_t epsilon)
    : Optimizer(learning_rate)
    , beta1_(beta1)
    , beta2_(beta2)
    , epsilon_(epsilon)
    , t_(0)
{}

void AdamOptimizer::add_parameters(const std::vector<Parameter> &params)
{
    size_t start_idx = parameters_.size();
    Optimizer::add_parameters(params);
    for (size_t i = start_idx; i < parameters_.size(); ++i)
    {
        auto &param = parameters_[i];
        if (param.value && param.gradient)
        {
            m_.emplace_back(param.value->rows(), param.value->cols());
            v_.emplace_back(param.value->rows(), param.value->cols());
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
    scalar_t beta1_t = std::pow(beta1_, static_cast<scalar_t>(t_));
    scalar_t beta2_t = std::pow(beta2_, static_cast<scalar_t>(t_));
    scalar_t m_corr = 1.0f / (1.0f - beta1_t);
    scalar_t v_corr = 1.0f / (1.0f - beta2_t);

    for (size_t i = 0; i < parameters_.size(); ++i)
    {
        auto &param = parameters_[i];
        if (!param.value || !param.gradient)
            continue;

        size_t total = param.value->size();
        scalar_t *w_ptr = param.value->data();
        scalar_t *wg_ptr = param.gradient->data();
        scalar_t *mw_ptr = m_[i].data();
        scalar_t *vw_ptr = v_[i].data();

#pragma omp parallel for if (total > 1000)
        for (size_t j = 0; j < total; ++j)
        {
            mw_ptr[j] = beta1_ * mw_ptr[j] + (1.0f - beta1_) * wg_ptr[j];
            vw_ptr[j] =
                beta2_ * vw_ptr[j] + (1.0f - beta2_) * wg_ptr[j] * wg_ptr[j];
            scalar_t m_hat = mw_ptr[j] * m_corr;
            scalar_t v_hat = std::max(0.0f, vw_ptr[j] * v_corr);
            w_ptr[j] -= learning_rate() * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
}

void AdamOptimizer::reset()
{
    t_ = 0;
    for (auto &m_mat : m_)
        m_mat.fill(0.0f);
    for (auto &v_mat : v_)
        v_mat.fill(0.0f);
}