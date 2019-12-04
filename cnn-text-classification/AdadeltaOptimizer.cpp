
#include "pch.h"
#include "AdadeltaOptimizer.h"


AdadeltaOptimizer::AdadeltaOptimizer(float decay_rate, float epsilon) {
    m_decay_rate = decay_rate;
    m_epsilon = epsilon;
}

AdadeltaOptimizer::~AdadeltaOptimizer() {
}

void AdadeltaOptimizer::Initialize(int numParams) {
    /* allocate memory for internal memory (use vector) */
    m_g2.resize(numParams);
    m_dx2.resize(numParams);

    /* zero-initialize */
    std::fill(m_g2.begin(), m_g2.end(), 0.0f);
    std::fill(m_dx2.begin(), m_dx2.end(), 0.0f);
}

float AdadeltaOptimizer::Optimize(int idx, float prev_param, float new_grad) {
    float &g2 = m_g2[idx];
    float &dx2 = m_dx2[idx];

    /* accumulate gradient */
    g2 = m_decay_rate * g2 + (1 - m_decay_rate) * std::pow(new_grad, 2);

    /* compute update */
    float dx = -(std::sqrt(dx2 + m_epsilon) / std::sqrt(g2 + m_epsilon)) * new_grad;

    /* accumulate updates */
    dx2 = m_decay_rate * dx2 + (1 - m_decay_rate) * std::pow(dx, 2);

    /* apply update */
    float new_param = prev_param + dx;
    return new_param;
}
