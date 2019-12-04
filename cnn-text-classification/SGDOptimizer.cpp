
#include "pch.h"
#include "SGDOptimizer.h"


SGDOptimizer::SGDOptimizer() {
    m_learning_rate = 0.025;
}

SGDOptimizer::~SGDOptimizer() {
}

void SGDOptimizer::Initialize(int numParams) {
    /* nothing to do, since an SGD optimizer does not have any internal memory */
}

float SGDOptimizer::Optimize(int idx, float prev_param, float new_grad) {
    float new_param = prev_param - m_learning_rate * new_grad;
    return new_param;
}

void SGDOptimizer::SetLearningRate(float learning_rate) {
    m_learning_rate = learning_rate;
}
