
#include "pch.h"
#include "InputLayer.h"


InputLayer::InputLayer() {
    m_OutputGrad = nullptr;
}

InputLayer::~InputLayer() {
}

float *InputLayer::GetCurrentData(const char *key) {
    return m_pProvider->GetCurrentData(key);
}

std::vector<int> InputLayer::GetDataDimension(const char *key) {
    return m_pProvider->GetDataDimension(key);
}

std::vector<int> InputLayer::GetCurrentDataDimension(const char *key) {
    return m_pProvider->GetCurrentDataDimension(key);
}

std::vector<int> InputLayer::GetOutputDimension() {
    throw std::exception("InputLayer: GetOutputDimension should not be called");
}

int InputLayer::GetNumberOfParams() {
    return 0;
}

void InputLayer::ForwardProp() {
    /* nothing */
}

void InputLayer::BackwardProp() {
    /* nothing */
}

float *InputLayer::GetOutput() {
    throw std::exception("InputLayer: GetOutput should not be called");
}

void InputLayer::UpdateParams() {
    /* nothing */
}

void InputLayer::SetInputDataProvider(InputDataProvider *pProvider) {
    m_pProvider = pProvider;
}
