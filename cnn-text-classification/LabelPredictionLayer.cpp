
#include "pch.h"
#include "LabelPredictionLayer.h"


LabelPredictionLayer::LabelPredictionLayer(Layer *pProbDist) {
    auto indim = pProbDist->GetOutputDimension();
    if (indim.size() != 1 || indim[0] <= 0) {
        throw std::exception("LabelPredictionLayer: Invalid input layer dimension");
    }

    m_dim = indim[0];

    AddInLayer(pProbDist);
    m_pProbDist = pProbDist;

    m_OutputGrad = nullptr;
}

LabelPredictionLayer::~LabelPredictionLayer() {
}

std::vector<int> LabelPredictionLayer::GetOutputDimension() {
    return std::vector<int>({ m_dim });
}

int LabelPredictionLayer::GetNumberOfParams() {
    return 0; // indicates non-necessity of optimizer
}

void LabelPredictionLayer::ForwardProp() {
    float *input = (float *)m_pProbDist->GetOutput();

    int argmax = 0;
    float max_val = input[0];
    for (int i = 0; i < m_dim; i++) {
        if (input[i] > max_val) {
            argmax = i;
            max_val = input[i];
        }
    }

    m_Output = argmax;
}

void LabelPredictionLayer::BackwardProp() {
    /* no backpropgation */
}

float* LabelPredictionLayer::GetOutput() {
    return &m_Output;
}

void LabelPredictionLayer::UpdateParams() {
    /* a label prediction layer does not have any associated parameters */
}
