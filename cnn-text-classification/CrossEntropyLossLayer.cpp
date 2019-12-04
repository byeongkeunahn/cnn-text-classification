
#include "pch.h"
#include "CrossEntropyLossLayer.h"


CrossEntropyLossLayer::CrossEntropyLossLayer(Layer *pProbDist, Layer *pTrueLabelOneHot)
{
    auto pdim = pProbDist->GetOutputDimension();
    if (pdim.size() != 1 || pdim[0] <= 0) {
        throw std::exception("CrossEntropyLossLayer: ProbDist input layer has wrong dimensions");
    }

    auto ldim = pTrueLabelOneHot->GetOutputDimension();
    if (ldim.size() != 1 || ldim[0] <= 0) {
        throw std::exception("CrossEntropyLossLayer: TrueLabelOneHot input layer has wrong dimensions");
    }

    m_dim = pdim[0];
    if (ldim[0] != m_dim) {
        throw std::exception("CrossEntropyLossLayer: ProbDist and TrueLabelOneHot input layers have different dimensions");
    }

    m_InputGrad = new float[m_dim];
    m_OutputGrad = nullptr; // we don't need output gradient since this is the final layer

    AddInLayer(pProbDist);
    AddInLayer(pTrueLabelOneHot);

    m_pProbDist = pProbDist;
    m_pTrueLabelOneHot = pTrueLabelOneHot;
}

CrossEntropyLossLayer::~CrossEntropyLossLayer()
{
    delete[] m_InputGrad;
}

std::vector<int> CrossEntropyLossLayer::GetOutputDimension() {
    return std::vector<int>({ 1 });
}

int CrossEntropyLossLayer::GetNumberOfParams() {
    return 0; // indicates non-necessity of optimizer
}

void CrossEntropyLossLayer::ForwardProp() {
    float *y = m_pProbDist->GetOutput(), *h = m_pTrueLabelOneHot->GetOutput();
    float *o = &m_Output; *o = 0;
    for (int i = 0; i < m_dim; i++) {
        *o += -h[i] * std::logf(y[i]);
    }
}

void CrossEntropyLossLayer::BackwardProp() {
    /* compute the gradients */
    float *y = m_pProbDist->GetOutput(), *h = m_pTrueLabelOneHot->GetOutput();
    for (int i = 0; i < m_dim; i++) {
        m_InputGrad[i] = -h[i] / y[i];
    }

    /* broadcasting */
    m_pProbDist->UpdateOutputGradient(m_InputGrad);
}

float* CrossEntropyLossLayer::GetOutput() {
    return &m_Output;
}

void CrossEntropyLossLayer::UpdateParams() {
    /* a cross-entropy loss layer does not have any associated parameters */
}
