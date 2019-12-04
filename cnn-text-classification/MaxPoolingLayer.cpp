
#include "pch.h"
#include "MaxPoolingLayer.h"


MaxPoolingLayer::MaxPoolingLayer(Layer *pInput)
{
    auto indim = pInput->GetOutputDimension();
    if (indim.size() != 1 || indim[0] == 0 || indim[0] <= -2) { /* we allow for indeterminate input dimension */
        throw std::exception("SoftmaxLayer: Input layer has wrong dimensions");
    }
    m_dim = indim[0];
    m_OutputGrad = new float[1];

    AddInLayer(pInput);
}

MaxPoolingLayer::~MaxPoolingLayer()
{
    delete[] m_OutputGrad;
}

std::vector<int> MaxPoolingLayer::GetOutputDimension() {
    return std::vector<int>({ 1 });
}

int MaxPoolingLayer::GetNumberOfParams() {
    return 0; // indicates non-necessity of optimizer
}

void MaxPoolingLayer::ForwardProp() {
    Layer *pInputLayer = m_inLayers[0]; // there is only one such layer
    m_dimCurrent = pInputLayer->GetCurrentOutputDimension()[0];

    float *input = (float *)pInputLayer->GetOutput();
    m_Output = input[0];
    m_ArgMaxIndex = 0;
    for (int i = 0; i < m_dimCurrent; i++) {
        if (input[i] > m_Output) {
            m_Output = input[i];
            m_ArgMaxIndex = i;
        }
    }

    memset(m_OutputGrad, 0, sizeof(float) * 1);
}

void MaxPoolingLayer::BackwardProp() {
    /* compute the gradients */
    m_InputGrad.resize(m_dimCurrent);
    memset(&m_InputGrad[0], 0, sizeof(float)*m_dimCurrent);
    m_InputGrad[m_ArgMaxIndex] = m_OutputGrad[0];

    /* broadcasting */
    BroadcastInputGradient(&m_InputGrad[0]);
}

float* MaxPoolingLayer::GetOutput() {
    return &m_Output;
}

void MaxPoolingLayer::UpdateParams() {
    /* a max-pooling layer does not have any associated parameters */
}
