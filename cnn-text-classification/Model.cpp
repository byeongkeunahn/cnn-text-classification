
#include "pch.h"
#include "Model.h"
#include "DirectedGraph.h"


Model::Model() {
}

Model::~Model() {
}

void Model::SetTrainDataset(Dataset *pTrainDataset) {
    m_pTrainDataset = pTrainDataset;
}

void Model::SetInputLayer(InputLayer *pInputLayer) {
    m_pInputLayer = pInputLayer;
}

void Model::SetOutputLayer(Layer *pOutputLayer) {
    m_pOutputLayer = pOutputLayer;
}

void Model::SetPredictionLayer(Layer *pPredictionLayer) {
    m_pPredictionLayer = pPredictionLayer;
}

void Model::SetOptimizer(Optimizer *pOptimizer) {
    m_pOptimizer = pOptimizer;
}

void Model::Build() {
    /* obtain a toplogical sort of the computation graph */
    bool fSucceeded;
    m_SortedLayers = ObtainTopologicalSort(m_pOutputLayer, &fSucceeded);
    if (!fSucceeded) {
        throw std::exception("Model: Current computation graph is not acyclic; Cannot evaluate the network");
    }
    m_SortedLayersForPrediction = ObtainTopologicalSort(m_pPredictionLayer, &fSucceeded);
    if (!fSucceeded) {
        throw std::exception("Model: Current computation graph for prediction is not acyclic; Cannot evaluate the network");
    }

    /* compute the total number of parameters of the model */
    int numParams = 0;
    for (auto pLayer : m_SortedLayers) {
        numParams += pLayer->GetNumberOfParams();
    }

    /* initialize the optimizer */
    m_pOptimizer->Initialize(numParams);

    /* supply the optimizer to the layers */
    int numParamStart = 0;
    for (auto pLayer : m_SortedLayers) {
        pLayer->SetOptimizer(m_pOptimizer, numParamStart);
        numParamStart += pLayer->GetNumberOfParams();
    }
}

void Model::TrainSingleEpoch(size_t minibatchSize) {
    /* obtain the training order */
    m_EpochIndices.resize(m_pTrainDataset->Count());
    for (size_t i = 0; i < m_EpochIndices.size(); i++) m_EpochIndices[i] = i;
    std::random_shuffle(m_EpochIndices.begin(), m_EpochIndices.end());

    /* prepare training mode */
    for (auto pLayer : m_SortedLayers) {
        pLayer->SetTesting(false);
    }

    /* perform training */
    size_t total = m_EpochIndices.size();
    size_t minibatch_count = 0;
    size_t remaining_count = total;
    float loss = 0;
    for (auto idx : m_EpochIndices) {
        /* set the training example */
        InputDataProvider idp(m_pTrainDataset, idx);
        m_pInputLayer->SetInputDataProvider(&idp);

        /* forward propagation */
        for (size_t i = 0; i < m_SortedLayers.size(); i++) {
            m_SortedLayers[i]->ForwardProp();
        }

        /* accumulation of loss */
        loss += m_pOutputLayer->GetOutput()[0];

        /* backward propagation */
        for (size_t i = m_SortedLayers.size() - 1; i < m_SortedLayers.size(); i--) {
            m_SortedLayers[i]->BackwardProp();
        }

        /* parameter update */
        minibatch_count++;
        remaining_count--;
        if (minibatch_count >= minibatchSize && remaining_count >= minibatchSize) {
            RequestParameterUpdates();
            printf("[%zu/%zu] Current loss: %.6f\n", total - remaining_count, total, loss / minibatch_count);
            minibatch_count = 0;
            loss = 0;
        }
    }
    if (minibatch_count > 0) {
        RequestParameterUpdates();
        printf("[%zu/%zu] Current loss: %.6f\n", total - remaining_count, total, loss / minibatch_count);
        minibatch_count = 0;
        loss = 0;
    }
}

void Model::Predict(Dataset *pTestDataset, int *PredictedLabels) {
    /* prepare testing mode */
    for (auto pLayer : m_SortedLayersForPrediction) {
        pLayer->SetTesting(true);
    }

    for (size_t i = 0; i < pTestDataset->Count(); i++) {
        /* set the test example */
        InputDataProvider idp(pTestDataset, i);
        m_pInputLayer->SetInputDataProvider(&idp);

        /* forward propagation */
        for (size_t i = 0; i < m_SortedLayersForPrediction.size(); i++) {
            m_SortedLayersForPrediction[i]->ForwardProp();
        }

        /* save the predicted label */
        PredictedLabels[i] = std::roundf(m_pPredictionLayer->GetOutput()[0]);
    }
}

std::vector<Layer*> Model::ObtainTopologicalSort(Layer *pFinalLayer, bool *pfSucceeded) {
    /* obtain a topological sort of the computation graph */
    DirectedGraph dig; // if A's result is used by B, then there is an edge A->B
    std::vector<Layer*> layersPtr;
    std::map<Layer*, int> layersPtrToIndex;

    int u, v;
    u = dig.NewVertex();
    layersPtr.push_back(pFinalLayer);
    layersPtrToIndex[pFinalLayer] = u;

    std::queue<Layer*> Q;
    Q.push(pFinalLayer);

    while (!Q.empty()) {
        auto pLayer = Q.front(); Q.pop();
        v = layersPtrToIndex[pLayer];

        Layer **ppInLayers = pLayer->GetInLayers();
        size_t InLayersCount = pLayer->GetInLayersCount();
        for (size_t i = 0; i < InLayersCount; i++) {
            Layer *pIn = ppInLayers[i];
            auto it = layersPtrToIndex.find(pIn);
            if (it == layersPtrToIndex.end()) {
                /* a new fresh in-layer */
                u = dig.NewVertex();
                layersPtr.push_back(pIn);
                layersPtrToIndex[pIn] = u;
                Q.push(pIn);
            }
            else {
                /* this layer was already added to the layers set */
                u = it->second;
            }
            dig.AddEdge(u, v);
        }
    }

    auto toposorted = dig.TopologicalSort(pfSucceeded);
    std::vector<Layer*> result;
    if (*pfSucceeded) {
        /* save the sorted results */
        result.resize(layersPtr.size());
        for (size_t i = 0; i < layersPtr.size(); i++) {
            result[i] = layersPtr[toposorted[i]];
        }
    }

    return result;
}

void Model::RequestParameterUpdates() {
    for (Layer *pLayer : m_SortedLayers) {
        pLayer->UpdateParams();
    }
}
