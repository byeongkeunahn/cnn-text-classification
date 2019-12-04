
#pragma once

#include "Layer.h"
#include "InputLayer.h"
#include "Dataset.h"
#include "Optimizer.h"


class Model
{
public:
    Model();
    ~Model();

public:
    void SetTrainDataset(Dataset *pTrainDataset);
    void SetInputLayer(InputLayer *pInputLayer);
    void SetOutputLayer(Layer *pOutputLayer);
    void SetPredictionLayer(Layer *pPredictionLayer);
    void SetOptimizer(Optimizer *pOptimizer);
    void Build();
    void TrainSingleEpoch(size_t minibatchSize);
    void Predict(Dataset *pTestDataset, int *PredictedLabels);

private:
    std::vector<Layer*> ObtainTopologicalSort(Layer *pFinalLayer, bool *pfSucceeded);
    void RequestParameterUpdates();

private:
    Dataset *m_pTrainDataset;
    std::vector<size_t> m_EpochIndices;

    InputLayer *m_pInputLayer;
    Layer *m_pOutputLayer;
    Layer *m_pPredictionLayer;
    Optimizer *m_pOptimizer;

    std::vector<Layer*> m_SortedLayers, m_SortedLayersForPrediction; // forward-sorted (input first, output last)
};
