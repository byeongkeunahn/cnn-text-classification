
#pragma once


/* layers */
#include "Layer.h"
#include "ConcatenationLayer.h"
#include "ConvolutionLayer.h"
#include "CrossEntropyLossLayer.h"
#include "DropoutLayer.h"
#include "EmbeddingLayer.h"
#include "FullyConnectedLayer.h"
#include "LabelPredictionLayer.h"
#include "InputDataProvider.h"
#include "InputLayer.h"
#include "InputProxyLayer.h"
#include "MaxPoolingLayer.h"
#include "OneHotLayer.h"
#include "ReLUActivationLayer.h"
#include "Sentence2DLayer.h"
#include "SoftmaxLayer.h"


/* datasets */
#include "Dataset.h"
#include "TextClassificationDataset.h"


/* optimizers */
#include "Optimizer.h"
#include "AdadeltaOptimizer.h"
#include "SGDOptimizer.h"


/* finally, our model class */
#include "Model.h"
