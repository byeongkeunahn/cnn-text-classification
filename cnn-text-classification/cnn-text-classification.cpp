// cnn-text-classification.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include "pch.h"
#include "IncludeAll.h"


int main()
{
    std::cout << "Hello World!\n";

    /* read the dataset */
    TextClassificationDataset train(
        LR"(D:\Dev\School2019\ML-2019\NEW (Sarcasm Classification)\01_preprocess\without_oov\sarcasm-stc-wordindices-train.txt)",
        LR"(D:\Dev\School2019\ML-2019\NEW (Sarcasm Classification)\01_preprocess\without_oov\sarcasm-label-train.txt)"
    );
    TextClassificationDataset test(
        LR"(D:\Dev\School2019\ML-2019\NEW (Sarcasm Classification)\01_preprocess\without_oov\sarcasm-stc-wordindices-test.txt)",
        LR"(D:\Dev\School2019\ML-2019\NEW (Sarcasm Classification)\01_preprocess\without_oov\sarcasm-label-test.txt)"
    );

    /* build the neural network */
    InputLayer layer_Input;
    InputDataProvider idp_tmp(&train, 0);
    layer_Input.SetInputDataProvider(&idp_tmp); // provide temporary data provider for model building (needed for dimension query)
    InputProxyLayer layer_Seqs(&layer_Input, "seqs");
    InputProxyLayer layer_Label(&layer_Input, "labels");

    EmbeddingLayer layer_Embedding(LR"(D:\Dev\School2019\ML-2019\NEW (Sarcasm Classification)\01_preprocess\without_oov\sarcasm-stc.txt.embed.vec.txt)");
    Sentence2DLayer layer_Sentence2D(&layer_Embedding, &layer_Seqs, false);

    std::vector<ConvolutionLayer*> layers_Conv;
    for (int H = 3; H <= 5; H++) {
        for (int i = 0; i < 100; i++) {
            auto p = new ConvolutionLayer(&layer_Sentence2D, H);
            layers_Conv.push_back(p);
        }
    }

    std::vector<Layer*> layers_MaxPool_Ptr;
    for (auto pLayer : layers_Conv) {
        auto p = new MaxPoolingLayer(pLayer);
        layers_MaxPool_Ptr.push_back(p);
    }

    ConcatenationLayer layer_Concat(&layers_MaxPool_Ptr[0], (int)layers_MaxPool_Ptr.size());
    DropoutLayer layer_Dropout(&layer_Concat, 0.5);
    FullyConnectedLayer layer_FC(&layer_Dropout, 2);
    SoftmaxLayer layer_Softmax(&layer_FC);
    OneHotLayer layer_OneHot(&layer_Label, { 0,1 });
    CrossEntropyLossLayer layer_xEnt(&layer_Softmax, &layer_OneHot);
    LabelPredictionLayer layer_Pred(&layer_Softmax);

    /* initialize model */
    Model model;
    model.SetTrainDataset(&train);
    model.SetInputLayer(&layer_Input);
    model.SetOutputLayer(&layer_xEnt);
    model.SetPredictionLayer(&layer_Pred);
    //AdadeltaOptimizer optimizer(0.95, 1e-6); // the best parameter given in the paper for MNIST
    SGDOptimizer sgd;
    sgd.SetLearningRate(0.02);
    model.SetOptimizer(&sgd);
    model.Build();

    /* training */
    const int Epochs = 5;
    for (int epoch = 1; epoch <= Epochs; epoch++) {
        printf("Epoch %d start\n", epoch);
        model.TrainSingleEpoch(50);
        printf("Epoch %d complete\n", epoch);
    }

    /* report final accuracy */
    TextClassificationDataset *pTest = &test;
    std::vector<int> labels_predict(pTest->Count());
    model.Predict(pTest, &labels_predict[0]);
    int correct = 0;
    for (size_t i = 0; i < pTest->Count(); i++) {
        int label = std::roundf(pTest->GetData(i, "labels")[0]);
        if (label == labels_predict[i]) correct++;
    }
    printf("\nTest Accuracy: %.6f\n\n", correct / (double)pTest->Count());

    /* cleanup */
    for (auto p : layers_MaxPool_Ptr) delete p;
    for (auto p : layers_Conv) delete p;

    return 0;
}
