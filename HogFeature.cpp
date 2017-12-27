//
// Created by manish on 12/26/17.
//
#include "fontHeader.h"
HOGDescriptor hog(
        Size(28,28), //winSize
        Size(14,14), //blocksize
        Size(7,7), //blockStride,
        Size(7,7), //cellSize,
        18, //nbins,
        1, //derivAper,
        -1, //winSigma,
        0, //histogramNormType,
        0.2, //L2HysThresh,
        0,//gammal correction,
        64,//nlevels=64
        1
);

void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels) {

    for (int i = 0; i<testResponse.rows; i++)
    {
        //cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
        if (testResponse.at<float>(i, 0) == testLabels[i]) {
            count = count + 1;
        }
    }
    accuracy = (count / testResponse.rows) * 100;
}

void computeHOG(vector<Mat> &inputCells, vector<vector<float> > &outputHOG) {

    for (int y = 0; y<inputCells.size(); y++) {
        vector<float> descriptors;
        hog.compute(inputCells[y], descriptors);
        outputHOG.push_back(descriptors);
    }
}

void getSVMParams(SVM *svm)
{
    cout << "Kernel type     : " << svm->getKernelType() << endl;
    cout << "Type            : " << svm->getType() << endl;
    cout << "C               : " << svm->getC() << endl;
    cout << "Degree          : " << svm->getDegree() << endl;
    cout << "Nu              : " << svm->getNu() << endl;
    cout << "Gamma           : " << svm->getGamma() << endl;
}


void ConvertVectortoMatrix(vector<vector<float> > &ipHOG, Mat & opMat)
{

    int descriptor_size = ipHOG[0].size();
    for (int i = 0; i<ipHOG.size(); i++) {
        for (int j = 0; j<descriptor_size; j++) {
            opMat.at<float>(i, j) = ipHOG[i][j];
        }
    }
}

void SVMtrain(Mat &trainMat, vector<int> &trainLabels, Mat &testResponse, Mat &testMat) {
    Ptr<SVM> svm = SVM::create();
    svm->setGamma(0.50625);
    svm->setC(2.5);
    svm->setKernel(SVM::RBF);
    svm->setType(SVM::C_SVC);
    Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
    svm->train(td);
//    svm->trainAuto(td);
    string savefile("/home/ClionProjects/model.yml");
    svm->save("model.xml");
    svm->predict(testMat, testResponse);
    getSVMParams(svm);

}

