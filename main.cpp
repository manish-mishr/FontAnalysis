#include "fontHeader.h"


int main() {
    srand(42);

    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Major version : " << CV_MAJOR_VERSION << endl;
    cout << "Minor version : " << CV_MINOR_VERSION << endl;
    cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;


/************************************************************************/
cout << "Load Data from  " << endl;
    DataSetManager dm;
    string positive = TRAINPATH+"Bold";
    string negative = TRAINPATH+"NotBold";
    dm.addData(positive,1);
    dm.addData(negative,-1);
    cout << "Total data length : " << dm.getTotalDataNum() << endl;
    dm.distribute();
    dm.display();

/***********load all the dataset into vector of Mat*********/
    vector<Mat> trainCells;
    vector<Mat> testCells;
    vector<int> trainLabels;
    vector<int> testLabels;
    for (int i = 0; i<dm.getTotalTrainDataNum(); i++) {
        string imageName = dm.TrainData[i].filename;
        Mat img = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
        trainCells.push_back(img);
        trainLabels.push_back(dm.TrainData[i].label);
    }
    for (int i = 0; i<dm.getTotalTestDataNum(); i++) {
        string imageName = dm.TestData[i].filename;
        Mat img = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
        testCells.push_back(img);
        testLabels.push_back(dm.TestData[i].label);
    }
/****************** Compute Hog Descriptor******************/

    std::vector<std::vector<float> > trainHOG;
    std::vector<std::vector<float> > testHOG;

    //compute_hog(trainCells, gradient_lst);
    computeHOG(trainCells, trainHOG);
    computeHOG(testCells, testHOG);

    int descriptor_size = trainHOG[0].size();
    cout << "Descriptor Size : " << descriptor_size << endl;

/********Prepeare trainData and test data and call SVM ML algorithm*********/
    Mat trainMat(trainHOG.size(), descriptor_size, CV_32FC1);
    Mat testMat(testHOG.size(), descriptor_size, CV_32FC1);
    ConvertVectortoMatrix(trainHOG, trainMat);
    ConvertVectortoMatrix(testHOG, testMat);

    Mat testResponse;
    SVMtrain(trainMat, trainLabels, testResponse, testMat);

    float countN = 0;
    float accuracyN = 0;
    SVMevaluate(testResponse, countN, accuracyN, testLabels);

    cout << "the accuracy is :" << accuracyN << endl;


    return 0;
}