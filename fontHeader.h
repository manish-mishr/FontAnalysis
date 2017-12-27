//
// Created by root on 12/22/17.
//

#ifndef CLIONPROJECTS_FONTHEADER_H
#define CLIONPROJECTS_FONTHEADER_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <sys/types.h>
#include <dirent.h>
#include <cstdlib>
#include <fstream>
#include <sstream>


using namespace cv;
using namespace cv::ml;
using namespace std;

#define KFOLD 5
#define WIDTH 28
#define AREA 784
#define TRAINPATH string("/home/manish/ClionProjects/data/")
void read_directory(const std::string& name, vector<string>& svec);
void shuffleArray(vector<int> &array,const int size);
vector<int> kfold( const int size, const int k, vector<int> &trainInd,vector<int> & valInd );
void getLabel(const vector<int> &data, const vector<string> &files, Mat &inputMat);
void getData(const vector<string> &files, const vector<int> &indices, Mat &inputMat);
void SVMevaluate(Mat &testResponse, float &count, float &accuracy, Mat &testLabels);
void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels);
void getSVMParams(SVM *svm);
void computeHOG(vector<Mat> &inputCells, vector<vector<float> > &outputHOG);
void ConvertVectortoMatrix(vector<vector<float> > &ipHOG, Mat & opMat);
void SVMtrain(Mat &trainMat, vector<int> &trainLabels, Mat &testResponse, Mat &testMat);



struct DataSet {
    std::string filename;
    float label;
};

class DataSetManager
{
private:
    // user defined data member
    float testDataPercent;
    float validationDataPercent;

    // derrived or internally calculated
    int totalDataNum;
    int totalTrainDataNum;
    int totalTestDataNum;
    int totalValidationDataNum;

public:
    //constructor
    DataSetManager();

    // setter and getter methods
    void setTestDataPercent(float num);
    void setValidationDataPercent(float num);

    int getTotalDataNum();
    int getTotalTrainDataNum();
    int getTotalTestDataNum();
    int getTotalValidationDataNum();

    // primary functions of the class
    void addData(std::string folderName, int classlabel);
    void read();
    void display();// displays the read file names for debugging
    void distribute();
    // ideally these are private; need to update
    std::vector<DataSet> dataList;
    std::vector<DataSet> TrainData;
    std::vector<DataSet> TestData;
    std::vector<DataSet> ValidationData;
};



#endif //CLIONPROJECTS_FONTHEADER_H
