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



using namespace cv;
using namespace cv::ml;
using namespace std;

#define KFOLD 5
#define WIDTH 28
#define AREA 784
#define TRAINPATH "/home/manish/ClionProjects/data/train/"
void read_directory(const std::string& name, vector<string>& svec);
void shuffleArray(vector<int> &array,const int size);
vector<int> kfold( const int size, const int k, vector<int> &trainInd,vector<int> & valInd );
short* getLabel(const vector<int> &data, const vector<string> &files);
float** getData(const vector<string> &files, const vector<int> &indices);
#endif //CLIONPROJECTS_FONTHEADER_H
