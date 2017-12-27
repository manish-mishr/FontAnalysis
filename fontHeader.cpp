//
// Created by root on 12/22/17.
//
#include "fontHeader.h"

void read_directory(const std::string& name, vector<string>& svec)
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL ) {
        if(dp->d_name[0]!= '.')
            svec.push_back(dp->d_name);
    }
    closedir(dirp);
}



void shuffleArray(vector<int> &array,const int size)
{

    for(int i=0; i<size; ++i)
    {
        // 0 <= k < n.
        unsigned int k = rand()%size;

        // n is now the last pertinent inde

        // swap array[n] with array[k]
        swap(array[i],array[k]);
    }
}

vector<int> kfold(const int size,const int k, vector<int> &trainInd,vector<int> & valInd )
{
    vector<int> indices(size);

    for (int i = 0; i < size; i++ )
        indices[ i ] = i%k;

    shuffleArray( indices, size );

    for(int i=0; i<size; ++i){
        if(indices[i] != 4)
            trainInd.push_back(i);
        else
            valInd.push_back(i);
    }
    return move(indices);
}

void getLabel(const vector<int> &data, const vector<string> &files, Mat& inputMat){
    int size = data.size();


    for(int i=0; i<data.size(); ++i){

        string fp = files[data[i]];
        auto val = stoi(fp.substr(8,4));


            if(val%2 == 0)
                inputMat.at<int>(i) = 1;
            else
                inputMat.at<int>(i) = -1;

        }

}

void getData(const vector<string> &files, const vector<int> &indices, Mat &inputMat){

    int sz = indices.size();

    for(int itr=0; itr<sz; ++itr){
        auto ind = indices[itr];
        string fp = TRAINPATH + files[ind];
        Mat grey = imread(fp,CV_LOAD_IMAGE_GRAYSCALE);
        Mat img;
        grey.convertTo(img,CV_32FC1);

        for(int i=0; i<WIDTH; ++i)
            for(int j =0; j<WIDTH; ++j)
               inputMat.at<float>(itr,i*WIDTH+j) = img.at<float>(i,j);
    }

}

void SVMevaluate(Mat &testResponse, float &count, float &accuracy, Mat &testLabels) {
    int dummy;
    for (int i = 0; i<testResponse.rows; i++)
    {
//        cout << testResponse.at<float>(i,0) << " " << testLabels.at<int>(i,0) << endl;
        if (testResponse.at<float>(i, 0) == testLabels.at<int>(i,0)) {

            count = count + 1;
        }

    }
    accuracy = (count / testResponse.rows) * 100;
}

