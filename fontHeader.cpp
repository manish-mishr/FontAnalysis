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

short* getLabel(const vector<int> &data, const vector<string> &files){
    int size = data.size();
    short* labels = new short[size];

    for(int i=0; i<data.size(); ++i){

        string fp = files[data[i]];
        auto val = stoi(fp.substr(8,4));


            if(val%2 == 0)
                labels[i] = 1;
            else
                labels[i] = -1;

        }
    return move(labels);
}

float** getData(const vector<string> &files, const vector<int> &indices){
    int sz = indices.size();
    float** arr = new float*[sz];
    for(int i=0; i<sz; ++i)
        arr[i] = new float[AREA];



    for(int itr=0; itr<sz; ++itr){
        auto ind = indices[itr];
        string fp = TRAINPATH + files[ind];
        Mat img = imread(fp,0);

        for(int i=0; i<WIDTH; ++i)
            for(int j =0; j<WIDTH; ++j)
                arr[itr][i*WIDTH+j] = static_cast<float>(img.at<uchar>(i,j));
    }


    return arr;
}