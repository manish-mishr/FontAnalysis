#include "fontHeader.h"


int main() {
    srand(42);

    // Get all the files in training Dir
    vector<string> fileNames;
    read_directory(TRAINPATH,fileNames);
    int filesz = fileNames.size();

    //Split data in training and Validation and get labels
    vector<int> trainInd, valInd;
    auto indices = kfold(filesz,KFOLD, trainInd,valInd);

    auto trainLabel = getLabel(trainInd,fileNames);
    auto valLabel = getLabel(valInd,fileNames);

    int width = 28, height = 28;
    string file = TRAINPATH+fileNames[0];


    // Example load Image and Display one
//    Mat img = imread(file,0);
//    namedWindow( "Display window", WINDOW_AUTOSIZE );
//    imshow("Display window",img);
//    waitKey(0);

    Mat labelsMat(trainInd.size(),1,CV_16SC1,trainLabel);


    float** trainData = nullptr;
    try{
        trainData = getData(fileNames,trainInd);
    }catch(std::bad_alloc& ba)
    {
        std::cout << "bad_alloc caught: " << ba.what() << '\n';
        exit(1);
    }
  Mat trainingMat(trainInd.size(),AREA,CV_32FC1,trainData);



//    SVM::Params params;
//    params.svmType    = SVM::C_SVC;
//    params.kernelType = SVM::LINEAR;
//    params.termCrit   = TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6);
//
//    Mat image = Mat::zeros(height,width,CV_8UC1);

    // Display Experimental result for correctness
   for(int t=0; t<1; ++t) {
       for(int i=0; i<AREA; ++i){
        auto val = trainData[t][i];
        cout << val << "\t" ;

    }
      cout << endl;
   }
    cout << "checkout:  " << sizeof(trainData[0]) << endl;
    return 0;
}