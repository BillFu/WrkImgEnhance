//
//  main.cpp
//  WrkImgEnhance
//
//  Created by meicet on 2022/11/9.
//

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "nlohmann/json.hpp"
#include "CLAHE.hpp"
#include "WrinkleFrangi.h"

using json = nlohmann::json;

using namespace std;
using namespace cv;

string outDir("");

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "{target} config_file" << endl;
        return 0;
    }
    
    string errorMsg;
    json config_json;            // 创建 json 对象
    ifstream jfile(argv[1]);
    jfile >> config_json;        // 以文件流形式读取 json 文件
        
    string inImgFile = config_json.at("InImg");
    outDir = config_json.at("OutDir");
    
    
    // Load Input Image
    Mat inImg = cv::imread(inImgFile.c_str());
    if(inImg.empty())
    {
        cout << "Failed to load input iamge: " << inImgFile << endl;
        return 0;
    }
    else
        cout << "Succeeded to load image: " << inImgFile << endl;
    cv::Mat imgGray;
    cvtColor(inImg, imgGray, COLOR_BGR2GRAY);

    string outGrayImgFileName =  outDir + "/grayParallel.png";
    imwrite(outGrayImgFileName, imgGray);
    
    /*
    int burKerS = 9;

    Mat blurGrImg;
    blur(imgGray, blurGrImg, Size(burKerS, burKerS));
    Mat clachRst;
    int gridSize = 24;
    ApplyCLAHE(blurGrImg, gridSize, clachRst);
    
    string claheFhFN =  outDir + "/clachFhb" +
        to_string(burKerS) + "_g" + to_string(gridSize) + ".png";
    imwrite(claheFhFN, clachRst);
    
    Mat frgiRespRz8U;
    CalcFrgiResp(clachRst, 2, frgiRespRz8U);
    clachRst.release();
    
    string frgiRespImgFile =  outDir + "/frgiFhb" +
        to_string(burKerS) + "_g" + to_string(gridSize) + ".png";

    imwrite(frgiRespImgFile, frgiRespRz8U);
    */
    
    return 0;
}
