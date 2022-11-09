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

    //string outGrayImgFileName =  outDir + "/grayParallel.png";
    //imwrite(outGrayImgFileName, imgGray);
    int imgW = imgGray.cols;
    int imgH = imgGray.rows;
    
    // 这个公式仅对前额区域有效；若imgW表示图像全域或其他子区域，这个公式需要调整。
    // 也许这个公式以后需要调整为普遍适用的公式。
    int blurKerS = imgW / 142; // 1/142 约等于9/1286
    if(blurKerS % 2 == 0)
        blurKerS += 1;  // make it be a odd number
    
    Mat blurGrImg;
    blur(imgGray, blurGrImg, Size(blurKerS, blurKerS));
    
    Mat clachRst;
    // 这个公式仅对前额区域有效；若imgW表示图像全域或其他子区域，这个公式需要调整。
    // 也许这个公式以后需要调整为普遍适用的公式。
    int gridSize = imgW / 54; // 1/54与24/1286有关
    ApplyCLAHE(blurGrImg, gridSize, clachRst);
    
    string claheFhFN =  outDir + "/clachFhb" +
        to_string(blurKerS) + "_g" + to_string(gridSize) + ".png";
    imwrite(claheFhFN, clachRst);
    
    Mat frgiRespRz8U;
    CalcFrgiResp(clachRst, 2, frgiRespRz8U);
    clachRst.release();
    
    string frgiRespImgFile =  outDir + "/frgiFhb" +
        to_string(blurKerS) + "_g" + to_string(gridSize) + ".png";

    imwrite(frgiRespImgFile, frgiRespRz8U);
    
    return 0;
}
