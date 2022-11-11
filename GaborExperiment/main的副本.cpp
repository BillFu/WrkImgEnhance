//
//  main.cpp
//  GaborExperiment
//
//  Created by meicet on 2022/11/11.
//

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>

using namespace cv;
using namespace std;

#include "nlohmann/json.hpp"

using json = nlohmann::json;
string outDir("");

cv::Mat BuildKernel(int kerSize, double sig, double th, double lm, double ps)
{
    int hks = (kerSize-1)/2;
    double theta = th*CV_PI/180;
    double psi = ps*CV_PI/180;
    double del = 2.0/(kerSize-1);
    double lmbd = lm;
    double sigma = sig/kerSize;
    double x_theta;
    double y_theta;
    
    cv::Mat kernel(kerSize,kerSize, CV_32F);
    
    for (int y=-hks; y<=hks; y++)
    {
        for (int x=-hks; x<=hks; x++)
        {
            x_theta = x*del*cos(theta)+y*del*sin(theta);
            y_theta = -x*del*sin(theta)+y*del*cos(theta);
            kernel.at<float>(hks+y,hks+x) = (float)exp(-0.5*(pow(x_theta,2)+pow(y_theta,2))/pow(sigma,2))* cos(2*CV_PI*x_theta/lmbd + psi);
        }
    }
    return kernel;
}

// BdKerAspRatio: build kernel with diferent width and height, i.e., the aspect ratio is NOT 1.0.
cv::Mat BdKerAspRatio(int kerSize,
                      double gamma, // aspect ratio, HorKerSize / VerKerSize
                      double sig,
                      double thetaDeg,  // in degree
                      double lambda,
                      double psiDeg   // in degree
                      )
{
    if(kerSize % 2 == 0)
        kerSize += 1;
    
    int hks = (kerSize - 1)/2; // h: half

    double theta = thetaDeg * CV_PI/180;
    double psi = psiDeg * CV_PI/180;
    double del = 2.0/(kerSize-1);
    double lmbd = lambda;
    double sigma = sig/kerSize;
    double x_theta;
    double y_theta;
    
    cv::Mat kernel(kerSize, kerSize, CV_32F);
    
    for (int y=-hks; y<=hks; y++)
    {
        for (int x=-hks; x<=hks; x++)
        {
            x_theta = x*del*cos(theta)+y*del*sin(theta);
            y_theta = -x*del*sin(theta)+y*del*cos(theta);
            kernel.at<float>(hks+y,hks+x) = (float)exp(-0.5*(pow(x_theta,2)+pow(y_theta,2))/pow(sigma,2))* cos(2*CV_PI*x_theta/lmbd + psi);
        }
    }
    return kernel;
}

int kernelSize = 21;
int pos_sigma= 5;
int pos_lm = 50;
int pos_th = 0;
int pos_psi = 90;
cv::Mat src_f;
cv::Mat gaborMap;

void Process(int , void *)
{
    double sig = pos_sigma;
    double lm = 0.5 + pos_lm/100.0;
    double th = pos_th;
    double ps = pos_psi;
    
    cv::Mat kernel = BuildKernel(kernelSize, sig, th, lm, ps);
    
    cv::filter2D(src_f, gaborMap, CV_32F, kernel);
    cv::imshow("Process window", gaborMap);
    cv::Mat Lkernel(kernelSize*20, kernelSize*20, CV_32F);
    cv::resize(kernel, Lkernel, Lkernel.size());
    Lkernel /= 2.;
    Lkernel += 0.5;
    
    cv::imshow("Kernel", Lkernel);
    
    cv::Mat mag;
    cv::pow(gaborMap, 2.0, mag);
    cv::imshow("Mag", mag);
}

int main(int argc, char** argv)
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
    //cv::Mat imgGray;
    //cvtColor(inImg, imgGray, COLOR_BGR2GRAY);
    
    inImg.convertTo(src_f, CV_32F, 1.0/255, 0);
    if (kernelSize % 2 == 0)
    {
        kernelSize += 1;
    }
    
    cv::namedWindow("Process window", 1);
    cv::createTrackbar("Sigma", "Process window", &pos_sigma, kernelSize, Process);
    cv::createTrackbar("Lambda", "Process window", &pos_lm, 100, Process);
    cv::createTrackbar("Theta", "Process window", &pos_th, 180, Process);
    cv::createTrackbar("Psi", "Process window", &pos_psi, 360, Process);
    Process(0,0);
    cv::waitKey(0);
    return 0;
}
