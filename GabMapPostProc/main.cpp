//
//  main.cpp
//  GabMapPostProc
//
//  Created by meicet on 2022/11/14.
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

typedef vector<vector<Point2i>> CONTOURS;

Mat drawHistogram(Mat &hist, int hist_h = 400, int hist_w = 1012, int hist_size = 256,
                  Scalar color = Scalar(255, 255, 255), int type = 2, string title = "Histogram")
{
    int bin_w = cvRound( (double) hist_w/hist_size );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    switch (type)
    {
    case 1:
        for(int i = 0; i < hist_size; i++)
        {
            const unsigned x = i;
            const unsigned y = hist_h;

            line(histImage, Point(bin_w * x, y),
                 Point(bin_w * x, y - cvRound(hist.at<float>(i))),
                 color);
        }
        break;
            
    case 2:
        for( int i = 1; i < hist_size; ++i)
        {
            Point pt1 = Point(bin_w * (i-1), hist_h);
            Point pt2 = Point(bin_w * i, hist_h);
            Point pt3 = Point(bin_w * i, hist_h - cvRound(hist.at<float>(i)));
            Point pt4 = Point(bin_w * (i-1), hist_h - cvRound(hist.at<float>(i-1)));
            Point pts[] = {pt1, pt2, pt3, pt4, pt1};

            fillConvexPoly(histImage, pts, 5, color);
        }
        break;
            
    default:
        for( int i = 1; i < hist_size; ++i)
        {
            line( histImage, Point( bin_w * (i-1), hist_h - cvRound(hist.at<float>(i-1))) ,
                             Point( bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
                             color, 1, 8, 0);
        }
        break;
    }

    imshow(title, histImage);

    return histImage;
}

double getEccentricity(Moments &mu)
{
    double bigSqrt = sqrt( ( mu.m20 - mu.m02 ) *  ( mu.m20 - mu.m02 )  + 4 * mu.m11 * mu.m11  );
    return (double) ( mu.m20 + mu.m02 + bigSqrt ) / ( mu.m20 + mu.m02 - bigSqrt );
}

double eccentricity2(vector<Point2i>& contour )
{
    RotatedRect ellipse = fitEllipse(contour);
    return ellipse.size.height / ellipse.size.width; // size is an Size2f, float
}

void projOnAxisY(const Mat& srcMat, vector<int>& projY)
{
    int rows = srcMat.rows;
    
    Mat line;
    for (int i = 0; i < srcMat.rows; i++)
    {
        line = srcMat.row(i);
        int sumLine = cv::sum(line)[0];
        projY.push_back(sumLine);
    }
    
    /*
    std::sort(projY.begin(), projY.end());
    
    for (int i = 0; i < srcMat.rows; i++)
    {
        cout << "i: " << i << "projY: " << projY[i] << endl;
    }
    */
}

Mat projOnAxisYV2(const Mat& srcMat)
{
    int rows = srcMat.rows;
    Mat projY(rows, 1, CV_32FC1, Scalar(0.0));
    
    Mat line;
    for (int i = 0; i < srcMat.rows; i++)
    {
        line = srcMat.row(i);
        int sumLine = cv::sum(line)[0];
        projY.at<float>(i, 0) = sumLine;
        //projY.push_back(sumLine);
        cout << i << ": " << sumLine << endl;
    }
    
    return projY;
}

int main(int argc, const char * argv[])
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
    Mat inImg = imread(inImgFile.c_str(), IMREAD_GRAYSCALE);
    if(inImg.empty())
    {
        cout << "Failed to load input iamge: " << inImgFile << endl;
        return 0;
    }
    else
        cout << "Succeeded to load image: " << inImgFile << endl;
    
    /// Establish the number of bins
    int histSize = 256;

    /// Set the range
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true;
    bool accumulate = false;

    // compute the histogram
    Mat hist;
    calcHist(&inImg, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

    // calculate cumulative histogram
    Mat c_hist(hist.size(), hist.type());

    c_hist.at<float>(0) = hist.at<float>(0);
    
    for(size_t k = 1; k < hist.rows; ++k)
        c_hist.at<float>(k) = hist.at<float>(k) + c_hist.at<float>(k-1);

    // draw histogram
    //drawHistogram(hist, 400, 1024, hist.rows, Scalar(255, 255, 255), 2);
    // draw cumulative histogram
    //drawHistogram(c_hist, 400, 1024, c_hist.rows, Scalar(255, 255, 255), 2, "cumHist");

    int imgH = inImg.rows;
    
    float accuRatio = (float)(imgH - 22 )/ (float)(imgH);
    int biTh = 0;
    
    int numPixelTh = accuRatio * inImg.cols * inImg.rows;
    for(size_t k = 1; k < hist.rows; ++k)
    {
        //cout << c_hist.at<float>(k) << endl;
        if(c_hist.at<float>(k) > numPixelTh)
        {
            biTh = k;
            break;
        }
    }
    
    cout << "biTh: " << biTh << endl;
    
    cv::Mat WrkBi(inImg.size(), CV_8UC1, cv::Scalar(0));

    threshold(inImg, WrkBi, biTh, 255, THRESH_BINARY);
    
    namedWindow("Binary Result", WINDOW_AUTOSIZE );
    imshow("Binary Result", WrkBi );
    
    Mat element3 = getStructuringElement(
                MORPH_ELLIPSE, Size(21, 3));
    Mat DilateWrkBi;
    
    Mat ErodeWrkBi;

    dilate(WrkBi, DilateWrkBi, element3, Point2i(-1,-1), 1);
    erode(DilateWrkBi, ErodeWrkBi, element3, Point2i(-1,-1), 1);
    DilateWrkBi.release();
    
    //imshow("DilateWrkBi", DilateWrkBi);
    imshow("ErodeWrkBi", ErodeWrkBi);
    
    CONTOURS Cts;
    findContours(ErodeWrkBi, Cts, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    CONTOURS::const_iterator it_ct = Cts.begin();
    unsigned long ct_size = Cts.size();
    
    CONTOURS wrkCts;
    for (unsigned int i = 0; i < ct_size; ++i)
    {
        if (it_ct->size() >= 80)
        {
            wrkCts.push_back(Cts[i]);
        }
        
        it_ct++;
    }
    
    Mat canvas(inImg.size(), CV_8UC1, Scalar(0));
    drawContours(canvas, wrkCts, -1, cv::Scalar(255), FILLED, 1);
    imshow("wrkCts", canvas);

    CONTOURS finalWrkCts;
    for (unsigned int i = 0; i < wrkCts.size(); ++i)
    {
        double ecc = eccentricity2(wrkCts[i]);
        //cout << "ecc: " << ecc << endl;
        
        if (ecc > 4.0)
        {
            finalWrkCts.push_back(wrkCts[i]);
        }
        
        it_ct++;
    }
    
    Mat canvas2(inImg.size(), CV_8UC1, Scalar(0));
    drawContours(canvas2, finalWrkCts, -1, cv::Scalar(255), FILLED, 1);
    imshow("finalWrkCts", canvas2);

    waitKey(0);
    
    return 0;
}
