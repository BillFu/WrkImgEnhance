#include <algorithm>

#include "WrinkleFrangi.h"
//#include "../ImgProc.h"
//#include "../Utils.hpp"
#include "frangi.h"

extern string outDir;  // 在这里申明，在main.cpp里定义。

////////////////////////////////////////////////////////////////////////////////////////

// Cvt: convert
Mat CvtFloatImgTo8UImg(Mat& ftImg)
{
    float maxV = *max_element(ftImg.begin<float>(), ftImg.end<float>());
    float minV = *min_element(ftImg.begin<float>(), ftImg.end<float>());
    
    cout << "maxV: " << maxV << endl;
    cout << "minV: " << minV << endl;
    
    float alpha = 255.0 / (maxV - minV);
    float beta = -255.0 * minV / (maxV - minV);
    
    Mat img8U;
    ftImg.convertTo(img8U, CV_8U, alpha, beta);

    return img8U;
}

void CalcFrgiResp(const Mat& grSrcImg,
                  int scaleRatio,
                  Mat& frgiRespRzU8)
{
    Size rzSize = grSrcImg.size() / scaleRatio;
    Mat rzImg;
    resize(grSrcImg, rzImg, rzSize);
    
    Mat rzFlImg;
    rzImg.convertTo(rzFlImg, CV_32FC1);
    rzImg.release();
    
    cv::Mat respScaleRz, respAngRz;
    frangi2d_opts opts;
    opts.sigma_start = 1;
    opts.sigma_end = 5;
    opts.sigma_step = 2;
    opts.BetaOne = 0.5;  // BetaOne: suppression of blob-like structures.
    opts.BetaTwo = 15.0; // background suppression. (See Frangi1998...)
    opts.BlackWhite = true;
    
    // !!! 计算fangi2d时，使用的是缩小版的衍生影像
    Mat frgiRespRz;
    frangi2d(rzFlImg, frgiRespRz, respScaleRz, respAngRz, opts);
    rzFlImg.release();
    
    //返回的scaleRz, anglesRz没有派上实际的用场
    respScaleRz.release();
    respAngRz.release();
        
    frgiRespRzU8 = CvtFloatImgTo8UImg(frgiRespRz);
}


void CalcSobelRespInFhReg(const Mat& grSrcImg,
                         const Rect& fhRect,
                         int scaleRatio,
                         Mat& frgiRespRz)
{
    Mat imgOfFh = grSrcImg(fhRect);
    
    GaussianBlur( imgOfFh, imgOfFh, Size(3,3), 0, 0, BORDER_DEFAULT );
    
    Mat grad_y;
    //Sobel(imgOfFh, grad_y, CV_8U, 0, 1, 3, 1, 0, BORDER_DEFAULT);

    Scharr(imgOfFh, grad_y, CV_16S, 0, 1);
    convertScaleAbs(grad_y, grad_y, 2.0, 20.0);
    
    //grad_y = ~grad_y;
#ifdef TEST_RUN2
    string gradImgFile = BuildOutImgFNV2(outDir, "gradYInFh.png");
    bool isOK = imwrite(gradImgFile, grad_y);
    assert(isOK);
#endif

}
