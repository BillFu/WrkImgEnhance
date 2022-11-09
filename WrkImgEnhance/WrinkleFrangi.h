//
//  wrinkle_frangi.h
//  MCSkinAnaLib
//
//  Created by Fu on 2022/10/20.
//  Copyright © 2022 MeiceAlg. All rights reserved.
//

/****************************************************************************
 本模块的功能是，用Frangi滤波来提取粗纹理。本模块由老版算法拆分而来。
 作者：傅晓强
 日期：2022/11/1
 ****************************************************************************/

#ifndef WRINKLE_FRANGI_H
#define WRINKLE_FRANGI_H

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;


void CalcFrgiResp(const Mat& grSrcImg,
                  int scaleRatio,
                  Mat& frangiRespRz);

#endif /* WRINKLE_FRANGI_H */
