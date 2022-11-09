//
//  CLAHE.cpp
//  BrownMapExperiment
//
//  Created by meicet on 2022/8/4.
//

#include "CLAHE.hpp"

void ApplyCLAHE(const Mat& inImg, Mat& outImg)
{
    cv::Ptr<CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    clahe->setTilesGridSize(cv::Size(40, 40));
    clahe->apply(inImg, outImg);
}
