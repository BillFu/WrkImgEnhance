//
//  CLAHE.cpp
//  BrownMapExperiment
//
//  Created by meicet on 2022/8/4.
//

#include "CLAHE.hpp"

void ApplyCLAHE(const Mat& inImg,
                int gridSize,
                Mat& outImg)
{
    cv::Ptr<CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    clahe->setTilesGridSize(cv::Size(gridSize, gridSize));
    clahe->apply(inImg, outImg);
}
