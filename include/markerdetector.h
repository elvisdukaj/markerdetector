// Copyright (c) 2017 Elvis Dukaj
// 
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
// 
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "marker.h"

class MarksDetector {
public:
    MarksDetector();

    void processFame(cv::Mat& grayscale);
    uint64_t encode() const;

    const std::vector<Marker>& markers() const noexcept;

private:
    void binarize(const cv::Mat &grayscale);
    void findContours();
    void findCandidates();
    void recognizeCandidates();
    void estimatePose();

    void applyImage(const cv::Mat& image);

private:
    int m_minCountournSize;
    uint64_t m_id;
    cv::Mat m_grayscale;
    cv::Mat m_binarized;
    std::vector<std::vector<cv::Point>> m_contours;
    std::vector<std::vector<cv::Point2f>> m_possibleContours;

    const cv::Size m_markerSize;
    std::vector<cv::Point2f> m_markerCorners2d;
    std::vector<Marker> m_markers;

    cv::Mat m_distortion;
    cv::Mat m_cameraMatrix;
};
