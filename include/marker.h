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

#include <opencv2/core.hpp>
#include <vector>
#include <cstdint>

class Marker {
public:
    Marker(const cv::Mat& image, const std::vector<cv::Point2f>& points);

    bool isValid() const noexcept { return m_isValid; }
    uint64_t id() const noexcept { return m_id; }
    const std::vector<cv::Point2f>& points() const noexcept { return m_points; }
    void precisePoints(const std::vector<cv::Point2f>& points) noexcept;
    void drawContours(cv::Mat& image, int thickness) const noexcept;

    void setCube(std::vector<std::vector<cv::Point2f>>& cube) { m_cube = cube; }
    void drawImage(cv::Mat& frame, const cv::Mat& image) const;

private:
    cv::Mat checkFrame(const cv::Mat& image) const noexcept;
    cv::Mat checkOrientationFrame(const cv::Mat& orientation) const noexcept;
    void encodeData(const cv::Mat& dataImage);

private:
    const cv::Size m_markerSize;
    const cv::Size m_squareSize;
    const int m_minArea;
    bool m_isValid;
    std::vector<cv::Point2f> m_undistortedPoints;
    std::vector<cv::Point2f> m_points;
    std::vector<std::vector<cv::Point2f>> m_cube;
    cv::Scalar m_color;
    uint64_t m_id;
};
