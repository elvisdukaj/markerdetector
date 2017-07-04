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
