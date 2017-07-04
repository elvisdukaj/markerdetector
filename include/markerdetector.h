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
