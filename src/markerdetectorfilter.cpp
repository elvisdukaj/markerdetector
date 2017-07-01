#include "markerdetectorfilter.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <QDebug>
#include <boost/crc.hpp>
#include <bitset>
#include <stdexcept>
#include <string>
#include <algorithm>

using namespace std;
using namespace string_literals;

float perimeter(const std::vector<cv::Point2f>& a)
{
    float sum=0, dx, dy;

    for (size_t i=0;i<a.size();i++)
    {
        size_t i2=(i+1) % a.size();

        dx = a[i].x - a[i2].x;
        dy = a[i].y - a[i2].y;

        sum += sqrt(dx*dx + dy*dy);
    }

    return sum;
}

QVideoFilterRunnable* MarkerDetectorFilter::createFilterRunnable()
{
    return new MarkerDetectorFilterRunnable(this);
}

MarkerDetectorFilterRunnable::MarkerDetectorFilterRunnable(MarkerDetectorFilter* filter)
    : m_filter{filter}
{
}

QVideoFrame MarkerDetectorFilterRunnable::run(QVideoFrame* frame, const QVideoSurfaceFormat&, QVideoFilterRunnable::RunFlags)
{
    if (!isFrameValid(frame))
    {
        qDebug() << "Frame is NOT valid";
        return QVideoFrame{};
    }

    if (!frame->map(QAbstractVideoBuffer::ReadWrite))
    {
        qDebug() << "Unable to map the videoframe in memory" << endl;
        return *frame;
    }

    try
    {
        cv::Mat frameMat, grayscale;
        videoFrameInGrayScaleAndColor(frame, grayscale, frameMat);

        m_marksDetector.processFame(grayscale);

        if (!m_marksDetector.markers().empty())
        {
            for(const Marker& marker : m_marksDetector.markers())
                marker.drawContours(frameMat, cv::Scalar{0, 255, 0});

            auto idStr = to_string(m_marksDetector.markers()[0].id());
            emit m_filter->markerFound(QString::fromStdString(idStr));
        }
    }
    catch(const std::exception& exc)
    {
        qDebug() << exc.what();
    }

    frame->unmap();


    return *frame;
}

MarksDetector::MarksDetector()
    : m_markerSize{240, 240}
{
    m_markerCorners2d.push_back(cv::Point2f(0,0));
    m_markerCorners2d.push_back(cv::Point2f(m_markerSize.width,0));
    m_markerCorners2d.push_back(cv::Point2f(m_markerSize.width,m_markerSize.height));
    m_markerCorners2d.push_back(cv::Point2f(0,m_markerSize.height));
}

void MarksDetector::processFame(cv::Mat& grayscale)
{
    m_contours.clear();
    m_possibleContours.clear();
    m_markers.clear();

    m_minCountournSize = grayscale.cols / 5;

    binarize(grayscale);
    findContours();
    findCandidates();
    recognizeCandidates();
}

const std::vector<Marker> &MarksDetector::markers() const noexcept
{
    return m_markers;
}

void MarksDetector::binarize(const cv::Mat& grayscale)
{
    m_grayscale = grayscale;
    cv::threshold(m_grayscale, m_binarized, 127, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
}

void MarksDetector::findContours()
{
    cv::findContours(m_binarized, m_contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
}

void MarksDetector::findCandidates()
{
    vector<cv::Point> approxCurve;
    vector<vector<cv::Point2f>> possibleMarkerPoints;

    // For each contour, analyze if it is a parallelepiped likely to be the marker
    for (size_t i=0; i < m_contours.size(); i++)
    {
        // Approximate to a polygon
        double eps = m_contours[i].size() * 0.05;
        cv::approxPolyDP(m_contours[i], approxCurve, eps, true);

        // We interested only in polygons that contains only four points
        if (approxCurve.size() != 4)
            continue;

        // And they have to be convex
        if (!cv::isContourConvex(approxCurve))
            continue;

        // Ensure that the distance between consecutive points is large enough
        float minDist = std::numeric_limits<float>::max();

        for (int i = 0; i < 4; i++)
        {
            cv::Point side = approxCurve[i] - approxCurve[(i+1)%4];
            float squaredSideLength = side.dot(side);
            minDist = std::min(minDist, squaredSideLength);
        }

        // Check that distance is not very small
        if (minDist < 500)
            continue;

        // All tests are passed. Save marker candidate:
        vector<cv::Point2f> markerPoints;
        markerPoints.reserve(4);

        for (int i = 0; i<4; i++)
            markerPoints.push_back( cv::Point2f(approxCurve[i].x,approxCurve[i].y) );

        // Sort the points in anti-clockwise order
        // Trace a line between the first and second point.
        // If the third point is at the right side, then the points are anti-clockwise
        auto v1 = markerPoints[1] - markerPoints[0];
        auto v2 = markerPoints[2] - markerPoints[0];

        double o = (v1.x * v2.y) - (v1.y * v2.x);

        if (o < 0.0)		 // if the third point is in the left side, then sort in anti-clockwise order
            std::swap(markerPoints[1], markerPoints[3]);

        possibleMarkerPoints.emplace_back(markerPoints);
    }

    // calculate the average distance of each corner to the nearest corner of the other marker candidate
    std::vector< std::pair<int,int> > tooNearCandidates;
    for (int i=0;i<possibleMarkerPoints.size();i++)
    {
        const auto& points1 = possibleMarkerPoints[i];

        //calculate the average distance of each corner to the nearest corner of the other marker candidate
        for (int j=i+1;j<possibleMarkerPoints.size();j++)
        {
            const auto& points2 = possibleMarkerPoints[j];

            float distSquared = 0;

            for (int c = 0; c < 4; c++)
            {
                auto v = points1[c] - points2[c];
                distSquared += v.dot(v);
            }

            distSquared /= 4;

            if (distSquared < 100)
                tooNearCandidates.push_back(std::pair<int,int>(i,j));
        }
    }

    // Mark for removal the element of the pair with smaller perimeter
    std::vector<bool> removalMask (possibleMarkerPoints.size(), false);

    for (size_t i = 0; i < tooNearCandidates.size(); i++)
    {
        float p1 = perimeter(possibleMarkerPoints[tooNearCandidates[i].first ]);
        float p2 = perimeter(possibleMarkerPoints[tooNearCandidates[i].second]);

        size_t removalIndex;
        if (p1 > p2)
            removalIndex = tooNearCandidates[i].second;
        else
            removalIndex = tooNearCandidates[i].first;

        removalMask[removalIndex] = true;
    }

    for (size_t i = 0; i < possibleMarkerPoints.size(); i++)
        if (!removalMask[i])
            m_possibleContours.push_back(possibleMarkerPoints[i]);
}

void MarksDetector::recognizeCandidates()
{
    for(auto& points: m_possibleContours)
    {
        cv::Mat canonicalMarkerImage;

        // Find the perspective transformation that brings current marker to rectangular form
        cv::Mat markerTransform = cv::getPerspectiveTransform(points, m_markerCorners2d);

        // Transform image to get a canonical marker image
        cv::warpPerspective(m_binarized, canonicalMarkerImage,  markerTransform, m_markerSize);

        Marker m(canonicalMarkerImage, points);

        if (!m.isValid())
            continue;

        cv::TermCriteria termCriteria = cv::TermCriteria{cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30, 0.01};
        cv::cornerSubPix(m_grayscale, points, cv::Size{5, 5}, cv::Size{-1, -1}, termCriteria);

        m.precisePoints(points);
        m_markers.push_back(m);
    }
}

void MarksDetector::filterContours()
{
    qDebug() << "First " << m_contours.size();
    auto toRemove = remove_if(begin(m_contours), end(m_contours),
                              [&](const std::vector<cv::Point>& vec)
    {
        return vec.size() > m_minCountournSize;
    }
    );

    m_contours.erase(toRemove, end(m_contours));
    qDebug() << "After" << m_contours.size();
}

Marker::Marker(const cv::Mat& image, const vector<cv::Point2f>& points)
    : m_squareSize{image.size() / 12}
    , m_minArea{m_squareSize.area() / 2}
    , m_isValid{false}
    , m_points{points}
{
    auto orientation = checkFrame(image);

    if (orientation.empty())
        return;

    auto data = checkOrientationFrame(orientation);

    if (data.empty())
        return;

    encodeData(data);
}

void Marker::precisePoints(const std::vector<cv::Point2f>& points) noexcept
{
    m_points = points;
}

void Marker::drawContours(cv::Mat& image, cv::Scalar color) const noexcept
{
    float thickness = 1;

    cv::line(image, m_points[0], m_points[1], color, thickness, CV_AA);
    cv::line(image, m_points[1], m_points[2], color, thickness, CV_AA);
    cv::line(image, m_points[2], m_points[3], color, thickness, CV_AA);
    cv::line(image, m_points[3], m_points[0], color, thickness, CV_AA);
}

cv::Mat Marker::checkFrame(const cv::Mat& image) const noexcept
{
    const cv::Rect topLineRect{
        0, 0,
        image.cols, m_squareSize.height
    };
    const cv::Rect bottomLineRect{
        0, image.rows - m_squareSize.height,
        image.cols, m_squareSize.height
    };
    const cv::Rect leftLineRect{
        0, 0,
        m_squareSize.width, image.rows
    };
    const cv::Rect rightLineRect{
        image.rows - m_squareSize.width, 0,
        m_squareSize.width, image.rows
    };

    const auto topLine = image(topLineRect);
    const auto bottomLine = image(bottomLineRect);
    const auto leftLine = image(leftLineRect);
    const auto rightLine = image(rightLineRect);

    const int squareCount = 12;

    // check top
    for(int i = 0; i < squareCount; ++i)
    {
        auto nonZeros = cv::countNonZero(
            topLine(cv::Rect{
                        m_squareSize.width * i, 0,
                        m_squareSize.width, m_squareSize.height
                }));

        if (nonZeros > m_minArea)
            return cv::Mat{};
    }

    // check bottom
    for(int i = 0; i < squareCount; ++i)
    {
        auto nonZeros = cv::countNonZero(bottomLine(
            cv::Rect{
                    m_squareSize.width * i, 0,
                    m_squareSize.width, m_squareSize.height
            }));

        if (nonZeros > m_minArea)
            return cv::Mat{};
    }

    // check leftLine
    for(int i = 0; i < squareCount; ++i)
    {
        auto nonZeros = cv::countNonZero(leftLine(
            cv::Rect{
                    0, m_squareSize.height * i,
                    m_squareSize.width, m_squareSize.height
            }));

        if (nonZeros > m_minArea)
            return cv::Mat{};
    }

    // check rightLine
    for(int i = 0; i < squareCount; ++i)
    {
        auto nonZeros = cv::countNonZero(rightLine(
            cv::Rect{
                    0, m_squareSize.height * i,
                    m_squareSize.width, m_squareSize.height
            }));

        if (nonZeros > m_minArea)
            return cv::Mat{};
    }

    return image(cv::Rect{
        m_squareSize.width, m_squareSize.height,
        image.cols - (2*m_squareSize.width), image.rows - (2*m_squareSize.height)
                   });
}

cv::Mat Marker::checkOrientationFrame(const cv::Mat& image) const noexcept
{
    cv::Mat rotated;

    const cv::Rect topLineRect{
        0, 0,
        image.cols, m_squareSize.height
    };
    const cv::Rect bottomLineRect{
        0, image.rows - m_squareSize.height,
        image.cols, m_squareSize.height
    };
    const auto topLine = image(topLineRect);
    const auto bottomLine = image(bottomLineRect);

    const int squareCount = 10;
    int whiteSquares = 0;

    // check top
    for(int i = 0; i < squareCount; ++i)
    {
        auto nonZeros = cv::countNonZero(
            topLine(cv::Rect{
                        m_squareSize.width * i, 0,
                        m_squareSize.width, m_squareSize.height
                }));

        if (nonZeros > m_minArea)
            ++whiteSquares;
    }

    // check bottom
    for(int i = 0; i < squareCount; ++i)
    {
        auto nonZeros = cv::countNonZero(bottomLine(
            cv::Rect{
                    m_squareSize.width * i, 0,
                    m_squareSize.width, m_squareSize.height
            }));

        if (nonZeros > m_minArea)
            ++whiteSquares;
    }


    if (whiteSquares == 3)
    {
        cv::Rect topLeftRect{
            0, 0,
            m_squareSize.width, m_squareSize.height
        };
        cv::Rect topRightRect{
            image.cols - m_squareSize.width, 0,
            m_squareSize.width, m_squareSize.height
        };
        cv::Rect bottomLeftRect{
            0, image.rows - m_squareSize.height,
            m_squareSize.width, m_squareSize.height
        };
        cv::Rect bottomRightRect{
            image.cols - m_squareSize.width, image.rows - m_squareSize.height,
            m_squareSize.width, m_squareSize.height
        };

        auto topLeftNonZeros = cv::countNonZero(image(topLeftRect));
        auto topRightNonZeros = cv::countNonZero(image(topRightRect));
        auto bottomLeftNonZeros = cv::countNonZero(image(bottomLeftRect));
        auto bottomRightNonZeros = cv::countNonZero(image(bottomRightRect));

        auto bottomRight = bottomRightNonZeros > m_minArea ? 1 : 0;
        auto topLeft = topLeftNonZeros > m_minArea ? 2 : 0;
        auto topRight = topRightNonZeros > m_minArea ? 4 : 0;
        auto bottomLeft = bottomLeftNonZeros > m_minArea ? 8 : 0;

        int rotation = bottomLeft | topLeft | topRight | bottomRight;
//        qDebug() << "rotation code is: " << rotation;

        switch (rotation) {
        case 7:
            cv::flip(image, rotated, 1);
            break;

        case 13:
            cv::rotate(image, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
            cv::flip(rotated, rotated, 1);
            break;

        case 11:
            cv::flip(image, rotated, 0);
            break;

        case 14:
            cv::rotate(image, rotated, cv::ROTATE_90_CLOCKWISE);
            cv::flip(rotated, rotated, 1);
            break;

        default:
            return cv::Mat{};
        }

        return rotated(cv::Rect{
            m_squareSize.width, m_squareSize.height,
            image.cols - (2*m_squareSize.width), image.rows - (2*m_squareSize.height)
            });

    }
    else
        return cv::Mat{};
}

void Marker::encodeData(const cv::Mat& dataImage)
{
    const auto& onlyDataImage = dataImage(
                                    cv::Rect{
                                        0, 0,
                                        dataImage.cols, m_squareSize.height * 6
                                    }
                                    );

    bitset<6*8> dataBits;

    for (int i = 0; i < 6; ++i)
    {
        cv::Rect lineRect{0, m_squareSize.height * i, dataImage.cols, m_squareSize.height};
        const auto& line =  onlyDataImage(lineRect);

        for (int j = 0; j < 8; ++j)
        {
            cv::Rect square{m_squareSize.width * j, 0, m_squareSize.width, m_squareSize.height};
            auto bit = cv::countNonZero(line(square)) > m_minArea ? 1 : 0;
            dataBits[i*8 + j] = bit;
        }
    }

    m_id = dataBits.to_ullong();

    const auto& onlyCRCImage = dataImage(
                                   cv::Rect{
                                       0, m_squareSize.height * 6,
                                       dataImage.cols, m_squareSize.height * 2
                                   });

    bitset<2*8> crcBits;

    for (int i = 0; i < 2; ++i)
    {
        cv::Rect lineRect{0, m_squareSize.height * i, onlyCRCImage.cols, m_squareSize.height};
        const auto& line =  onlyCRCImage(lineRect);

        for (int j = 0; j < 8; ++j)
        {
            cv::Rect square{m_squareSize.width * j, 0, m_squareSize.width, m_squareSize.height};
            auto bit = cv::countNonZero(line(square)) > m_minArea ? 1 : 0;
            crcBits[i*8 + j] = bit;
        }
    }

    boost::crc_16_type crc;
    crc.process_bytes(&m_id, sizeof(m_id));

    if (crcBits.to_ullong() != crc.checksum())
    {
        qDebug() << "CRC Mismatch found " << dataBits.to_ullong() << " with crc "
                 << crcBits.to_ullong() << " calculated " << crc.checksum();
        return;
    }

    m_isValid = true;
}
