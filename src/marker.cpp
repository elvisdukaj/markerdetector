#include "marker.h"
#include <boost/crc.hpp>
#include <opencv2/imgproc.hpp>
#include <bitset>
#include <iostream>

using namespace cv;
using namespace std;

Marker::Marker(const Mat& image, const vector<Point2f>& points)
    : m_squareSize{image.size() / 12}
    , m_minArea{m_squareSize.area() / 2}
    , m_isValid{false}
    , m_points{points}
    , m_color{Scalar::all(255)}
{
    auto orientation = checkFrame(image);

    if (orientation.empty())
        return;

    auto data = checkOrientationFrame(orientation);

    if (data.empty())
        return;

    encodeData(data);
}

void Marker::precisePoints(const std::vector<Point2f>& points) noexcept
{
    m_points = points;
}

void Marker::drawContours(Mat& image, int thickness) const noexcept
{
    line(image, m_points[0], m_points[1], m_color, thickness, CV_AA);
    line(image, m_points[1], m_points[2], m_color, thickness, CV_AA);
    line(image, m_points[2], m_points[3], m_color, thickness, CV_AA);
    line(image, m_points[3], m_points[0], m_color, thickness, CV_AA);

    for(const auto& line2d : m_cube)
        line(image, line2d[0], line2d[1], m_color, thickness, CV_AA);
}

Mat Marker::checkFrame(const Mat& image) const noexcept
{
    const Rect topLineRect{
        0, 0,
        image.cols, m_squareSize.height
    };
    const Rect bottomLineRect{
        0, image.rows - m_squareSize.height,
        image.cols, m_squareSize.height
    };
    const Rect leftLineRect{
        0, 0,
        m_squareSize.width, image.rows
    };
    const Rect rightLineRect{
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
        auto nonZeros = countNonZero(
            topLine(Rect{
                        m_squareSize.width * i, 0,
                        m_squareSize.width, m_squareSize.height
                }));

        if (nonZeros > m_minArea)
            return Mat{};
    }

    // check bottom
    for(int i = 0; i < squareCount; ++i)
    {
        auto nonZeros = countNonZero(bottomLine(
            Rect{
                    m_squareSize.width * i, 0,
                    m_squareSize.width, m_squareSize.height
            }));

        if (nonZeros > m_minArea)
            return Mat{};
    }

    // check leftLine
    for(int i = 0; i < squareCount; ++i)
    {
        auto nonZeros = countNonZero(leftLine(
            Rect{
                    0, m_squareSize.height * i,
                    m_squareSize.width, m_squareSize.height
            }));

        if (nonZeros > m_minArea)
            return Mat{};
    }

    // check rightLine
    for(int i = 0; i < squareCount; ++i)
    {
        auto nonZeros = countNonZero(rightLine(
            Rect{
                    0, m_squareSize.height * i,
                    m_squareSize.width, m_squareSize.height
            }));

        if (nonZeros > m_minArea)
            return Mat{};
    }

    return image(Rect{
        m_squareSize.width, m_squareSize.height,
        image.cols - (2*m_squareSize.width), image.rows - (2*m_squareSize.height)
                   });
}

Mat Marker::checkOrientationFrame(const Mat& image) const noexcept
{
    Mat rotated;

    const Rect topLineRect{
        0, 0,
        image.cols, m_squareSize.height
    };
    const Rect bottomLineRect{
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
        auto nonZeros = countNonZero(
            topLine(Rect{
                        m_squareSize.width * i, 0,
                        m_squareSize.width, m_squareSize.height
                }));

        if (nonZeros > m_minArea)
            ++whiteSquares;
    }

    // check bottom
    for(int i = 0; i < squareCount; ++i)
    {
        auto nonZeros = countNonZero(bottomLine(
            Rect{
                    m_squareSize.width * i, 0,
                    m_squareSize.width, m_squareSize.height
            }));

        if (nonZeros > m_minArea)
            ++whiteSquares;
    }


    if (whiteSquares == 3)
    {
        Rect topLeftRect{
            0, 0,
            m_squareSize.width, m_squareSize.height
        };
        Rect topRightRect{
            image.cols - m_squareSize.width, 0,
            m_squareSize.width, m_squareSize.height
        };
        Rect bottomLeftRect{
            0, image.rows - m_squareSize.height,
            m_squareSize.width, m_squareSize.height
        };
        Rect bottomRightRect{
            image.cols - m_squareSize.width, image.rows - m_squareSize.height,
            m_squareSize.width, m_squareSize.height
        };

        auto topLeftNonZeros = countNonZero(image(topLeftRect));
        auto topRightNonZeros = countNonZero(image(topRightRect));
        auto bottomLeftNonZeros = countNonZero(image(bottomLeftRect));
        auto bottomRightNonZeros = countNonZero(image(bottomRightRect));

        auto bottomRight = bottomRightNonZeros > m_minArea ? 1 : 0;
        auto topLeft = topLeftNonZeros > m_minArea ? 2 : 0;
        auto topRight = topRightNonZeros > m_minArea ? 4 : 0;
        auto bottomLeft = bottomLeftNonZeros > m_minArea ? 8 : 0;

        int rotation = bottomLeft | topLeft | topRight | bottomRight;
//        qDebug() << "rotation code is: " << rotation;

        switch (rotation) {
        case 7:
            flip(image, rotated, 1);
            break;

        case 13:
            rotate(image, rotated, ROTATE_90_COUNTERCLOCKWISE);
            flip(rotated, rotated, 1);
            break;

        case 11:
            flip(image, rotated, 0);
            break;

        case 14:
            rotate(image, rotated, ROTATE_90_CLOCKWISE);
            flip(rotated, rotated, 1);
            break;

        default:
            return Mat{};
        }

        return rotated(Rect{
            m_squareSize.width, m_squareSize.height,
            image.cols - (2*m_squareSize.width), image.rows - (2*m_squareSize.height)
            });

    }
    else
        return Mat{};
}

void Marker::encodeData(const Mat& dataImage)
{
    const auto& onlyDataImage = dataImage(
                                    Rect{
                                        0, 0,
                                        dataImage.cols, m_squareSize.height * 6
                                    }
                                    );

    bitset<6*8> dataBits;

    for (int i = 0; i < 6; ++i)
    {
        Rect lineRect{0, m_squareSize.height * i, dataImage.cols, m_squareSize.height};
        const auto& line =  onlyDataImage(lineRect);

        for (int j = 0; j < 8; ++j)
        {
            Rect square{m_squareSize.width * j, 0, m_squareSize.width, m_squareSize.height};
            auto bit = countNonZero(line(square)) > m_minArea ? 1 : 0;
            dataBits[i*8 + j] = bit;
        }
    }

    m_id = dataBits.to_ullong();

    const auto& onlyCRCImage = dataImage(
                                   Rect{
                                       0, m_squareSize.height * 6,
                                       dataImage.cols, m_squareSize.height * 2
                                   });

    bitset<2*8> crcBits;

    for (int i = 0; i < 2; ++i)
    {
        Rect lineRect{0, m_squareSize.height * i, onlyCRCImage.cols, m_squareSize.height};
        const auto& line =  onlyCRCImage(lineRect);

        for (int j = 0; j < 8; ++j)
        {
            Rect square{m_squareSize.width * j, 0, m_squareSize.width, m_squareSize.height};
            auto bit = countNonZero(line(square)) > m_minArea ? 1 : 0;
            crcBits[i*8 + j] = bit;
        }
    }

    boost::crc_16_type crc;
    crc.process_bytes(&m_id, sizeof(m_id));

    if (crcBits.to_ullong() != crc.checksum())
    {
        cerr << "CRC Mismatch found " << dataBits.to_ullong() << " with crc "
                 << crcBits.to_ullong() << " calculated " << crc.checksum();
        return;
    }

    auto r = (m_id & 0x0000ff) >> 0;
    auto g = (m_id & 0x00ff00) >> 8;
    auto b = (m_id & 0xff0000) >> 16;

    m_color = Scalar(r, g, b);
    m_isValid = true;
}
