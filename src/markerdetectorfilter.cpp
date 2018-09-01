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

#include "markerdetectorfilter.h"
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace string_literals;

QVideoFilterRunnable* MarkerDetectorFilter::createFilterRunnable()
{
    return new MarkerDetectorFilterRunnable(this);
}

MarkerDetectorFilterRunnable::MarkerDetectorFilterRunnable(MarkerDetectorFilter* filter)
try : m_filter{filter}
{
//    m_pattern = cv::imread("pattern.bmp", cv::IMREAD_COLOR);

//    if (!m_pattern.data)
//    {
//        cerr << "Unable to open pattern.bmp";
//        throw runtime_error{"Unable to open pattern.bmp"};
//    }
}
catch(const runtime_error& err)
{
    std::cerr << err.what() << std::endl;
    throw;
}

QVideoFrame MarkerDetectorFilterRunnable::run(QVideoFrame* frame, const QVideoSurfaceFormat&, QVideoFilterRunnable::RunFlags)
{
    if (!isFrameValid(frame))
    {
        cerr << "Frame is NOT valid" << endl;
        return QVideoFrame{};
    }

    if (!frame->map(QAbstractVideoBuffer::ReadWrite))
    {
        cerr << "Unable to map the videoframe in memory" << endl;
        return *frame;
    }

    try
    {
        cv::Mat frameMat, grayscale;
        videoFrameInGrayScaleAndColor(frame, grayscale, frameMat);

        m_marksDetector.processFame(grayscale);


        string idStr;
        if (!m_marksDetector.markers().empty())
        {
            for(const Marker& marker : m_marksDetector.markers())
            {
                idStr += to_string(marker.id()) + " "s;
//              marker.drawImage(frameMat, m_pattern);
                marker.drawContours(frameMat, 3);
            }
        }

        emit m_filter->markerFound(QString::fromStdString(idStr));
    }
    catch(const exception& exc)
    {
        cerr << exc.what() << endl;
    }

    frame->unmap();


    return *frame;
}
