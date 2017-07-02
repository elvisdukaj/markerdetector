#include "markerdetectorfilter.h"
#include <iostream>

using namespace std;
using namespace string_literals;

QVideoFilterRunnable* MarkerDetectorFilter::createFilterRunnable()
{
    return new MarkerDetectorFilterRunnable(this);
}

MarkerDetectorFilterRunnable::MarkerDetectorFilterRunnable(MarkerDetectorFilter* filter)
try : m_filter{filter}, m_marksDetector{}
{
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
