#ifndef KALMANTRACKER_H_
#define KALMANTRACKER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>
namespace tld
{
    class KalmanTracker
    {
    public:
        int stateSize = 6;
        int measSize = 4;
        int contrSize = 0;
        unsigned int type = CV_32F;
        bool initialized;
        cv::Rect *kalmanBB;
        cv::KalmanFilter kf;
        cv::Mat state;
        cv::Mat meas;
        double ticks;

        KalmanTracker();
        virtual ~KalmanTracker();
        void init(cv::Rect *prevBB);
        void release();
        void track(const cv::Mat &currImg, cv::Rect *prevBB);
        void update(const cv::Rect *bb);
         

    };
}


#endif