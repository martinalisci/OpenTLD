#include "KalmanTracker.h"

using namespace cv;

namespace tld
{
    KalmanTracker::KalmanTracker()
    {
        kalmanBB = NULL;
        kf = cv::KalmanFilter(stateSize, measSize, contrSize, type);
        state = cv::Mat(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
        meas = cv::Mat(measSize, 1, type);    // [z_x,z_y,z_w,z_h]

    }
    KalmanTracker::~KalmanTracker(){
        delete kalmanBB;
        kalmanBB = NULL;
        
    }

    void KalmanTracker::release()
    {
        delete kalmanBB;
        kalmanBB = NULL;
        
    }

    void KalmanTracker::init(cv::Rect *prevBB){
            
        //cv::Mat procNoise(stateSize, 1, type)
        // [E_x,E_y,E_v_x,E_v_y,E_w,E_h]
    
        // Transition State Matrix A
        // Note: set dT at each processing step!
        // [ 1 0 dT 0  0 0 ]
        // [ 0 1 0  dT 0 0 ]
        // [ 0 0 1  0  0 0 ]
        // [ 0 0 0  1  0 0 ]
        // [ 0 0 0  0  1 0 ]
        // [ 0 0 0  0  0 1 ]
        cv::setIdentity(kf.transitionMatrix);
    
        // Measure Matrix H
        // [ 1 0 0 0 0 0 ]
        // [ 0 1 0 0 0 0 ]
        // [ 0 0 0 0 1 0 ]
        // [ 0 0 0 0 0 1 ]
        kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
        kf.measurementMatrix.at<float>(0) = 1.0f;
        kf.measurementMatrix.at<float>(7) = 1.0f;
        kf.measurementMatrix.at<float>(16) = 1.0f;
        kf.measurementMatrix.at<float>(23) = 1.0f;
    
        // Process Noise Covariance Matrix Q
        // [ Ex   0   0     0     0    0  ]
        // [ 0    Ey  0     0     0    0  ]
        // [ 0    0   Ev_x  0     0    0  ]
        // [ 0    0   0     Ev_y  0    0  ]
        // [ 0    0   0     0     Ew   0  ]
        // [ 0    0   0     0     0    Eh ]
        //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
        kf.processNoiseCov.at<float>(0) = 1e-2;
        kf.processNoiseCov.at<float>(7) = 1e-2;
        kf.processNoiseCov.at<float>(14) = 5.0f;
        kf.processNoiseCov.at<float>(21) = 5.0f;
        kf.processNoiseCov.at<float>(28) = 1e-2;
        kf.processNoiseCov.at<float>(35) = 1e-2;
    
        // Measures Noise Covariance Matrix R
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
        // <<<< Kalman Filter
        ticks = 0;
        initialized = true;

        double precTick = ticks;
        ticks = (double) cv::getTickCount();
        double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds

        meas.at<float>(0) =prevBB->x + prevBB->width / 2;
        meas.at<float>(1) = prevBB->y + prevBB->height / 2;
        meas.at<float>(2) = (float)prevBB->width;
        meas.at<float>(3) = (float)prevBB->height;

        // >>>> Initialization
        kf.errorCovPre.at<float>(0) = 1; // px
        kf.errorCovPre.at<float>(7) = 1; // px
        kf.errorCovPre.at<float>(14) = 1;
        kf.errorCovPre.at<float>(21) = 1;
        kf.errorCovPre.at<float>(28) = 1; // px
        kf.errorCovPre.at<float>(35) = 1; // px

        state.at<float>(0) = meas.at<float>(0);
        state.at<float>(1) = meas.at<float>(1);
        state.at<float>(2) = 0;
        state.at<float>(3) = 0;
        state.at<float>(4) = meas.at<float>(2);
        state.at<float>(5) = meas.at<float>(3);
        // <<<< Initialization

        kf.statePost = state;

    }

    void KalmanTracker::track(const cv::Mat &currImg, cv::Rect *prevBB)
    {
        double precTick = ticks;
        ticks = (double) cv::getTickCount();
        double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds

        // >>>> Matrix A
        kf.transitionMatrix.at<float>(2) = dT;
        kf.transitionMatrix.at<float>(9) = dT;
        // <<<< Matrix A

        //cout << "dT:" << endl << dT << endl;

        state = kf.predict();
        //cout << "State post:" << endl << state << endl;
     
        kalmanBB->width = state.at<float>(4);
        kalmanBB->height = state.at<float>(5);
        kalmanBB->x = state.at<float>(0) - kalmanBB->width / 2;
        kalmanBB->y = state.at<float>(1) - kalmanBB->height / 2;

    }

    void KalmanTracker::update(const cv::Rect *bb)
    {
        meas.at<float>(0) = bb->x + bb->width / 2;
        meas.at<float>(1) = bb->y + bb->height / 2;
        meas.at<float>(2) = bb->width;
        meas.at<float>(3) = bb->height;
        kf.correct(meas);
    }

}