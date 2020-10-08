/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/

/*
 * TLD.h
 *
 *  Created on: Nov 17, 2011
 *      Author: Georg Nebehay
 */

#ifndef TLD_H_
#define TLD_H_

#include <opencv/cv.h>

#include "MedianFlowTracker.h"
#include "DetectorCascade.h"
//#include "KalmanTracker.h"

namespace tld
{
class Metrics
{
    public:
        //centroids are stored
        std::vector<int> misses;
        std::vector<int> falsePostives;
        std::vector<int> mismatches;
        std::vector<int> nmatches; //matches for frame t
        std::vector<double> distances; //sum of distances for frame t
        std::vector<char> matches;

        //precision on object position
        //double motop = 0.0;
        
        //accuracy over time
        //double mota = 0.0;
        
        //correspondence
        double threshold =  500;


    Metrics(int n);
    virtual ~Metrics();
    //euclidean distance between the two centroids
    double distanceCalculate(double x1, double y1, double x2, double y2);
    void processFrame(int i, cv::Rect* object, cv::Rect* hypotesisMFT, cv::Rect* hypotesisKalman);
    double mota(int n);
    double motp(int n);



};

class KalmanTracker
    {
    public:
        int stateSize = 6;
        int measSize = 4;
        int contrSize = 0;
        unsigned int type = CV_32F;
        
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

class TLD
{
    void storeCurrentData();
    void fuseHypotheses();
    void learn();
    void initialLearning();
public:
    bool trackerEnabled;
    bool detectorEnabled;
    bool learningEnabled;
    bool alternating;

    MedianFlowTracker *medianFlowTracker;
    DetectorCascade *detectorCascade;
    NNClassifier *nnClassifier;
    KalmanTracker *kalmanTracker;
    bool valid;
    bool wasValid;
    cv::Mat prevImg;
    cv::Mat currImg;
    cv::Rect *prevBB;
    cv::Rect *currBB;
    float currConf;
    bool learning;

    TLD();
    virtual ~TLD();
    void release();
    void selectObject(const cv::Mat &img, cv::Rect *bb);
    void processImage(const cv::Mat &img);
    void writeToFile(const char *path);
    void readFromFile(const char *path);
};

} /* namespace tld */
#endif /* TLD_H_ */
