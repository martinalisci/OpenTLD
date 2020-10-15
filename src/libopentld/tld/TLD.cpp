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
 * TLD.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: Georg Nebehay
 */

#include "TLD.h"

#include <iostream>

#include "NNClassifier.h"
#include "TLDUtil.h"

using namespace std;
using namespace cv;

namespace tld
{
//*****************************************Metrica
Metrics::Metrics()
{
    count = 0;
    misses = std::vector<int>(1000,0);
    falsePostives = std::vector<int>(1000,0);
    mismatches = std::vector<int>(1000,0);
    nmatches = std::vector<int>(1000,0);
    distances = std::vector<float>(1000,0);
    matches = std::vector<int>(1000,0);
}

Metrics::~Metrics()
{
    misses.clear();
    falsePostives.clear();
    mismatches.clear();
    nmatches.clear(); //matches for frame t
    distances.clear(); //sum of distances for frame t
    matches.clear();
}


float Metrics::distanceCalculate(float x1, float y1, float x2, float y2)
{
	float x = x1 - x2; //calculating number to square in next step
	float y = y1 - y2;
	float dist=0.0;
	dist = pow(x, 2) + pow(y, 2);     //calculating Euclidean distance
	dist = sqrt(dist);        
    
	return dist;
}

void Metrics::processFrame(cv::Rect object, cv::Rect hypothesisMFT, cv::Rect hypothesisKalman)
{
    std::cout<<"in process Frame"<<std::endl;
    double x_prev = object.x + object.width / 2;
    double y_prev = object.y + object.height / 2;
    double x_tr = hypothesisMFT.x + hypothesisMFT.width / 2;
    double y_tr = hypothesisMFT.y + hypothesisMFT.height / 2;
    double x_k = hypothesisKalman.x + hypothesisKalman.width / 2;
    double y_k = hypothesisKalman.y + hypothesisKalman.height / 2;
    
    float distMFT = distanceCalculate(x_prev,y_prev,x_tr,y_tr);
    float distK = distanceCalculate(x_prev,y_prev,x_k,y_k);
    
    int result = 0;
    char str1;
    char str2;
    //1 mft 2 k
    std::cout<<"ok qui"<<std::endl;

    if (distMFT < threshold)
    {
        if(distMFT<= distK )
        {
            matches[count] =1;
            nmatches[count]=1;

            if (count>1)
            {
                str1 = matches[count];
                str2 = matches[count-1];
                result = strcmp(&str1,&str2);
                if (result!=0)
                {
                    mismatches[count]=1;
                }
                else
                {
                    mismatches[count]=0;
                }
            }
            else
            {
                mismatches[count] = 0;
            }

            if( distK < threshold)
            {
                falsePostives[count]=1;
                distances[count] = distK+distMFT;
            }
            else
            {
                falsePostives[count]=0;
                distances[count] = distMFT;
            }
            misses[count]=0;
            
        }
        //dist mft < 500 ma mft>k quindi ok
        else 
        {
            matches[count]=2;
            nmatches[count]=1;
            
            if (count>1)
            {
                str1 = matches[count];
                str2 = matches[count-1];
                result = strcmp(&str1,&str2);
                if (result!=0)
                {
                    mismatches[count]=1;
                }
                else
                {
                    mismatches[count]=0;
                }
        
            }
            else
            {
                mismatches[count]=0; 
            }
            falsePostives[count] = 1;
            distances[count] = distK+distMFT;
            misses[count]=0;
        } 
    }
    else if(distK < threshold)
    {
        matches[count]=2;
        nmatches[count]=1;
            
        if (count>1)
        {
            str1 = matches[count];
            str2 = matches[count-1];
            result = strcmp(&str1,&str2);
            if (result!=0)
            {
                mismatches[count]=1;
            }
            else
            {
                mismatches[count]=0;
            }
        }
        else
        {
            mismatches[count]=0; 
        }
        falsePostives[count] = 0;
        distances[count] = distK;
        misses[count]=0;
        
    }
    else
    {
        //entrambi > threshold
        misses[count] = 1;
        falsePostives[count] = 0;
        mismatches[count] = 0;
        nmatches[count] = 0; //matches for frame t
        distances[count] = 0; //sum of distances for frame t

    }
    
    count++;
}

float Metrics::motp()
{
    int i = 0;
    double sumDistances = 0;
    double sumMatches = 0;

    for(i=0; i<count; i++)
    {
        sumDistances += distances[i];
        sumMatches += nmatches[i];
	}
    return (sumDistances/double(count));
}

float Metrics::mota()
{
    int i = 0;
    double sumMisses = 0;
    double sumFp = 0;
    double sumMismatches = 0;
    double sum = 0;

    for (i=0;i<count;i++)
    {
        sumMisses += misses[i];
        sumFp += falsePostives[i];
        sumMismatches += mismatches[i];
    }
    sum = sumMismatches+sumFp+sumMisses;

    return (1-((sum)/double(count)));
}

//********************************************KalmanTracker
KalmanTracker::KalmanTracker()
    {
        kalmanBB =  NULL;
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

    kalmanBB = new Rect(0, 0, 0, 0);
    kalmanBB->width = prevBB->width;
    kalmanBB->height = prevBB->height;
    //kalmanBB->x = prevBB->x - prevBB->width / 2;
    //kalmanBB->y = prevBB->y - prevBB->height / 2;
    kalmanBB->x = prevBB->x;
    kalmanBB->y = prevBB->y;
    ticks = 0;
    
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

    double precTick = ticks;
    ticks = (double) cv::getTickCount();
    double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds

    meas.at<float>(0) =prevBB->x + prevBB->width / 2;
    meas.at<float>(1) = prevBB->y + prevBB->height / 2;
    meas.at<float>(2) = (float)prevBB->width;
    meas.at<float>(3) = (float)prevBB->height;
    //meas.at<float>(0) =prevBB->x;
    //meas.at<float>(1) = prevBB->y;
    //meas.at<float>(2) = prevBB->width + prevBB->x -1;
    //meas.at<float>(3) = prevBB->height + prevBB->y -1;

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

    //DEBUG
    //printf("ok init kalman\n");
    //FINE
}

void KalmanTracker::track(const cv::Mat &currImg, cv::Rect *prevBB)
{
    //DEBUG
    //printf("in kalman track\n");
    //FINE
    if(prevBB->width <= 0 || prevBB->height <= 0)
    {
        return;
    }

    double precTick = ticks;
    //ticks = (double) cv::getTickCount();
    double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds

    // >>>> Matrix A
    kf.transitionMatrix.at<float>(2) = dT;
    kf.transitionMatrix.at<float>(9) = dT;
    // <<<< Matrix A

    //cout << "dT:" << endl << dT << endl;

    state = kf.predict();
    //cout << "State post:" << endl << state << endl;
    //cout << typeid(kalmanBB->width).name() << endl;
    //cout << typeid(state.at<float>(4)).name() << endl;
    //cout <<"state at 4 " <<(state.at<float>(4)) << endl;
    //cout <<"width"<< kalmanBB->width << endl;
    //cout << "state at 5"<<state.at<float>(5) << endl;
    //cout <<"height"<< kalmanBB->height << endl;

    std::cout <<"prev bb w "<< prevBB->width << endl;
    std::cout <<"prev bb h "<< prevBB->height << endl;
    std::cout <<"prev bb x "<< prevBB->x << endl;
    std::cout <<"prev bb y "<< prevBB->y << endl;

    float w = floor(state.at<float>(4));
    float h = floor(state.at<float>(5));
    float x = floor(state.at<float>(0) - w /2);
    float y = floor(state.at<float>(1) - h /2);

    if(x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > currImg.cols || y + h > currImg.rows || x != x || y != y || w != w || h != h) //x!=x is check for nan
    {
        //Leave it empty
        //printf("nada\n");
        
    }
    else
    {
        kalmanBB = new Rect(x, y, w, h);
            //DEBUG
        printf("ok kalman track\n");
        //FINE
    }

    //cout <<"height"<< h << "\n" << endl;
    //cout <<"width"<< w << "\n" << endl;
    //cout <<"x" << x << "\n"<< endl;
    //cout << "y" << y << "\n" << endl;

    
}

void KalmanTracker::update(const cv::Rect *bb)
{
    meas.at<float>(0) = bb->x + bb->width / 2;
    meas.at<float>(1) = bb->y + bb->height / 2;
    meas.at<float>(2) = bb->width;
    meas.at<float>(3) = bb->height;
    kf.correct(meas);
}

//********************************************TLD

TLD::TLD()
{
    trackerEnabled = true;
    detectorEnabled = true;
    learningEnabled = true;
    alternating = false;
    valid = false;
    wasValid = false;
    learning = false;
    currBB = NULL;
    prevBB = new Rect(0,0,0,0);

    detectorCascade = new DetectorCascade();
    nnClassifier = detectorCascade->nnClassifier;
    kalmanTracker = new KalmanTracker();
    medianFlowTracker = new MedianFlowTracker();

    metric = Metrics();
}

TLD::~TLD()
{
    storeCurrentData();

    if(currBB)
    {
        delete currBB;
        currBB = NULL;
    }

    if(detectorCascade)
    {
        delete detectorCascade;
        detectorCascade = NULL;
    }

    if(medianFlowTracker)
    {
        delete medianFlowTracker;
        medianFlowTracker = NULL;
    }

    if(kalmanTracker)
    {
        delete kalmanTracker;
        kalmanTracker = NULL;
    }

    if(prevBB)
    {
        delete prevBB;
        prevBB = NULL;
    }
    //metric.~Metrics();
}

void TLD::release()
{
    detectorCascade->release();
    medianFlowTracker->cleanPreviousData();
    kalmanTracker->release();

    if(currBB)
    {
        delete currBB;
        currBB = NULL;
    }
}

void TLD::storeCurrentData()
{
    prevImg.release();
    prevImg = currImg; //Store old image (if any)
    if(currBB)//Store old bounding box (if any)
    {
        prevBB->x = currBB->x;
        prevBB->y = currBB->y;
        prevBB->width = currBB->width;
        prevBB->height = currBB->height;
    }
    else
    {
        prevBB->x = 0;
        prevBB->y = 0;
        prevBB->width = 0;
        prevBB->height = 0;
    }

    detectorCascade->cleanPreviousData(); //Reset detector results
    medianFlowTracker->cleanPreviousData();

    wasValid = valid;
}

void TLD::selectObject(const Mat &img, Rect *bb)
{
    //Delete old object
    detectorCascade->release();

    detectorCascade->objWidth = bb->width;
    detectorCascade->objHeight = bb->height;

    //Init detector cascade
    detectorCascade->init();
    kalmanTracker->init(bb);

    currImg = img;
    if(currBB)
    {
        delete currBB;
        currBB = NULL;
    }
    currBB = tldCopyRect(bb);
    currConf = 1;
    valid = true;

    initialLearning();

}

void TLD::processImage(const Mat &img)
{
   
    storeCurrentData();
     
    Mat grey_frame;
    cvtColor(img, grey_frame, CV_BGR2GRAY);
    currImg = grey_frame; // Store new image , right after storeCurrentData();

    if(trackerEnabled)
    {
        medianFlowTracker->track(prevImg, currImg, prevBB);
    }

    if(detectorEnabled && (!alternating || medianFlowTracker->trackerBB == NULL))
    {
        detectorCascade->detect(grey_frame);
    }
   
    kalmanTracker->ticks = (double) cv::getTickCount();
    kalmanTracker->track(currImg,prevBB);
   
    fuseHypotheses();

    learn();
}

void TLD::fuseHypotheses()
{
    Rect *trackerBB = medianFlowTracker->trackerBB;
    int numClusters = detectorCascade->detectionResult->numClusters;
    Rect *detectorBB = detectorCascade->detectionResult->detectorBB;
    Rect *kalmanBB = kalmanTracker->kalmanBB;
    
    if (prevBB && trackerBB && kalmanBB)
    {
        metric.processFrame(*prevBB, *trackerBB, *kalmanBB);
    }
    
    if(currBB)
    {
        delete currBB;
        currBB = NULL;
    }
    currConf = 0;
    valid = false;

    float confDetector = 0;

    if(numClusters == 1)
    {
        confDetector = nnClassifier->classifyBB(currImg, detectorBB);
    }


    if(trackerBB != NULL)
    {
        float confTracker = nnClassifier->classifyBB(currImg, trackerBB);
        float confKalman = nnClassifier->classifyBB(currImg,kalmanBB);
        std::cout<<"conf MDF tracker: "<<confTracker<<std::endl;
        std::cout<<"conf Kalman tracker: "<<confKalman<<std::endl;

        if(currBB)
        {
            delete currBB;
            currBB = NULL;
        }

        if(numClusters == 1 && confDetector > confTracker && tldOverlapRectRect(*trackerBB, *detectorBB) < 0.5)
        {

            currBB = tldCopyRect(detectorBB);
            currConf = confDetector;
            std::cout<<"scelto detector"<<endl;
            kalmanTracker->update(detectorBB);
        }
        else if (kalmanBB != NULL && confKalman>=0.85 && confKalman >=confTracker )
        {
            //float currentRatio = currBB->height/currBB->width;
            //float previousRatio = prevBB->height/prevBB->width;
            //if (currentRatio == previousRatio){
            currConf = confKalman;
            currBB = tldCopyRect(kalmanBB);
            //}
            kalmanTracker->update(kalmanBB);
            std::cout<<"kalman aggiornato con kalmanBB"<<endl;
        }
        else
        {
            currBB = tldCopyRect(trackerBB);
            currConf = confTracker;
            std::cout<<"scelto trackerBB"<<endl;
            if (confKalman<0.6)
            {
                kalmanTracker->update(trackerBB);
                std::cout<<"kalman aggiornato con trackerBB"<<endl;
            }
            else
            {
                kalmanTracker->update(kalmanBB);
                std::cout<<"kalman aggiornato con kalmanBB ma conf <0.85"<<endl;
            }

            if(confTracker > nnClassifier->thetaTP)
            {
                valid = true;
            }
            else if(wasValid && confTracker > nnClassifier->thetaFP)
            {
                valid = true;
            }
        }
    }
    else if(numClusters == 1)
    {
        if(currBB)
        {
            delete currBB;
            currBB = NULL;
        }
        currBB = tldCopyRect(detectorBB);
        currConf = confDetector;
        std::cout<<"scelto detectorBB"<<endl;
    }

    /*
    float var = CalculateVariance(patch.values, nn->patch_size*nn->patch_size);

    if(var < min_var) { //TODO: Think about incorporating this
        printf("%f, %f: Variance too low \n", var, classifier->min_var);
        valid = 0;
    }*/
}

void TLD::initialLearning()
{
    learning = true; //This is just for display purposes

    DetectionResult *detectionResult = detectorCascade->detectionResult;

    detectorCascade->detect(currImg);

    //This is the positive patch
    NormalizedPatch patch;
    tldExtractNormalizedPatchRect(currImg, currBB, patch.values);
    patch.positive = 1;

    float initVar = tldCalcVariance(patch.values, TLD_PATCH_SIZE * TLD_PATCH_SIZE);
    detectorCascade->varianceFilter->minVar = initVar / 2;


    float *overlap = new float[detectorCascade->numWindows];
    tldOverlapRect(detectorCascade->windows, detectorCascade->numWindows, currBB, overlap);

    //Add all bounding boxes with high overlap

    vector< pair<int, float> > positiveIndices;
    vector<int> negativeIndices;

    //First: Find overlapping positive and negative patches

    for(int i = 0; i < detectorCascade->numWindows; i++)
    {

        if(overlap[i] > 0.6)
        {
            positiveIndices.push_back(pair<int, float>(i, overlap[i]));
        }

        if(overlap[i] < 0.2)
        {
            float variance = detectionResult->variances[i];

            if(!detectorCascade->varianceFilter->enabled || variance > detectorCascade->varianceFilter->minVar)   //TODO: This check is unnecessary if minVar would be set before calling detect.
            {
                negativeIndices.push_back(i);
            }
        }
    }

    sort(positiveIndices.begin(), positiveIndices.end(), tldSortByOverlapDesc);

    vector<NormalizedPatch> patches;

    patches.push_back(patch); //Add first patch to patch list

    int numIterations = std::min<size_t>(positiveIndices.size(), 10); //Take at most 10 bounding boxes (sorted by overlap)

    for(int i = 0; i < numIterations; i++)
    {
        int idx = positiveIndices.at(i).first;
        //Learn this bounding box
        //TODO: Somewhere here image warping might be possible
        detectorCascade->ensembleClassifier->learn(&detectorCascade->windows[TLD_WINDOW_SIZE * idx], true, &detectionResult->featureVectors[detectorCascade->numTrees * idx]);
    }

    srand(1); //TODO: This is not guaranteed to affect random_shuffle

    std::random_shuffle(negativeIndices.begin(), negativeIndices.end());

    //Choose 100 random patches for negative examples
    for(size_t i = 0; i < std::min<size_t>(100, negativeIndices.size()); i++)
    {
        int idx = negativeIndices.at(i);

        NormalizedPatch patch;
        tldExtractNormalizedPatchBB(currImg, &detectorCascade->windows[TLD_WINDOW_SIZE * idx], patch.values);
        patch.positive = 0;
        patches.push_back(patch);
    }
    //DEBUG
    //printf("ok initial learning\n");
    //FINE
    detectorCascade->nnClassifier->learn(patches);

    delete[] overlap;

}

//Do this when current trajectory is valid
void TLD::learn()
{
    if(!learningEnabled || !valid || !detectorEnabled)
    {
        learning = false;
        return;
    }

    learning = true;

    DetectionResult *detectionResult = detectorCascade->detectionResult;

    if(!detectionResult->containsValidData)
    {
        detectorCascade->detect(currImg);
    }

    //This is the positive patch
    NormalizedPatch patch;
    tldExtractNormalizedPatchRect(currImg, currBB, patch.values);

    float *overlap = new float[detectorCascade->numWindows];
    tldOverlapRect(detectorCascade->windows, detectorCascade->numWindows, currBB, overlap);

    //Add all bounding boxes with high overlap

    vector<pair<int, float> > positiveIndices;
    vector<int> negativeIndices;
    vector<int> negativeIndicesForNN;

    //First: Find overlapping positive and negative patches

    for(int i = 0; i < detectorCascade->numWindows; i++)
    {

        if(overlap[i] > 0.6)
        {
            positiveIndices.push_back(pair<int, float>(i, overlap[i]));
        }

        if(overlap[i] < 0.2)
        {
            if(!detectorCascade->ensembleClassifier->enabled || detectionResult->posteriors[i] > 0.5)   //Should be 0.5 according to the paper
            {
                negativeIndices.push_back(i);
            }

            if(!detectorCascade->ensembleClassifier->enabled || detectionResult->posteriors[i] > 0.5)
            {
                negativeIndicesForNN.push_back(i);
            }

        }
    }

    sort(positiveIndices.begin(), positiveIndices.end(), tldSortByOverlapDesc);

    vector<NormalizedPatch> patches;

    patch.positive = 1;
    patches.push_back(patch);
    //TODO: Flip


    int numIterations = std::min<size_t>(positiveIndices.size(), 10); //Take at most 10 bounding boxes (sorted by overlap)

    for(size_t i = 0; i < negativeIndices.size(); i++)
    {
        int idx = negativeIndices.at(i);
        //TODO: Somewhere here image warping might be possible
        detectorCascade->ensembleClassifier->learn(&detectorCascade->windows[TLD_WINDOW_SIZE * idx], false, &detectionResult->featureVectors[detectorCascade->numTrees * idx]);
    }

    //TODO: Randomization might be a good idea
    for(int i = 0; i < numIterations; i++)
    {
        int idx = positiveIndices.at(i).first;
        //TODO: Somewhere here image warping might be possible
        detectorCascade->ensembleClassifier->learn(&detectorCascade->windows[TLD_WINDOW_SIZE * idx], true, &detectionResult->featureVectors[detectorCascade->numTrees * idx]);
    }

    for(size_t i = 0; i < negativeIndicesForNN.size(); i++)
    {
        int idx = negativeIndicesForNN.at(i);

        NormalizedPatch patch;
        tldExtractNormalizedPatchBB(currImg, &detectorCascade->windows[TLD_WINDOW_SIZE * idx], patch.values);
        patch.positive = 0;
        patches.push_back(patch);
    }

    detectorCascade->nnClassifier->learn(patches);

    //cout << "NN has now " << detectorCascade->nnClassifier->truePositives->size() << " positives and " << detectorCascade->nnClassifier->falsePositives->size() << " negatives.\n";

    delete[] overlap;
}

typedef struct
{
    int index;
    int P;
    int N;
} TldExportEntry;

void TLD::writeToFile(const char *path)
{
    NNClassifier *nn = detectorCascade->nnClassifier;
    EnsembleClassifier *ec = detectorCascade->ensembleClassifier;

    FILE *file = fopen(path, "w");
    std::fprintf(file, "#Tld ModelExport\n");
    std::fprintf(file, "%d #width\n", detectorCascade->objWidth);
    std::fprintf(file, "%d #height\n", detectorCascade->objHeight);
    std::fprintf(file, "%f #min_var\n", detectorCascade->varianceFilter->minVar);
    std::fprintf(file, "%d #Positive Sample Size\n", nn->truePositives->size());



    for(size_t s = 0; s < nn->truePositives->size(); s++)
    {
        float *imageData = nn->truePositives->at(s).values;

        for(int i = 0; i < TLD_PATCH_SIZE; i++)
        {
            for(int j = 0; j < TLD_PATCH_SIZE; j++)
            {
                std::fprintf(file, "%f ", imageData[i * TLD_PATCH_SIZE + j]);
            }

            std::fprintf(file, "\n");
        }
    }

    std::fprintf(file, "%d #Negative Sample Size\n", nn->falsePositives->size());

    for(size_t s = 0; s < nn->falsePositives->size(); s++)
    {
        float *imageData = nn->falsePositives->at(s).values;

        for(int i = 0; i < TLD_PATCH_SIZE; i++)
        {
            for(int j = 0; j < TLD_PATCH_SIZE; j++)
            {
                std::fprintf(file, "%f ", imageData[i * TLD_PATCH_SIZE + j]);
            }

            std::fprintf(file, "\n");
        }
    }

    std::fprintf(file, "%d #numtrees\n", ec->numTrees);
    detectorCascade->numTrees = ec->numTrees;
    std::fprintf(file, "%d #numFeatures\n", ec->numFeatures);
    detectorCascade->numFeatures = ec->numFeatures;

    for(int i = 0; i < ec->numTrees; i++)
    {
        std::fprintf(file, "#Tree %d\n", i);

        for(int j = 0; j < ec->numFeatures; j++)
        {
            float *features = ec->features + 4 * ec->numFeatures * i + 4 * j;
            std::fprintf(file, "%f %f %f %f # Feature %d\n", features[0], features[1], features[2], features[3], j);
        }

        //Collect indices
        vector<TldExportEntry> list;

        for(int index = 0; index < pow(2.0f, ec->numFeatures); index++)
        {
            int p = ec->positives[i * ec->numIndices + index];

            if(p != 0)
            {
                TldExportEntry entry;
                entry.index = index;
                entry.P = p;
                entry.N = ec->negatives[i * ec->numIndices + index];
                list.push_back(entry);
            }
        }

        std::fprintf(file, "%d #numLeaves\n", list.size());

        for(size_t j = 0; j < list.size(); j++)
        {
            TldExportEntry entry = list.at(j);
            std::fprintf(file, "%d %d %d\n", entry.index, entry.P, entry.N);
        }
    }

    std::fclose(file);

}

void TLD::readFromFile(const char *path)
{
    release();

    NNClassifier *nn = detectorCascade->nnClassifier;
    EnsembleClassifier *ec = detectorCascade->ensembleClassifier;

    FILE *file = fopen(path, "r");

    if(file == NULL)
    {
        printf("Error: Model not found: %s\n", path);
        exit(1);
    }

    int MAX_LEN = 255;
    char str_buf[255];
    std::fgets(str_buf, MAX_LEN, file); /*Skip line*/

    std::fscanf(file, "%d \n", &detectorCascade->objWidth);
    std::fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/
    std::fscanf(file, "%d \n", &detectorCascade->objHeight);
    std::fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

    std::fscanf(file, "%f \n", &detectorCascade->varianceFilter->minVar);
    std::fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

    int numPositivePatches;
    std::fscanf(file, "%d \n", &numPositivePatches);
    std::fgets(str_buf, MAX_LEN, file); /*Skip line*/


    for(int s = 0; s < numPositivePatches; s++)
    {
        NormalizedPatch patch;

        for(int i = 0; i < 15; i++)   //Do 15 times
        {

            std::fgets(str_buf, MAX_LEN, file); /*Read sample*/

            char *pch;
            pch = strtok(str_buf, " \n");
            int j = 0;

            while(pch != NULL)
            {
                float val = atof(pch);
                patch.values[i * TLD_PATCH_SIZE + j] = val;

                pch = strtok(NULL, " \n");

                j++;
            }
        }

        nn->truePositives->push_back(patch);
    }

    int numNegativePatches;
    std::fscanf(file, "%d \n", &numNegativePatches);
    std::fgets(str_buf, MAX_LEN, file); /*Skip line*/


    for(int s = 0; s < numNegativePatches; s++)
    {
        NormalizedPatch patch;

        for(int i = 0; i < 15; i++)   //Do 15 times
        {

            std::fgets(str_buf, MAX_LEN, file); /*Read sample*/

            char *pch;
            pch = strtok(str_buf, " \n");
            int j = 0;

            while(pch != NULL)
            {
                float val = atof(pch);
                patch.values[i * TLD_PATCH_SIZE + j] = val;

                pch = strtok(NULL, " \n");

                j++;
            }
        }

        nn->falsePositives->push_back(patch);
    }

    std::fscanf(file, "%d \n", &ec->numTrees);
    detectorCascade->numTrees = ec->numTrees;
    std::fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

    std::fscanf(file, "%d \n", &ec->numFeatures);
    detectorCascade->numFeatures = ec->numFeatures;
    std::fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

    int size = 2 * 2 * ec->numFeatures * ec->numTrees;
    ec->features = new float[size];
    ec->numIndices = pow(2.0f, ec->numFeatures);
    ec->initPosteriors();

    for(int i = 0; i < ec->numTrees; i++)
    {
        std::fgets(str_buf, MAX_LEN, file); /*Skip line*/

        for(int j = 0; j < ec->numFeatures; j++)
        {
            float *features = ec->features + 4 * ec->numFeatures * i + 4 * j;
            std::fscanf(file, "%f %f %f %f", &features[0], &features[1], &features[2], &features[3]);
            std::fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/
        }

        /* read number of leaves*/
        int numLeaves;
        std::fscanf(file, "%d \n", &numLeaves);
        std::fgets(str_buf, MAX_LEN, file); /*Skip rest of line*/

        for(int j = 0; j < numLeaves; j++)
        {
            TldExportEntry entry;
            std::fscanf(file, "%d %d %d \n", &entry.index, &entry.P, &entry.N);
            ec->updatePosterior(i, entry.index, 1, entry.P);
            ec->updatePosterior(i, entry.index, 0, entry.N);
        }
    }

    detectorCascade->initWindowsAndScales();
    detectorCascade->initWindowOffsets();

    detectorCascade->propagateMembers();

    detectorCascade->initialised = true;

    ec->initFeatureOffsets();

    std::fclose(file);
}


} /* namespace tld */
