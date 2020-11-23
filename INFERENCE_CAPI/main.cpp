#include "opencv2/opencv.hpp"
#include "TrtNet.h"
#include "argsParser.h"
#include "configs.h"
#include <chrono>
#include "dataReader.h"
#include "eval.h"
#include <string>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <unistd.h>

//OpenCL
#include "opencv2/core/ocl.hpp"

#define LOAD_FROM_ENGINE

typedef struct
{
    int ImageCounter;
    bool SaveNextImage;
    bool busy;
   	cv::Mat frame; 
} CUSTOMDATA;

cv::Mat video_img;

cv::Mat video_img1;
cv::Mat video_img2;
cv::Mat video_img3;

using namespace std;
using namespace argsParser;
using namespace Tn;
using namespace cv;

vector<Bbox> cam1_bbox, cam2_bbox, cam3_bbox;

////////////////////////////////////////////////////////////////////
// Callback called for new images by the internal appsink
vector<float> prepareImage(cv::Mat& img, int nC, int nW, int nH) 
{
    cv::Mat img_resized;
    auto ReSize = cv::Size(nH,nW);
    cv::resize(img, img, ReSize);

    cv::Mat img_float;
    img.convertTo(img_float, CV_32FC3, 1/255.0);

    /*   
    cv::Mat img_float;
    img.convertTo(img_float, CV_32FC3);
    float r,g,b;
    float mean[3] = {103.53, 116.28, 123.675};
    float std[3] = {57.375, 57.12, 58.395};
    for(int n = 0; n < nW; n++){
        for(int m = 0; m < nH; m++){
            //BGR to RGB
            r = (img.at<cv::Vec3b>(n,m)[0] - mean[0]) / std[0];
            if(m==0 && n==0)
                cout << r;
            g = (img.at<cv::Vec3b>(n,m)[1] - mean[1]) / std[1];
            b = (img.at<cv::Vec3b>(n,m)[1] - mean[1]) / std[1];
            img_float.at<cv::Vec3b>(n,m)[0] = r;
            img_float.at<cv::Vec3b>(n,m)[1] = g;
            img_float.at<cv::Vec3b>(n,m)[2] = b;
        }
    }
    */
    cout << "Value 0,0:" <<img_float.at<cv::Vec3b>(0,0)[0] << img_float.at<cv::Vec3b>(0,0)[1] << img_float.at<cv::Vec3b>(0,0)[2] <<"\n";
    vector<Mat> input_channels(nC);
    cv::split(img_float, input_channels);

    vector<float> result(nH * nW * nC);
    auto data = result.data();
    int channelLength = nH * nW;
    for (int i = 0; i < nC; ++i) {
        //input_channels[i] = (input_channels[i] - mean[i]) / std[i];
        memcpy(data, input_channels[i].data, channelLength * sizeof(float));
        data += channelLength;
    }
    
    /*
    cv::Mat rgb;
    cv::cvtColor(img, rgb, CV_BGR2RGB); //BGR to RGB 컨버팅

    cv::Mat img_float;
    if (nC == 3)					// 컬러 이미지일 경우
        rgb.convertTo(img_float, CV_32FC3, 1/255.0);
    else						// 흑백 이미지일 경우
        rgb.convertTo(img_float, CV_32FC1 ,1/255.0);

    //딥러닝 입력을 위해 이미지의 HWC 순서를 CHW 순서로 변환
    vector<Mat> input_channels(nC);
    cv::split(img_float, input_channels);

    vector<float> result(nH * nW * nC);
    auto data = result.data();
    int channelLength = nH * nW;
    for (int i = 0; i < nC; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }
    */
    return result;
}

vector<string> split(const string& str, char delim)
{
    stringstream ss(str);
    string token;
    vector<string> container;
    while (getline(ss, token, delim)) {
        container.push_back(token);
    }
    return container;
}

void postProcessImg(cv::Mat& img,vector<float>& detections,int nClass, int nC, int nW, int nH, string filename, string show_result)
{
    using namespace cv;

    std::vector<std::vector<std::vector<float>>> outDataV(nClass, std::vector<std::vector<float>>(nW, std::vector<float>(nH, 0.0)));
    int classPredId[19] = {7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33};
    int classPredColors[19][3] = {  
        {128, 64, 128},
        {244, 35, 232},
        {70, 70, 70},
        {102, 102, 156},
        {190, 153, 153},
        {153, 153, 153},
        {250, 170, 30},
        {220, 220, 0},
        {107, 142, 35},
        {152, 251, 152},
        {0, 130, 180},
        {220, 20, 60},
        {255, 0, 0},
        {0, 0, 142},
        {0, 0, 70},
        {0, 60, 100},
        {0, 80, 100},
        {0, 0, 230},
        {119, 11, 32},
    };
    uint8_t PredSeg[nW][nH];
    uint8_t PredSegID[nW][nH];

    int nCount = 0;

    for(int i = 0; i < nClass; i++){
        for(int n = 0; n < nW; n++){
            for(int m = 0; m < nH; m++){
                outDataV[i][n][m] = detections[nCount];
                nCount +=1;        
            }   
        }   
    }
    
    int n, m, i;
    for(n = 0; n < nW; n++){
        for(m = 0; m < nH; m++){
            int max_indx = 0;
            float max = -999.0;
            for(i = 0; i < nClass; i++){
                float a = outDataV[i][n][m];
                if(max < a){ 
                    max_indx = i;
                    max = a;
                }
            }   
            PredSeg[n][m] = max_indx;
            PredSegID[n][m] = classPredId[max_indx];
        }   
    }

    if(show_result == "on"){
        Mat segMapImg = cv::Mat(nW, nH, CV_8UC1, &PredSegID);
        string Dir = "../pred/";
        string fName = filename.erase(filename.length()-4) + "_segID.jpg";
        string segMapImgName = "./pred/" + fName.substr(fName.find_last_of("\\/")+1);

        cv::imwrite(segMapImgName, segMapImg);

        Mat segResultColor = cv::Mat(nW, nH, CV_8UC3);
        for(int n = 0; n < nW; n++){
            for(int m = 0; m < nH; m++){
               //BGR to RGB
               segResultColor.at<cv::Vec3b>(n,m)[0] = (0.2 * img.at<cv::Vec3b>(n,m)[2] + 0.8 * classPredColors[PredSeg[n][m]][2]);
               segResultColor.at<cv::Vec3b>(n,m)[1] = (0.2 * img.at<cv::Vec3b>(n,m)[1] + 0.8 * classPredColors[PredSeg[n][m]][1]);
               segResultColor.at<cv::Vec3b>(n,m)[2] = (0.2 * img.at<cv::Vec3b>(n,m)[0] + 0.8 * classPredColors[PredSeg[n][m]][0]);
            }
        }

        string fName1 = filename.erase(filename.length()-4) + "_predSeg.jpg";
        string segOverlapImgName = "./pred/" + fName1.substr(fName1.find_last_of("\\/")+1);
        cv::imwrite(segOverlapImgName, segResultColor);

    }
}

int main( int argc, char* argv[] ){
	int i = 0;
    parser::ADD_ARG_INT("C",Desc("channel"),DefaultValue(to_string(INPUT_CHANNEL)));
    parser::ADD_ARG_INT("H",Desc("height"),DefaultValue(to_string(INPUT_HEIGHT)));
    parser::ADD_ARG_INT("W",Desc("width"),DefaultValue(to_string(INPUT_WIDTH)));
    parser::ADD_ARG_STRING("mode",Desc("runtime mode"),DefaultValue(MODE), ValueDesc("fp32/fp16/int8"));
    parser::ADD_ARG_INT("class",Desc("num of classes"),DefaultValue(to_string(DETECT_CLASSES)));

    // input
    parser::ADD_ARG_STRING("input",Desc("input image file"),DefaultValue(INPUT_IMAGE),ValueDesc("file"));
    parser::ADD_ARG_STRING("cam",Desc("camera on"),DefaultValue("off"),ValueDesc("on, off"));
    parser::ADD_ARG_STRING("evallist",Desc("eval gt list"),DefaultValue(EVAL_LIST),ValueDesc("file"));
    parser::ADD_ARG_STRING("video",Desc("video on"),DefaultValue("off"),ValueDesc("on, off"));
    parser::ADD_ARG_STRING("img",Desc("img on"),DefaultValue("off"),ValueDesc("on, off"));
    parser::ADD_ARG_STRING("engine",Desc("input engine"),DefaultValue("./FCHardNet_static608_fp16.engine"),ValueDesc("engine file name"));

    // result show
    parser::ADD_ARG_STRING("show_result",Desc("Segmented result show"),DefaultValue("on"),ValueDesc("on, off"));

    CUSTOMDATA CustomData1;
    CUSTOMDATA CustomData2;
    CUSTOMDATA CustomData3;

    CustomData1.ImageCounter = 0;
    CustomData1.SaveNextImage = false;
    std::cout <<"camera setup..."<< endl;
    
    if(argc < 2){
        parser::printDesc();
        exit(-1);
    }

    parser::parseArgs(argc,argv);
    
    string cam_input = parser::getStringValue("cam");
    string img_input = parser::getStringValue("img");
	string video_input = parser::getStringValue("video");
    string show_result = parser::getStringValue("show_result");
	
    vector<vector<float>> calibData;
    string calibFileList = parser::getStringValue("calib");
    string mode = parser::getStringValue("mode");

    RUN_MODE run_mode = RUN_MODE::FLOAT16;
    
    // mode에 맞는 engine 파일 불러오기
    //string engineName = "FCHardNet_static608_" + mode + ".engine";
    string engineName = parser::getStringValue("engine");
    std::cout <<"Load model = " << engineName << endl;

#ifdef LOAD_FROM_ENGINE    
    trtNet net(engineName);
#endif
    int outputCount = net.getOutputSize()/sizeof(float);
    
    std::cout <<" OutputCount size from NET:" << outputCount << "\n";

    unique_ptr<float[]> outputData(new float[outputCount]);
    int classNum = parser::getIntValue("class");

    string listFile = parser::getStringValue("evallist");
    string inputFileName = parser::getStringValue("input");

    list<string> fileNames;
    list<vector<Bbox>> groundTruth;

    if(listFile.length() > 0){
        std::cout << "loading from eval list " << listFile << std::endl; 
        tie(fileNames, groundTruth) = readObjectLabelFileList(listFile);
    }
    else if (cam_input == "on"){
    	std::cout << "camera start" << endl;
    }
    else{
    	std::cout << "fileNames call" << endl;
        fileNames.push_back(inputFileName);
    }

    if(img_input == "on")
    {
    	cout << "img input call \n";

    	for (const auto& filename :fileNames) {	    
    	    cout << "process: " << filename << endl;

            int c = parser::getIntValue("C");   //net C
            int h = parser::getIntValue("H");   //net h
            int w = parser::getIntValue("W");   //net w

            //Prepare-Process
			auto t_start = std::chrono::high_resolution_clock::now(); 
    	    cv::Mat img = cv::imread(filename); 

    	    vector<float> inputData = prepareImage(img, c, w, h);	
    	    auto t_end = std::chrono::high_resolution_clock::now();
    	    float total = std::chrono::duration<float>(t_end - t_start).count();

    	    cout << "Preprocess time is: " << total <<" s " << endl;
    	    if (!inputData.data())
    	        continue;

            //Inference-Process
            t_start = std::chrono::high_resolution_clock::now(); 

    	    net.doInference(inputData.data(), outputData.get());

            t_end = std::chrono::high_resolution_clock::now();
            total = std::chrono::duration<float>(t_end - t_start).count();
            cout << "Net processing time is: " << total <<"s" << "(" << 1/total << ")fps" << endl;

            //Post-Process
            t_start = std::chrono::high_resolution_clock::now(); 
      	    //Get Output
    	    auto output = outputData.get();

            //later detect result
    	    vector<float> result;
    	    result.resize(1 * classNum * w * h);

            memcpy(result.data(), &output[0], 1 * classNum * w * h * sizeof(float));

            postProcessImg(img, result, classNum, c, w, h, filename, show_result);

            t_end = std::chrono::high_resolution_clock::now();
            total = std::chrono::duration<float>(t_end - t_start).count();
            cout << "Postprocess time is: " << total <<" s " << endl;
			
			//cv::imshow("img", img);
			//cv::waitKey(0);
    	}
    	return 0;
    }
}