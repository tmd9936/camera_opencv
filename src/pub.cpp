#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sstream> // for converting the command line parameter to integer

#include <linefinder.h>
#include <IPM.h>

#define PI 3.1415926

using namespace std;
using namespace cv;

int width = 640;
int height = 480;

double IPM_BOTTOM_RIGHT = width+100;
double IPM_BOTTOM_LEFT = -100;
double IPM_RIGHT = width/2+380;
double IPM_LEFT = width/2-380;
int IPM_diff = 0;

vector<Point2f> origPoints;
vector<Point2f> dstPoints;

int main(int argc, char** argv)
{
  // Check if video source has been passed as a parameter, 파라미터가 없으면아예 실행이 안됨..
  if(argv[1] == NULL) return 1;
	
  ros::init(argc, argv, "image_publisher");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub = it.advertise("camera/image", 1);

  // Convert the passed as command line parameter index for the video device to an integer
  std::istringstream video_sourceCmd(argv[1]);
  int video_source;
  // Check if it is indeed a number
  if(!(video_sourceCmd >> video_source)) return 1;

  cv::VideoCapture cap(video_source);
  // cap setting
  cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
  cap.set(CV_CAP_PROP_FRAME_WIDTH, height);
  cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('D', 'I', 'V', 'X'));
  
  // Check if video device can be opened with the given index
  if(!cap.isOpened()) return 1;
  cv::Mat frame, outputFrame;
  cv::UMat gray, blur, sobel;
  cv::Mat contours;
  IPM ipm;
  LineFinder ld;

  sensor_msgs::ImagePtr msg;

  origPoints.push_back(Point2f(IPM_BOTTOM_LEFT, (height-120)));
  origPoints.push_back(Point2f(IPM_BOTTOM_RIGHT, height-120));
  origPoints.push_back(Point2f(IPM_RIGHT, height/2+100));
  origPoints.push_back(Point2f(IPM_LEFT, height/2+100));	

  // The 4-points correspondences in the destination image
    
  dstPoints.push_back(Point2f(0, height));
  dstPoints.push_back(Point2f(width, height));
  dstPoints.push_back(Point2f(width, 0));
  dstPoints.push_back(Point2f(0, 0));

  // IPM object
  ipm.setIPM(Size(width, height), Size(width, height), origPoints, dstPoints);

  // Process
  //clock_t begin = clock();

  //ros::Rate loop_rate(5);
  while (nh.ok()) {
    cap >> frame;

    ipm.applyHomography(frame, outputFrame);

    // ---------  선분 검출 -------------

    cv::resize(outputFrame, outputFrame, cv::Size(320, 240));
    outputFrame = outputFrame(Range::all(), Range(40,280));
    cv::imshow("outputFrame", outputFrame);

    cv::cvtColor(outputFrame, gray, COLOR_RGB2GRAY);
    cv::blur(gray, blur, cv::Size(5,5));
    cv::Sobel(blur, sobel, blur.depth(), 1, 0, 3, 0.5, 127);
    cv::threshold(sobel, contours, 145, 255, CV_THRESH_BINARY);

    ld.setLineLengthAndGap(20, 120);
    ld.setMinVote(55);
    std::vector<cv::Vec4i> li = ld.findLines(contours);
    ld.drawDetectedLines(contours);
    
    cv::imshow("contours", contours);

    // --------- 자율주행  -------------

    if(!frame.empty()) {
      msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
      pub.publish(msg);
      cv::waitKey(1);
    }

    ros::spinOnce();
    //loop_rate.sleep();
  }
}
