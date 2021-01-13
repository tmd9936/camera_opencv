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
#include <std_msgs/Int16.h>

#include <camera_opencv/TrafficState.h>


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

int bottom_center = 160;
int sum_centerline = 0;
int count_centerline = 0;
int first_centerline = 0;
int last_centerline = 0;
double avr_center_to_left = 0;
double avr_center_to_right = 0;

double center_to_right = -1;
double center_to_left = -1; 

int centerline = 0;

int diff = 0;

int degree = 0;

int counter = 0;
int move_mouse_pixel = 0;

int main(int argc, char** argv)
{
  // Check if video source has been passed as a parameter, 파라미터가 없으면 아예 실행이 안됨..
  if(argv[1] == NULL) return 1;
	
  ros::init(argc, argv, "camera_opencv_node");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  // 이미지 보여주기용 퍼블리셔
  image_transport::Publisher image_pub = it.advertise("camera/image", 1);

  // 데이터 전송용 퍼블리셔
  ros::Publisher traffic_pub = nh.advertise<camera_opencv::TrafficState>("traffic_state",1);

  camera_opencv::TrafficState traffic_state_msg;
  traffic_state_msg.line_state = 0;

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
  // 선분검출용 mat 데이터
  cv::Mat frame, outputFrame;
  cv::UMat gray, blur, sobel;
  cv::Mat contours;

  // 주차구역 검출용 mat 데이터
  cv::Mat frame_hsv;
  cv::Mat red_mask, red_frame;
  cv::Mat red_image;

  vector<vector<Point>> rect_cont;
  vector<Point2f> approx;
  int approx_size = 0;

  Scalar lower_red = Scalar(160, 20, 100);
  Scalar upper_red = Scalar(179, 255, 255);

  ///////

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
  while (nh.ok()) 
  {
    cap >> frame;

    ipm.applyHomography(frame, outputFrame);

    // ---------  선분 검출 -------------

    cv::resize(outputFrame, outputFrame, cv::Size(320, 240));
    outputFrame = outputFrame(Range::all(), Range(40,280));
    cv::imshow("outputFrame", outputFrame);

    cv::cvtColor(outputFrame, gray, COLOR_RGB2GRAY);
    cv::blur(gray, blur, cv::Size(15,15));
    cv::Sobel(blur, sobel, blur.depth(), 1, 0, 3, 0.5, 127);
    cv::threshold(sobel, contours, 145, 255, CV_THRESH_BINARY);

    ld.setLineLengthAndGap(20, 120);
    ld.setMinVote(55);
    std::vector<cv::Vec4i> li = ld.findLines(contours);
    ld.drawDetectedLines(contours);
    
    //cv::imshow("contours", contours);

    // ---------  자율주행용 차선의 라디안 추출  -------------

		bottom_center = 160;
		sum_centerline = 0;
		count_centerline = 0;
		first_centerline = 0;
		last_centerline = 0;
		avr_center_to_left = 0;
		avr_center_to_right = 0;
    degree = 0;

		//#pragma omp parallel for
		for(int i=240; i>30; i--)
    {
			center_to_right = -1;
			center_to_left = -1;

			for (int j=0;j<150;j++) 
      {
				if (contours.at<uchar>(i, bottom_center+j) == 112 && center_to_right == -1) 
        {
					center_to_right = j;
				}
				if (contours.at<uchar>(i, bottom_center-j) == 112 && center_to_left == -1) 
        {
					center_to_left = j;
				}
			}
			if(center_to_left!=-1 && center_to_right!=-1)
      {
				centerline = (center_to_right - center_to_left +2*bottom_center)/2;
				if (first_centerline == 0 ) 
        {
					first_centerline = centerline;
				}
				sum_centerline += centerline;
				avr_center_to_left = (avr_center_to_left * count_centerline + center_to_left)/count_centerline+1;
				avr_center_to_right = (avr_center_to_right * count_centerline + center_to_right)/count_centerline+1;
				last_centerline = centerline;
				count_centerline++;
			} else {}
		}
    
    diff = 0;

    if (count_centerline!=0)
    {
      diff = sum_centerline/count_centerline - bottom_center;
      degree = atan2 (last_centerline - first_centerline, count_centerline) * 180 / PI;
      move_mouse_pixel = 0 - counter + diff;

      //ROS_INFO("move_mouse_pixel = %d", move_mouse_pixel);
      //ROS_INFO("degree msg = %d", degree);

      counter = diff;

      
    }
    traffic_state_msg.line_state = degree;

    // ---------  주차 구역 검출  ------------- 
    cv::cvtColor(outputFrame, frame_hsv, COLOR_BGR2HSV);

    cv::inRange(frame_hsv, lower_red, upper_red, red_mask);


    	//morphological opening 작은 점들을 제거 
		erode(red_mask, red_mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(red_mask, red_mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));


		//morphological closing 영역의 구멍 메우기 
		dilate(red_mask, red_mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(red_mask, red_mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    //cv::bitwise_and(outputFrame, outputFrame, red_image, red_mask);
    
    //cv::imshow("red_image", red_image);

    // 사각형 검출
    cv::findContours(red_mask, rect_cont, RETR_LIST, CHAIN_APPROX_SIMPLE);
    approx_size = 0;

    for(size_t i=0; i<rect_cont.size(); i++)
    {
      approxPolyDP(Mat(rect_cont[i]), approx, arcLength(Mat(rect_cont[i]), true)*0.02, true);

      if(fabs(contourArea(Mat(approx))) > 100)
      {
        approx_size = approx.size();

        if (approx_size % 2 == 0) {
				line(red_mask, approx[0], approx[approx.size() - 1], Scalar(0, 255, 0), 3);

          for (int k = 0; k < approx_size - 1; k++)
            line(red_mask, approx[k], approx[k + 1], Scalar(0, 255, 0), 3);

          for (int k = 0; k < approx_size; k++)
            circle(red_mask, approx[k], 3, Scalar(0, 0, 255));
			  }
        else {
          line(red_mask, approx[0], approx[approx.size() - 1], Scalar(0, 255, 0), 3);

          for (int k = 0; k < approx_size - 1; k++)
            line(red_mask, approx[k], approx[k + 1], Scalar(0, 255, 0), 3);

          for (int k = 0; k < approx_size; k++)
            circle(red_mask, approx[k], 3, Scalar(0, 0, 255));
        }

        // 꼭지점의 개수가 4개일때 출력
        //if(size == 4 && isContourConvex(Mat(approx)))

      }
      
    }

    // ---------  신호등 색 검출  -------------
    Mat frame_light;
    ipm.applyHomography(frame, frame_light);

    // Range 클래스는 Mat 클래스에서 행 또는 열 범위를 지정하는 템플릿 클래스
    // Range(start, end) -> start포함, end 포함x
    frame_light = frame_light(Range(0, frame_light.size().height * 2/3), Range::all());

    Mat gray_light;
    cvtColor(frame_light, gray_light, CV_BGR2GRAY);
    Mat blur_light;
    GaussianBlur(gray_light, blur_light, Size(0,0), 1.0);

    // 원 검출
    vector<Vec3f> circles;
    // 수정 필요한 곳
    HoughCircles(blur_light, circles, CV_HOUGH_GRADIENT, 1, 50, 120, 50, 50, 90);

    for(size_t i = 0; i < circles.size(); i++){
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1])));
      int radius = cvRound(circles[i][2]);
      circle(frame_light, center, radius, Scalar(0,0,255), 3, 8, 0);

      int x1 = int(cvRound(circles[i][0] - radius));
      int y1 = int(cvRound(circles[i][1] - radius));
      Rect rect(x1, y1, 2 * radius, 2 * radius);
      Mat crop_light = frame(rect);
      //imshow("crop", crop);

      int cw = rect.size().width;
      int ch = rect.size().height;

      // 신호등 영역 ROI mask 영상
      Mat mask_light(cw, ch, CV_8UC1, Scalar::all(0));
      Point crop_center(int(cw / 2), int(ch / 2));
      circle(mask_light, crop_center, radius, Scalar::all(255) , -1, 8, 0);
      //imshow("mask", mask);

      // 색 인식
      Mat hsv_light;
      cvtColor(crop_light, hsv_light, CV_BGR2HSV);
      vector<Mat> channels;
      split(hsv_light, channels);
      channels[0] += 30;
      merge(channels, hsv_light);

      float mean_hue_light = mean(hsv_light, mask_light)[0];
      // printf("%f \n", mean_hue_light);

      string color = "none";
      // 수정 필요한 곳
      if(mean_hue_light > 0 && mean_hue_light < 50) {
        color = "red"; // 숫자 뭘로 할지
      } else if(mean_hue_light > 90 && mean_hue_light < 150) {
        color = "green"; // 숫자 정하기
      }
      putText(frame_light, color, center, CV_FONT_HERSHEY_SIMPLEX, 0.75, Scalar::all(255));

      imshow("traffic_light", frame_light);
    }
    // -------------------------------------
      
    traffic_state_msg.station_area = approx_size;
    
    traffic_state_msg.traffic_color = 1;
    
    //ROS_INFO("line_state = %d", traffic_state_msg.line_state);
    //ROS_INFO("station_area = %d", traffic_state_msg.station_area);
    // 라인 degree 퍼블리싱
    traffic_pub.publish(traffic_state_msg);

    cv::imshow("red_mask", red_mask);

    if(!frame.empty()) 
    {
      msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
      image_pub.publish(msg);
      cv::waitKey(1);
    }

    ros::spinOnce();
    //loop_rate.sleep();
  }
}
