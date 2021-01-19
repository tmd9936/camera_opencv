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

#define RED 0
#define GREEN 1

#define CENTER_VALUE 479

using namespace std;
using namespace cv;

int width = 640;
int height = 480;

double IPM_BOTTOM_RIGHT = width + 100;
double IPM_BOTTOM_LEFT = -100;
double IPM_RIGHT = width / 2 + 380;
double IPM_LEFT = width / 2 - 380;
int IPM_diff = 0;

vector<Point2f> origPoints;
vector<Point2f> dstPoints;

int bottom_center = 320;
int sum_centerline = 0;
int count_centerline = 0;
int first_centerline = 0;
int last_centerline = 0;
double avr_center_to_left = 0;
double avr_center_to_right = 0;

double center_to_right = -1;
double center_to_left = -1;

// 선이 1개 일 때의 변수
int top_x = 0;
int top_y = 0;
int bottom_x = 0;
int bottom_y = 0;
int center_x = 0;
int center_y = 0;
double line_degree = 0.0;
double y_intercept = 0.0;
Point2i pre_center_point;
int chg_amount = 0;

int pass_chg_amount_value = 120;

//
int centerline = 0;

int diff = 0;

int degree = 0;

int counter = 0;
int move_mouse_pixel = 0;

int line_count = 0;
int pre_line_count = 0;

void initParams(ros::NodeHandle *nh_priv)
{
	nh_priv->param("pass_chg_amount_value", pass_chg_amount_value, pass_chg_amount_value);
}

int main(int argc, char **argv)
{
	// Check if video source has been passed as a parameter, 파라미터가 없으면 아예 실행이 안됨..
	if (argv[1] == NULL)
		return 1;

	ros::init(argc, argv, "camera_opencv_node");
	ros::NodeHandle nh;
	image_transport::ImageTransport it(nh);
	// 이미지 보여주기용 퍼블리셔
	image_transport::Publisher image_pub = it.advertise("camera/image", 1);

	// 데이터 전송용 퍼블리셔
	ros::Publisher traffic_pub = nh.advertise<camera_opencv::TrafficState>("traffic_state", 1);
	ros::NodeHandle nh_priv{"~"};
	initParams(&nh_priv);

	camera_opencv::TrafficState traffic_state_msg;
	traffic_state_msg.line_state = 0;

	// Convert the passed as command line parameter index for the video device to an integer
	std::istringstream video_sourceCmd(argv[1]);
	int video_source;
	// Check if it is indeed a number
	if (!(video_sourceCmd >> video_source))
		return 1;

	cv::VideoCapture cap(video_source);
	// cap setting
	cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, height);
	cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('D', 'I', 'V', 'X'));

	// Check if video device can be opened with the given index
	if (!cap.isOpened())
		return 1;
	// 선분검출용 mat 데이터
	cv::Mat frame, outputFrame;
	cv::UMat gray, blur, sobel;
	cv::Mat contours;

	// 왼쪽 오른쪽 구역 직선의 방성식을 얻기위한 mat 데이터
	cv::Mat slice_line_mat = cv::Mat::zeros(height, width, CV_8SC1);

	// 주차구역 검출용 mat 데이터
	cv::Mat frame_hsv;
	cv::Mat red_mask, red_frame;
	cv::Mat red_image;
	pre_center_point = Point2i(0, 0);

	vector<vector<Point>> rect_cont;
	vector<Point2f> approx;
	int approx_size = 0;

	Scalar lower_red = Scalar(160, 20, 100);
	Scalar upper_red = Scalar(179, 255, 255);

	///////
	IPM ipm;
	LineFinder ld;

	sensor_msgs::ImagePtr msg;

	origPoints.push_back(Point2f(IPM_BOTTOM_LEFT, (height - 120)));
	origPoints.push_back(Point2f(IPM_BOTTOM_RIGHT, height - 120));
	origPoints.push_back(Point2f(IPM_RIGHT, height / 2 + 100));
	origPoints.push_back(Point2f(IPM_LEFT, height / 2 + 100));

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
		cv::imshow("frame", frame);
		ipm.applyHomography(frame, outputFrame);

		ROS_INFO("pass_chg_amount_value = %d", pass_chg_amount_value);

		// ---------  선분 검출 -------------

		//cv::resize(outputFrame, outputFrame, cv::Size(640, 480));
		outputFrame = outputFrame(Range::all(), Range(80, width));

		// 560, 480
		cv::imshow("outputFrame", outputFrame);

		cv::cvtColor(outputFrame, gray, COLOR_RGB2GRAY);
		cv::blur(gray, blur, cv::Size(15, 15));
		cv::Sobel(blur, sobel, blur.depth(), 1, 0, 3, 0.5, 127);
		cv::threshold(sobel, contours, 145, 255, CV_THRESH_BINARY);

		ld.setLineLengthAndGap(20, 120);
		ld.setMinVote(55);
		std::vector<cv::Vec4i> lines = ld.findLines(contours);
		ld.drawDetectedLines(contours);

		cv::imshow("contours", contours);

		//morphological opening 작은 점들을 제거
		erode(contours, contours, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(contours, contours, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(contours, contours, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		//morphological closing 영역의 구멍 메우기
		dilate(contours, contours, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(contours, contours, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		
		// ---------  자율주행용 차선의 라디안 추출  -------------
		bottom_center = width/2 - 40;
		sum_centerline = 0;
		count_centerline = 0;
		first_centerline = 0;
		last_centerline = 0;
		avr_center_to_left = 0;
		avr_center_to_right = 0;
		degree = 0;
		diff = 0;
		
		// 선이 1개 일 때의 변수
		top_x = -1;
		top_y = 0;
		bottom_x = -1;
		bottom_y = height;
		center_x = 0;
		center_y = height/2;
		line_degree = 0.0;
		y_intercept = 0.0;

		//ROS_INFO("lines.size() %d", lines.size());
		// line_size()가 5개 이하면 라인이 없다고 판단
		if (lines.size() < 6)
		{
			line_count = 0;
		}
		else
		{
			// 중앙 선분 체크
			for (int i = height; i > 0; i--)
			{
				center_to_right = -1;
				center_to_left = -1;

				for (int j = 0; j < width/2; j++)
				{
					if (contours.at<uchar>(i, bottom_center + j) == 112 && center_to_right == -1)
					{
						center_to_right = j;
						//ROS_INFO("center_to_right %d", j);
					}
					if (contours.at<uchar>(i, bottom_center - j) == 112 && center_to_left == -1)
					{
						center_to_left = j;
						//ROS_INFO("center_to_left %d", j);
					}
				}
				if (center_to_left != -1 && center_to_right != -1)
				{
					centerline = (center_to_right - center_to_left + 2 * bottom_center) / 2;
					if (first_centerline == 0)
					{
						first_centerline = centerline;
					}
					sum_centerline += centerline;
					avr_center_to_left = (avr_center_to_left * count_centerline + center_to_left) / count_centerline + 1;
					avr_center_to_right = (avr_center_to_right * count_centerline + center_to_right) / count_centerline + 1;
					last_centerline = centerline;
					count_centerline++;

				}
				else
				{
				}
			}
			
			// 선분이 2개 일 때
			//if (count_centerline > 400 && count_centerline < 500)
			if (count_centerline == CENTER_VALUE && lines.size() > 23)
			{
				diff = sum_centerline / count_centerline - bottom_center;
				//degree = atan2(last_centerline - first_centerline, count_centerline) * 180 / PI;
				move_mouse_pixel = 0 - counter + diff;

				counter = diff;
				line_count = 2;
				degree = 0;
			}
			// 선분이 1개 일 때
			else if(count_centerline == 0 || centerline == 280)
			{
				// 위에 좌표, 아래좌표 얻기
				for (int i = 0; i < width; i++)
				{
					// 위에 좌표
					if (contours.at<uchar>(0, i) == 112)
					{
						top_x = i;
						//ROS_INFO("top_x %d", i);
					}
					// 아래 좌표
					if (contours.at<uchar>(height-5, i) == 112)
					{
						bottom_x = i;
						//ROS_INFO("bottom_x %d", i);
					}
				}
				// 기울기 구하기
				//line_degree = (height-5) / ((double)bottom_x - (double)top_x);

				// y절편 구하기
				//y_intercept = -(line_degree * top_x);
				
				// 중앙점 좌표구하기
				//one_line_center.x = (line_degree*one_line_center.y) + y_intercept;
				if (bottom_x != -1 && top_x != -1)
				{
					// 중앙점의 x좌표 구하기
					center_x = (bottom_x + top_x) / 2;

					//ROS_INFO("center_x = %d", center_x);

					if (pre_line_count == 1 || pre_line_count == 2)
					{
						pre_center_point.y = height/2;
						pre_center_point.x = center_x;
					}

					chg_amount = pre_center_point.x - center_x;

					if(abs(chg_amount) < pass_chg_amount_value)
					{
						degree = 0;
					}
					else
					{
						if(chg_amount > 0)
						{
							degree = 1;
						}
						else
						{
							degree = 0;
						}
						
					}

					line_count = 1;
				}
				//ROS_INFO("y_intercept = %f", y_intercept);
				//ROS_INFO("one_line_center.x = %d", one_line_center.x);
			}
			else {}
			//ROS_INFO("diff %d", int(diff));
			//ROS_INFO("move_mouse_pixel %d", int(move_mouse_pixel));
			//ROS_INFO("degree %d", int(degree));
		}

		pre_line_count = line_count;

		//ROS_INFO("count_centerline %d", count_centerline);
		//ROS_INFO("centerline %d", centerline);

		// 라인 관련 메세지(라인 각도, 라인 수)
		//traffic_state_msg.line_state = degree;
		traffic_state_msg.line_state = degree;
		traffic_state_msg.line_count = line_count;


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

		//cv::imshow("red_mask", red_mask);

		// 사각형 검출
		cv::findContours(red_mask, rect_cont, RETR_LIST, CHAIN_APPROX_SIMPLE);
		approx_size = 0;

		//ROS_INFO("rect_cont.size() %d", rect_cont.size());

		if (rect_cont.size() == 1)
		{
			for (size_t i = 0; i < rect_cont.size(); i++)
			{
				approxPolyDP(Mat(rect_cont[i]), approx, arcLength(Mat(rect_cont[i]), true) * 0.02, true);

				if (fabs(contourArea(Mat(approx))) > 100)
				{
					approx_size = approx.size();

					if (approx_size % 2 == 0)
					{
						line(red_mask, approx[0], approx[approx.size() - 1], Scalar(0, 255, 0), 3);

						for (int k = 0; k < approx_size - 1; k++)
							line(red_mask, approx[k], approx[k + 1], Scalar(0, 255, 0), 3);

						for (int k = 0; k < approx_size; k++)
							circle(red_mask, approx[k], 3, Scalar(0, 0, 255));
					}
					else
					{
						line(red_mask, approx[0], approx[approx.size() - 1], Scalar(0, 255, 0), 3);

						for (int k = 0; k < approx_size - 1; k++)
							line(red_mask, approx[k], approx[k + 1], Scalar(0, 255, 0), 3);

						for (int k = 0; k < approx_size; k++)
							circle(red_mask, approx[k], 3, Scalar(0, 0, 255));
					}

					// 꼭지점의 개수가 4개일때 출력
					if(approx_size == 4 && isContourConvex(Mat(approx)))
					{
						approx_size = 4;
						//ROS_INFO("rect_cont[%d][0].x %d", i, rect_cont[i][0].x);
						//ROS_INFO("rect_cont[%d][0].y %d", i, rect_cont[i][0].y);
					}
					else 
					{
						approx_size = 0;
					}
				} //(fabs(contourArea(Mat(approx))) > 100)
			} // for (size_t i = 0; i < rect_cont.size(); i++)
		}//(rect_cont.size() == 1)

		traffic_state_msg.station_area = approx_size;

		// ---------  신호등 색 검출  -------------
		Mat frame_light;

		// frame.copyTo(frame_light);
		frame_light = frame.clone();

		frame_light = frame_light(Range(0, frame_light.size().height * 2 / 3), Range::all());

		Mat gray_light;
		cvtColor(frame_light, gray_light, CV_BGR2GRAY);
		Mat blur_light;
		GaussianBlur(gray_light, blur_light, Size(0, 0), 1.0);

		// 원 검출
		vector<Vec3f> circles;
		// 수정 필요한 곳
		HoughCircles(blur_light, circles, CV_HOUGH_GRADIENT, 1, 100, 60, 80, 30, 90);

		// ROS_INFO("cnt = %d", circles.size());
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(circles[i][0], circles[i][1]);
			int radius = circles[i][2];
			circle(frame_light, center, radius, Scalar(0, 0, 255), 3, 8, 0);

			int x1 = int(circles[i][0] - radius);
			int y1 = int(circles[i][1] - radius);
			Rect rect(x1, y1, 2 * radius, 2 * radius);

			// ROS_INFO("rect.x = %d", rect.x);
			// ROS_INFO("rect.y = %d", rect.y);
			// ROS_INFO("rect.width = %d", rect.width);
			// ROS_INFO("rect.height = %d", rect.height);

			if (rect.x >= 0 && rect.y >= 0 && rect.width >= 0 && rect.height >= 0 && rect.width + rect.x < frame_light.cols && rect.height + rect.y < frame_light.rows)
			{

				Mat crop_light = frame_light(rect);
				//imshow("crop", crop_light);

				int cw = rect.width;
				int ch = rect.height;

				// 신호등 영역 ROI mask 영상
				Mat mask_light(cw, ch, CV_8UC1, Scalar::all(0));
				Point crop_center(int(cw / 2), int(ch / 2));
				circle(mask_light, crop_center, radius, Scalar::all(255), -1, 8, 0);
				//imshow("mask", mask_light);

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
				if (mean_hue_light > 30 && mean_hue_light < 60 || mean_hue_light > 170)
				{
					color = "red";
					traffic_state_msg.traffic_color = RED;
				}
				else if (mean_hue_light > 80 && mean_hue_light < 110)
				{
					color = "green";
					traffic_state_msg.traffic_color = GREEN;
				}
				putText(frame_light, color, center, CV_FONT_HERSHEY_SIMPLEX, 0.75, Scalar::all(255));
				Point center_plus_y(circles[i][0], circles[i][1] + 20);
				putText(frame_light, to_string(mean_hue_light), center_plus_y, CV_FONT_HERSHEY_SIMPLEX, 0.75, Scalar::all(255));

				//waitKey(1);
			}
		}

		// -------------------------------------

		//imshow("traffic_light", frame_light);

		//traffic_state_msg.traffic_color = 1;

		//ROS_INFO("line_state = %d", traffic_state_msg.line_state);
		//ROS_INFO("station_area = %d", traffic_state_msg.station_area);
		// 라인 degree 퍼블리싱
		traffic_pub.publish(traffic_state_msg);
		//cv::imshow("red_mask", red_mask);

		//imshow("frame", frame);

		if (!frame.empty())
		{
			// mono8 => grayscale
			msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", contours).toImageMsg();

			//msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
			image_pub.publish(msg);
			cv::waitKey(1);
		}

		ros::spinOnce();
		//loop_rate.sleep();
	}
}
