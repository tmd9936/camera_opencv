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

using namespace std;
using namespace cv;

int width = 320;
int height = 240;

double IPM_BOTTOM_RIGHT = width + 100;
double IPM_BOTTOM_LEFT = -100;
double IPM_RIGHT = width / 2 + 380;
double IPM_LEFT = width / 2 - 380;
int IPM_diff = 0;

vector<Point2f> origPoints;
vector<Point2f> dstPoints;

// 선검출
int deviation = 0;

double boundary = 0.15;

// 외쪽 오른쪽 선분 검출 결과
int is_left = 0;
int is_right = 0;

void initParams(ros::NodeHandle *nh_priv)
{
	nh_priv->param("boundary", boundary, boundary);
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
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
	//cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('D', 'I', 'V', 'X'));

	// Check if video device can be opened with the given index
	if (!cap.isOpened())
		return 1;
	// 선분검출용
	cv::Mat frame, outputFrame, view_frame, new_frame, canny;
	cv::UMat gray, blur, sobel;
	cv::Mat contours, region_mask, edges_mask;

	vector<Point> polygon;
	polygon.push_back(Point(0, height));
	polygon.push_back(Point(0, height / 2));
	polygon.push_back(Point(width, height / 2));
	polygon.push_back(Point(width, height));

	// 왼쪽 오른쪽 구역 직선의 방성식을 얻기위한 mat 데이터
	cv::Mat slice_line_mat = cv::Mat::zeros(height, width, CV_8SC1);

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

		frame.copyTo(view_frame);

		// ---------  선분 검출 -------------

		// detect_edges
		//cv::cvtColor(frame, gray, COLOR_BGR2GRAY);
		//cv::blur(gray, blur, cv::Size(15, 15));

		cv::cvtColor(frame, blur, COLOR_BGR2HSV);

		cv::blur(blur, blur, cv::Size(15, 15));
		
		cv::inRange(blur, Scalar(0,0,0,0), Scalar(180, 255, 80, 0), edges_mask);

		cv::Canny(edges_mask, contours, 50, 100);

		//cv::Sobel(blur, sobel, blur.depth(), 1, 0, 3, 0.5, 127);
		//cv::threshold(sobel, contours, 145, 255, CV_THRESH_BINARY);

		//morphological closing 영역의 구멍 메우기
		//dilate(contours, contours, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		//dilate(contours, contours, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		
		//erode(contours, contours, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		// ld.setLineLengthAndGap(20, 120);
		// ld.setMinVote(55);
		// vector<cv::Vec4i> lines = ld.findLines(contours);

		// cv::cvtColor(contours, new_frame, cv::COLOR_GRAY2BGR);
		//ld.drawDetectedLines(new_frame, Scalar(255, 0, 0));

		// cv::cvtColor(new_frame, new_frame, cv::COLOR_BGR2HSV);

		// // cv::inRange()
		// dilate(new_frame, new_frame, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		// //erode(new_frame, new_frame, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		// cv::inRange(new_frame, Scalar(90, 120, 0), Scalar(150, 255, 255), edges_mask);
		// cv::Canny(edges_mask, canny, 50, 100);

		//region_of_interest
		region_mask = Mat::zeros(height, width, 0);
		cv::fillConvexPoly(region_mask, polygon, 255);
		cv::bitwise_and(contours, region_mask, contours);

		// detect_line_segments
		vector<cv::Vec4i> line_segments;
		cv::HoughLinesP(contours, line_segments, 1, CV_PI / 180.0, 10, 5, 150);

		// average_slope_intercept
		vector<Vec4i> lane_lines;

		vector<Point2f> left_fit;
		vector<Point2f> right_fit;

		double left_region_boundary = width * (1.0 - boundary);
		double right_region_boundary = width * (boundary);

		is_left = 0;
		is_right = 0;

		for (auto &line_segment : line_segments)
		{

			int x1 = line_segment[0];
			int y1 = line_segment[1];
			int x2 = line_segment[2];
			int y2 = line_segment[3];

			if (x1 == x2)
				continue;

			double slope = (double(y2 - y1)) / (double(x2 - x1));
			double intercept = y1 - (slope * x1);

			if (slope < 0.)
			{
				if (x1 < left_region_boundary && x2 < left_region_boundary)
				{
					left_fit.push_back(Point2f(slope, intercept));
				}
			}
			else
			{
				if (x1 > right_region_boundary && x2 > right_region_boundary)
				{
					right_fit.push_back(Point2f(slope, intercept));
				}
			}
		}

		Point2f left_fit_average = Point2f(0., 0.);
		for (auto &fit : left_fit)
		{
			left_fit_average.x += fit.x;
			left_fit_average.y += fit.y;
		}
		left_fit_average.x /= float(left_fit.size());
		left_fit_average.y /= float(left_fit.size());

		if (left_fit.size() > 0)
		{
			// 왼쪽 길이 있으면 1을 대입
			is_left = 1;

			float slope = left_fit_average.x;
			float intercept = left_fit_average.y;

			int y1 = height;
			int y2 = int(y1 / 2.0);

			if (slope == 0)
				slope = 0.1;

			int x1 = int((y1 - intercept) / slope);
			int x2 = int((y2 - intercept) / slope);

			lane_lines.push_back(cv::Vec4i(x1, y1, x2, y2));
		}

		Point2f right_fit_average = Point2f(0., 0.);
		for (auto &fit : right_fit)
		{
			right_fit_average.x += fit.x;
			right_fit_average.y += fit.y;
		}
		right_fit_average.x /= right_fit.size();
		right_fit_average.y /= right_fit.size();

		if (right_fit.size() > 0)
		{
			// 오른쪽 길이 있으면 1을 대입
			is_right = 1;

			float slope = right_fit_average.x;
			float intercept = right_fit_average.y;

			int y1 = height;
			int y2 = int(y1 / 2);

			if (slope == 0)
				slope = 0.1;

			int x1 = int((y1 - intercept) / slope);
			int x2 = int((y2 - intercept) / slope);

			lane_lines.push_back(cv::Vec4i(x1, y1, x2, y2));
		}

		// disply lines
		Mat line_image = Mat::zeros(height, width, frame.type());

		if (lane_lines.size() != 0)
		{
			for (auto &line : lane_lines)
			{
				int x1 = line[0];
				int y1 = line[1];
				int x2 = line[2];
				int y2 = line[3];

				cv::line(line_image, Point(x1, y1), Point(x2, y2), Scalar(125, 125, 125), 6);
			}
		}

		cv::addWeighted(view_frame, 0.8, line_image, 1, 1, view_frame);

		// get_steering_angle
		double x_offset = 0.;
		double y_offset = 0.;

		if (lane_lines.size() == 2)
		{
			int left_x2 = lane_lines[0][2];
			int right_x2 = lane_lines[1][2];
			int mid = int(width / 2.0);
			x_offset = (left_x2 + right_x2) / 2.0 - mid;
			y_offset = int(height / 2.0);
		}
		else if (lane_lines.size() == 1)
		{
			int x1 = lane_lines[0][0];
			int x2 = lane_lines[0][2];
			x_offset = x2 - x1;
			y_offset = int(height / 2.0);
		}
		else if (lane_lines.size() == 0)
		{
			x_offset = 0;
			y_offset = int(height / 2.0);
		}
		else
		{
			x_offset = 0;
			y_offset = int(height / 2.0);
		}

		double angle_to_mid_radian = atan(x_offset / y_offset);
		double angle_to_mid_deg = angle_to_mid_radian * 180.0 / CV_PI;
		int steering_angle = int(angle_to_mid_deg) + 90;

		//display heading line
		Mat heading_image = Mat::zeros(height, width, frame.type());

		double steering_angle_radian = steering_angle / 180.0 * CV_PI;

		int steer_x1 = int(width / 2.0);
		int steer_y1 = height;
		int steer_x2 = (steer_x1 - height / 2 / tan(steering_angle_radian));
		int steer_y2 = int(height / 2);

		cv::line(heading_image, Point(steer_x1, steer_y1), Point(steer_x2, steer_y2), Scalar(0, 0, 255), 5);

		addWeighted(view_frame, 0.8, heading_image, 1, 1, view_frame);

		int deviation = steering_angle - 90;
		// ROS_INFO("deviation  %d", deviation);

		// 라인 관련 메세지(라인 각도, 라인 수)
		traffic_state_msg.line_state = deviation;
		traffic_state_msg.left_line_count = is_left;
		traffic_state_msg.right_line_count = is_right;

		// ---------  주차 구역 검출  -------------
		//ipm.applyHomography(frame, outputFrame);
		cv::cvtColor(frame, frame_hsv, COLOR_BGR2HSV);

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

		double min_x = 9999.0;
		double max_x = 0.;

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
						line(view_frame, approx[0], approx[approx.size() - 1], Scalar(0, 255, 0), 3);

						for (int k = 0; k < approx_size - 1; k++){
							line(view_frame, approx[k], approx[k + 1], Scalar(0, 255, 0), 3);
							//ROS_INFO("approx[%d].x %f", k, approx[k].x);
							//ROS_INFO("approx[%d].y %f", k, approx[k].y);
							//ROS_INFO("approx[%d].x %f", k+1, approx[k+1].x);
							//ROS_INFO("approx[%d].y %f", k+1, approx[k+1].y);
						}

						for (int k = 0; k < approx_size; k++)
							circle(view_frame, approx[k], 3, Scalar(0, 0, 255));
					}

					// 꼭지점의 개수가 4개일때 출력
					if (approx_size == 4 && isContourConvex(Mat(approx)))
					{
						for (int k = 0; k < approx_size; k++)
						{
							if (approx[k].x < min_x)
							{
								min_x = approx[k].x;
							}

							if (approx[k].x > max_x)
							{
								max_x = approx[k].x;
							}
						}
						//if ((max_x - min_x) > width / 2.0)
						if ((max_x - min_x) > 180.0)
							approx_size = 4;
						else
							approx_size = 0;
						// ROS_INFO("max - min = %f , approx_size = %d", max_x - min_x, approx_size);
						// ROS_INFO("approx[%d][0].x %f", i, approx[0].x);
						// ROS_INFO("approx[%d][0].y %f", i, approx[0].y);
						// ROS_INFO("approx[%d][1].x %f", i, approx[1].x);
						// ROS_INFO("approx[%d][1].y %f", i, approx[1].y);
					}
					else
					{
						approx_size = 0;
					}
				} //(fabs(contourArea(Mat(approx))) > 100)
			}	  // for (size_t i = 0; i < rect_cont.size(); i++)
		}		  //(rect_cont.size() == 1)

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
			circle(view_frame, center, radius, Scalar(0, 0, 255), 3, 8, 0);

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
				putText(view_frame, color, center, CV_FONT_HERSHEY_SIMPLEX, 0.75, Scalar::all(255));
				Point center_plus_y(circles[i][0], circles[i][1] + 20);
				putText(view_frame, to_string(mean_hue_light), center_plus_y, CV_FONT_HERSHEY_SIMPLEX, 0.75, Scalar::all(255));

				//waitKey(1);
			}
		}

		// -------------------------------------

		//imshow("traffic_light", frame_light);

		//ROS_INFO("line_state = %d", traffic_state_msg.line_state);
		//ROS_INFO("station_area = %d", traffic_state_msg.station_area);
		// 메세지 퍼블리싱
		traffic_pub.publish(traffic_state_msg);

		// cv::imshow("view_frame", view_frame);

		if (!frame.empty())
		{
			// mono8 => grayscale
			//msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", contours).toImageMsg();

			msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", view_frame).toImageMsg();
			image_pub.publish(msg);
			cv::waitKey(1);
		}

		ros::spinOnce();
		//loop_rate.sleep();
	}
}
