/*
说明书：首先输入图片路径（请用“\\”代替“\”否则路径错误）
然后输入你想使用的pooling掩膜大小（掩膜边长）
若不输入整数，输入“y”则使用默认掩膜大小3
*/
#include "opencv2/opencv.hpp"
#include "pch.h"
#include "opencv2/opencv.hpp"
#include "pch.h"
#include <string>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <list>
typedef cv::Mat Mat;
using namespace std;
using namespace cv;

Mat padding(Mat& src, int kernal_size, int row, int col) {//for pad aroundly, this function needs to initialize four paraments!

	Mat src_pad;
	int pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
	if ((row % kernal_size) && (pad_bottom = (kernal_size - (row % kernal_size))))
	{
		if (!(pad_bottom & -2))
		{
			pad_top = pad_bottom / 2;
			pad_bottom = pad_top + 1;
		}
		else
		{
			pad_top = pad_bottom = pad_bottom / 2;
		}
	}
	if ((col % kernal_size) && (pad_right = (kernal_size - (col % kernal_size))))
	{
		if (!(pad_right & -2))
		{
			pad_left = pad_right / 2;
			pad_right = pad_left + 1;
		}
		else
		{
			pad_left = pad_right = pad_right / 2;
		}
	}
	copyMakeBorder(src, src_pad, pad_top, pad_bottom, pad_left, pad_right, BORDER_REFLECT_101);
	return src_pad;
}

template<typename T>
bool SortPredicate_Descending(const T& p1, const T& p2) {

	return (p1 > p2);
}

template<typename T>
T pixel_max(list<T> mask_list) {

	mask_list.sort(SortPredicate_Descending<T>);
	return *(mask_list.begin());
}

void maskpixel_max_generator(Mat& src,Mat& new_src, int kernal_size, int channle_number) {//i should improve it to be more generalizable by use template.

	list<float> shortage, shortageB, shortageG, shortageR;
	int n_row  = -1;
	for (size_t i = 0; i < src.rows - kernal_size; i += kernal_size)
	{
		int n_col = -1;
		++n_row;
		for (size_t j = 0; j < src.cols - kernal_size; j += kernal_size)
		{
			++n_col;
			for (size_t s = 0; s < kernal_size; s++)
			{
				auto pixel_value = src.ptr<float>(i + s);
				for (size_t v = 0; v < kernal_size; v++)
				{
					switch (channle_number)
					{
					case(1):
						shortage.push_back(pixel_value[j + v]);
						//shortage.push_back(src.at<float>(i + s, j + v));//why can't use 'at'?
						break;
					case(3):
						shortageB.push_back(pixel_value[(j + v) * 3]);
						shortageG.push_back(pixel_value[(j + v) * 3 + 1]);
						shortageR.push_back(pixel_value[(j + v) * 3 + 2]);
						break;
					default:
						break;
					}
				}
			}
			if (channle_number == 1)
			{
				float maxp = (pixel_max<float>(shortage));
				new_src.ptr<float>(n_row)[n_col] = maxp;
				shortage.clear();
			}
			else
			{
				float B, G, R;
				B = (pixel_max<float>(shortageB));
				G = (pixel_max<float>(shortageG));
				R = (pixel_max<float>(shortageR));
				new_src.ptr<float>(n_row)[n_col*3] = B;
				new_src.ptr<float>(n_row)[n_col*3+1] = G;
				new_src.ptr<float>(n_row)[n_col*3+2] = R;
				shortageB.clear();
				shortageG.clear();
				shortageR.clear();
			}

		}
	}
}

void channle1(Mat& src, int kernal_size) {

	Mat new_src(src.rows / kernal_size, src.cols / kernal_size, CV_32FC1);
	maskpixel_max_generator(src, new_src, kernal_size, 1);
	imshow("pooling", new_src);
}

void channle3(Mat& src, int kernal_size) {

	Mat new_src(src.rows / kernal_size, src.cols / kernal_size, CV_32FC3);
	maskpixel_max_generator(src, new_src, kernal_size, 3);
	imshow("pooling", new_src);
}

int pooling(Mat& src, int kernal_size=3) {//defalut kernal_size=3

	//judge kernal_size is invaliable or not.
	
	if (kernal_size <= 0)
	{
		cout << "your kernal parament is wron!" << endl;
		return -1;
	}
	int row = src.rows;
	int col = src.cols;
	if (row < kernal_size || col < kernal_size)
	{
		cout << "index overload!!" << endl;
		return -1;
	}
	
	//padding image by kernal_size 
	Mat src_pad = padding(src, kernal_size, row, col);
	
	//judge the channles of image is invaliable or not. then process it.
	switch (src_pad.channels())
	{
	case(1):
		channle1(src_pad, kernal_size);
		waitKey(0);
		return 1;
		break;
	case(3):
		channle3(src_pad, kernal_size);
		waitKey(0);
		return 1;
		break;
	default:
		cout << "this processer can't solve the channle what is'not 1 or 3!" << endl;
		return -1;
		break;
	}
}

int main() {
	cout << "input your image's path:" << endl << ">>>";
	string path;
	cin >> path;
	cin.get();
	cout << endl;
	Mat origin_image = imread(path, -1);
	origin_image.convertTo(origin_image, CV_32FC3, 1/255.0);
	imshow("origine", origin_image);
	cout << "input your kernal's size(if you want to process defaultly=3, inpute \"y\"):" << endl << ">>>";
	int kernal_size;
	cin >> kernal_size;
	cin.get();
	cout << endl;
	int row = origin_image.rows;
	int col = origin_image.cols;
	if (cin.good() == 1)
	{
		cout << "setted by yourself!" << endl;
		return(pooling(origin_image, kernal_size));
	}
	cout << "setted by default!" << endl;
	return(pooling(origin_image));
}
