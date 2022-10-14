// homework.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。
//
//test new eqe
#include "pch.h"
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>
#include <set>
#include <ctime>

using namespace cv;
using namespace std;

void exch(int &a, int &b);
void least_square(vector<int> &a, vector<int> &b, double &result_a, double &result_b);//最小平方法
void DFPD_g_estimate(vector < vector<int> >&estimate_v, vector < vector<int> >&estimate_h, Mat &test);
void DFPD_color_diffrence(vector < vector<int> >  &estimate_v, vector < vector<int> >  &estimate_h, vector < vector<int> >  &estimate_v_c, vector < vector<int> >  &estimate_h_c, Mat &test);
void DFPD_sum_of_the_gradients(vector < vector<int> >  &estimate_v_c, vector < vector<int> >  &estimate_h_c, vector < vector<int> >  &estimate_v_d_total, vector < vector<int> >  &estimate_h_d_total);
void DFPD_direction_choose(vector < vector<int> >&estimate_v, vector < vector<int> >&estimate_h, vector < vector<int> >  &estimate_v_d_total, vector < vector<int> >  &estimate_h_d_total, vector < vector<int> >  &estimate_green, vector < vector<double> >  &estimate_s, Mat &test, int k = 3);//k = 3;//論文裡面定義的
void fuzzy_edge_strength(vector < vector<double> > &estimate_s, Mat &test);
void fuzzy_smooth_method(vector < vector<int> >  &estimate_green, vector < vector<double> >  &estimate_s, Mat &test, int i, int j);
void tantative(vector < vector<int> > &estimate_green, vector < vector<int> > &estimate, Mat &test, int color, int win = 5);//預設win size是5
void get_pair(vector<int> &a, vector<int> &b, vector < vector<int> >  &estimate_green, Mat &test, int i, int j, int win, int color);//color紀錄要算的顏色是blue還是red，color == 0處理紅色 color ==1 處理藍色
void get_error(double &error, double result_a, double result_b, vector < vector<int> >  &estimate_green, Mat &test, int i, int j, int win, int color);//color紀錄要算的顏色是blue還是red，color == 0處理紅色 color ==1 處理藍色
void push_estimate(double result_a_average, double result_b_average, vector < vector<int> > &estimate, vector < vector<int> > &estimate_green);
void tantative_minus(vector < vector<int> > &tempatative, vector < vector<int> > &estimate, Mat &test, int color);
void deal_with_red(vector < vector<int> > &tempatative_r_r);
void deal_with_blue(vector < vector<int> > &tempatative_b_b);
void tantative_add(vector < vector<int> > & tempatative, vector < vector<int> > &estimate);
void rusult(vector < vector<int> >  &estimate_blue, vector < vector<int> >  &estimate_green, vector < vector<int> >  &estimate_red, Mat &result_map);
Mat ConvertBGR2Bayer(Mat BGRImage);
double CPSNR(Mat &input, Mat &origin);
void test22(int &a)
{
	cout << &a << endl;
	a = 435345345;
}
//--------------------------------------【main( )函數】---------------------------------------
//          描述：控制臺應用程式的入口函數，我們的程式從這里開始執行
//-----------------------------------------------------------------------------------------------
int main()
{
	int a[2] = { 1,2 };

	int b = 5;
	cout << (&b) << endl;
	test22(b);
	cout << b;
	/*vector<Mat> list_image;
	for (int i = 1; i <= 18; i++)
		list_image.push_back(imread(to_string(i) + ".tif"));
	//for (int i = 1; i <= 12; i++)
		//list_image.push_back(imread("kodim" + to_string(i) + ".png"));

	//list_image.push_back(imread("18.tif")); //讀圖片位置

	int image_num = 0;
	vector<double> image_CPSNR;
	vector<double> time_duration;
	while (image_num < list_image.size())
	{
		Mat img = list_image.at(image_num);
		Mat dst;
		Mat test;
		dst = ConvertBGR2Bayer(img);
		copyMakeBorder(dst, test, 2, 2, 2, 2, BORDER_REFLECT_101);
		int rows = dst.rows;                     //圖片row
		int cols = dst.cols;				     //圖片col
		int new_rows = rows + 4;
		int new_cols = cols + 4;
		int k = 2;
		int win = 5;

		vector < vector<int> >  estimate_green(rows + 4, vector<int>(cols + 4));
		vector < vector<int> >  estimate_red(rows + 4, vector<int>(cols + 4));
		vector < vector<int> >  estimate_blue(rows + 4, vector<int>(cols + 4));

		vector < vector<int> >  estimate_v(rows + 4, vector<int>(cols + 4));
		vector < vector<int> >  estimate_h(rows + 4, vector<int>(cols + 4));

		vector < vector<int> >  estimate_v_c(rows + 4, vector<int>(cols + 4));
		vector < vector<int> >  estimate_h_c(rows + 4, vector<int>(cols + 4));

		vector < vector<int> >  estimate_v_d_total(rows + 4, vector<int>(cols + 4));
		vector < vector<int> >  estimate_h_d_total(rows + 4, vector<int>(cols + 4));

		vector < vector<double> >  estimate_s(rows + 4, vector<double>(cols + 4));

		vector < vector<int> >  estimate_v_d(rows + 4, vector<int>(cols + 4));
		vector < vector<int> >  estimate_h_d(rows + 4, vector<int>(cols + 4));

		vector < vector<int> >  MLRI_estimate_red(rows + 4, vector<int>(cols + 4));
		vector < vector<int> >  MLRI_estimate_blue(rows + 4, vector<int>(cols + 4));

		vector < vector<int> >  tempatative_r_r(rows + 4, vector<int>(cols + 4));
		vector < vector<int> >  tempatative_b_b(rows + 4, vector<int>(cols + 4));

		Mat result_map(rows, cols, CV_8UC3, Scalar((int)255, (int)255, (int)255));
		int begin;
		int end;
		begin = clock();
		DFPD_g_estimate(estimate_v, estimate_h, test);
		DFPD_color_diffrence(estimate_v, estimate_h, estimate_v_c, estimate_h_c, test);
		DFPD_sum_of_the_gradients(estimate_v_c, estimate_h_c, estimate_v_d_total, estimate_h_d_total);
		fuzzy_edge_strength(estimate_s, test);
		DFPD_direction_choose(estimate_v, estimate_h, estimate_v_d_total, estimate_h_d_total, estimate_green, estimate_s, test, 3); //k = 3;//論文裡面定義的
		tantative(estimate_green, estimate_red, test, 0, win); //0 red
		tantative(estimate_green, estimate_blue, test, 1, win);//1 blue

		tantative_minus(tempatative_r_r, estimate_red, test, 0);
		tantative_minus(tempatative_b_b, estimate_blue, test, 1);

		deal_with_red(tempatative_r_r);
		deal_with_blue(tempatative_b_b);

		tantative_add(tempatative_r_r, estimate_red);
		tantative_add(tempatative_b_b, estimate_blue);

		for (int i = 2; i < estimate_blue.size() - 2; i++)
			for (int j = 2; j < estimate_blue.at(0).size() - 2; j++)
			{
				if (estimate_red.at(i).at(j) < 0)
					estimate_red.at(i).at(j) = 0;
				if (estimate_red.at(i).at(j) > 255)
					estimate_red.at(i).at(j) = 255;
				if (estimate_blue.at(i).at(j) < 0)
					estimate_blue.at(i).at(j) = 0;
				if (estimate_blue.at(i).at(j) > 255)
					estimate_blue.at(i).at(j) = 255;
			}
		rusult(estimate_blue, estimate_green, estimate_red, result_map);
		end = clock();
		time_duration.push_back(double(end - begin) / CLOCKS_PER_SEC);
		imshow("原圖" + to_string(image_num + 1), img);
		//imshow("馬賽克圖", test);
		imshow("還原圖" + to_string(image_num + 1), result_map);
		//imwrite("C:\\Users\\USER\\Desktop\\de_result\\" + to_string(image_num) + ".png", result_map);
		image_CPSNR.push_back(CPSNR(img, result_map));
		image_num++;
	}
	cout << "--------------------------------------------\n";
	double total_CPSNR = 0;
	for (int i = 0; i < image_CPSNR.size(); i++)
	{
		cout << to_string(i + 1) + ".tif: " << image_CPSNR.at(i) << "     耗費: " << time_duration.at(i) << "秒" << endl;
		//cout << "kodim" + to_string(i + 1) + ".png: " << image_CPSNR.at(i) << "     耗費: " << time_duration.at(i) << "秒" << endl;
		total_CPSNR += image_CPSNR.at(i);
	}
	double CPSNR_average = total_CPSNR / image_CPSNR.size();
	cout << "--------------------------------------------\n";
	cout << "CPSNR_average: " << CPSNR_average << endl;
	waitKey();*/
	system("pause");
	return 0;
}

void least_square(vector<int> &a, vector<int> &b, double &result_a, double &result_b) //最小平方法
{
	int left_top = 0, right_top = 0, left_down = 0, right_down = 0;
	int top = 0;//右邊矩陣相乘的上部(AT*B)
	int down = 0;//右邊矩陣相乘的下部(AT*B)
	for (int p = 0; p < a.size(); p++)
	{
		left_top += pow(a.at(p), 2);
		right_top += a.at(p);
		top += a.at(p)*b.at(p);
		down += b.at(p);
	}
	left_down = right_top;
	right_down = a.size();
	int det = abs(left_top*right_down - right_top * left_down);
	double value = (double)1 / (double)det;
	exch(left_top, right_down);
	right_top = -right_top;
	left_down = -left_down;
	result_a = value * (double)(left_top * top + right_top * down);
	result_b = value * (double)(left_down * top + right_down * down);
}

void exch(int &a, int &b)
{
	int temp;
	temp = a;
	a = b;
	b = temp;
}

Mat ConvertBGR2Bayer(Mat BGRImage) {
	/*
	Assuming a Bayer filter that looks like this:

	# // 0  1  2  3  4  5
	/////////////////////
	0 // B  G  B  G  B  G
	1 // G  R  G  R  G  R
	2 // B  G  B  G  B  G
	3 // G  R  G  R  G  R
	4 // B  G  B  G  B  G
	5 // G  R  G  R  G  R

	*/

	Mat BayerImage(BGRImage.rows, BGRImage.cols, CV_8UC1);

	int channel;

	for (int row = 0; row < BayerImage.rows; row++)
	{
		for (int col = 0; col < BayerImage.cols; col++)
		{
			if (row % 2 == 0)
			{
				//even columns and even rows = red = channel:2
				//even columns and uneven rows = green = channel:1
				channel = (col % 2 == 0) ? 2 : 1;
			}
			else
			{
				//uneven columns and even rows = green = channel:1
				//uneven columns and uneven rows = blue = channel:0
				channel = (col % 2 == 0) ? 1 : 0;
			}

			BayerImage.at<uchar>(row, col) = BGRImage.at<Vec3b>(row, col).val[channel];
		}
	}

	return BayerImage;
}

double CPSNR(Mat &input, Mat &origin)
{
	double total = 0;
	double green_total = 0;
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
		{
			total += sqrt((double)pow((input.at<Vec3b>(i, j)[0] - origin.at<Vec3b>(i, j)[0]), 2));
			total += sqrt((double)pow((input.at<Vec3b>(i, j)[1] - origin.at<Vec3b>(i, j)[1]), 2));
			total += sqrt((double)pow((input.at<Vec3b>(i, j)[2] - origin.at<Vec3b>(i, j)[2]), 2));
		}
	total = total / (3 * input.rows* input.cols);
	total = (double)255 * 255 / total;

	double result = 10 * log10(total);

	cout << "CPSNR: " << result << endl;
	return result;
}

void DFPD_g_estimate(vector < vector<int> >&estimate_v, vector < vector<int> >&estimate_h, Mat &test)
{
	int row = estimate_h.size();
	int col = estimate_h.at(0).size();
	for (int i = 2; i < row - 2; i++)
		for (int j = 2; j < col - 2; j++)
		{
			if ((i + j) % 2 == 0)
			{
				estimate_h.at(i).at(j) = (((int)test.at<uchar>(i, j + 1) + (int)test.at<uchar>(i, j - 1)) / 2);//no_color_diffrence
				estimate_v.at(i).at(j) = (((int)test.at<uchar>(i + 1, j) + (int)test.at<uchar>(i - 1, j)) / 2);//no_color_diffrence
				if (estimate_h.at(i).at(j) > 255)
					estimate_h.at(i).at(j) = 255;
				if (estimate_h.at(i).at(j) < 0)
					estimate_h.at(i).at(j) = 0;

				if (estimate_v.at(i).at(j) > 255)
					estimate_v.at(i).at(j) = 255;
				if (estimate_v.at(i).at(j) < 0)
					estimate_v.at(i).at(j) = 0;
			}
		}
}

void DFPD_color_diffrence(vector < vector<int> >  &estimate_v, vector < vector<int> >  &estimate_h, vector < vector<int> >  &estimate_v_c, vector < vector<int> >  &estimate_h_c, Mat &test)
{
	int row = estimate_h.size();
	int col = estimate_h.at(0).size();
	for (int i = 2; i < row - 2; i++)
		for (int j = 2; j < col - 2; j++)
		{
			if ((i + j) % 2 == 0)
			{
				estimate_h_c.at(i).at(j) = estimate_h.at(i).at(j);//no_color_diffrence
				estimate_v_c.at(i).at(j) = estimate_v.at(i).at(j);//no_color_diffrence
			}
		}
}

void DFPD_sum_of_the_gradients(vector < vector<int> >  &estimate_v_c, vector < vector<int> >  &estimate_h_c, vector < vector<int> >  &estimate_v_d_total, vector < vector<int> >  &estimate_h_d_total)
{
	int row = estimate_h_c.size();
	int col = estimate_h_c.at(0).size();
	int first, second, third, total; //second 要乘以三,這步驟嚴格上來說做了兩部
	for (int i = 2; i < row - 2; i++)
		for (int j = 2; j < col - 2; j++)
		{
			int first = 0, second = 0, third = 0, total = 0;
			if ((i + j) % 2 == 0)
			{
				if (i - 2 > 2 && j + 2 < col - 2)
					first = abs(estimate_h_c.at(i - 2).at(j) - estimate_h_c.at(i - 2).at(j + 2));
				if (j + 2 < col - 2)
					second = abs(3 * (estimate_h_c.at(i).at(j) - estimate_h_c.at(i).at(j + 2)));  //這是最中間值  要乘以三倍
				if (i + 2 < row - 2 && j + 2 < col - 2)
					third = abs(estimate_h_c.at(i + 2).at(j) - estimate_h_c.at(i + 2).at(j + 2));
				total = first + second + third;
				estimate_h_d_total.at(i).at(j) = total;
				first = 0;
				second = 0;
				third = 0;
				total = 0;
				if (j - 2 > 2 && i + 2 < row - 2)
					first = abs(estimate_v_c.at(i).at(j - 2) - estimate_v_c.at(i + 2).at(j - 2));
				if (i + 2 < row - 2)
					second = abs(3 * (estimate_v_c.at(i).at(j) - estimate_v_c.at(i + 2).at(j)));  //這是最中間值  要乘以三倍
				if (j + 2 < col - 2 && i + 2 < row - 2)
					third = abs(estimate_v_c.at(i).at(j + 2) - estimate_v_c.at(i + 2).at(j + 2));
				total = first + second + third;
				estimate_v_d_total.at(i).at(j) = total;
			}
		}
}

void DFPD_direction_choose(vector <vector<int> > &estimate_v, vector < vector<int> >&estimate_h, vector < vector<int> >  &estimate_v_d_total, vector < vector<int> >  &estimate_h_d_total, vector < vector<int> >  &estimate_green, vector < vector<double> >  &estimate_s, Mat &test, int k)   //k = 3;//論文裡面定義的
{
	int row = estimate_v_d_total.size();
	int col = estimate_v_d_total.at(0).size();
	for (int i = 2; i < row - 2; i++)
		for (int j = 2; j < col - 2; j++)
		{
			if ((i + j) % 2 == 0)
			{
				if (estimate_v_d_total.at(i).at(j) == 0 || ((double)estimate_h_d_total.at(i).at(j) / (double)estimate_v_d_total.at(i).at(j)) > 3)
					estimate_green.at(i).at(j) = estimate_v.at(i).at(j);
				else if (estimate_h_d_total.at(i).at(j) == 0 || ((double)estimate_h_d_total.at(i).at(j) / (double)estimate_v_d_total.at(i).at(j)) < ((double)1 / (double)3))
					estimate_green.at(i).at(j) = estimate_h.at(i).at(j);
				else
					fuzzy_smooth_method(estimate_green, estimate_s, test, i, j);
			}
			else
				estimate_green.at(i).at(j) = (int)test.at<uchar>(i, j);
		}
}

void fuzzy_edge_strength(vector < vector<double> > &estimate_s, Mat &test)
{
	int row = estimate_s.size();
	int col = estimate_s.at(0).size();
	int biggest = 0;   //算u(i,j)隸屬矩陣
	for (int i = 2; i < row - 2; i++)
		for (int j = 2; j < col - 2; j++)
		{
			estimate_s.at(i).at(j) = abs((int)test.at<uchar>(i, j - 1) - (int)test.at<uchar>(i, j + 1)) + abs((int)test.at<uchar>(i - 1, j) - (int)test.at<uchar>(i + 1, j)) + (abs((int)test.at<uchar>(i - 1, j - 1) - (int)test.at<uchar>(i + 1, j + 1)) / 2) + (abs((int)test.at<uchar>(i + 1, j - 1) - (int)test.at<uchar>(i - 1, j + 1)) / 2);
			if (biggest < estimate_s.at(i).at(j))
				biggest = estimate_s.at(i).at(j);
		}

	for (int i = 2; i < row - 2; i++)
		for (int j = 2; j < col - 2; j++)
			estimate_s.at(i).at(j) = 1 - (estimate_s.at(i).at(j) / biggest);
}

void fuzzy_smooth_method(vector < vector<int> >  &estimate_green, vector < vector<double> >  &estimate_s, Mat &test, int i, int j)
{
	int fuzzy_g = int((estimate_s.at(i - 1).at(j)*(int)test.at<uchar>(i - 1, j) + estimate_s.at(i + 1).at(j)*(int)test.at<uchar>(i + 1, j) + estimate_s.at(i).at(j - 1)*(int)test.at<uchar>(i, j - 1) + estimate_s.at(i).at(j + 1)*(int)test.at<uchar>(i, j + 1)) / (estimate_s.at(i - 1).at(j) + estimate_s.at(i + 1).at(j) + estimate_s.at(i).at(j - 1) + estimate_s.at(i).at(j + 1)));
	int fuzzy_b_r = ((4 * (estimate_s.at(i).at(j)*(int)test.at<uchar>(i, j))) + (estimate_s.at(i - 2).at(j)*(int)test.at<uchar>(i - 2, j)) + (estimate_s.at(i + 2).at(j)*(int)test.at<uchar>(i + 2, j)) + (estimate_s.at(i).at(j - 2)*(int)test.at<uchar>(i, j - 2)) + (estimate_s.at(i).at(j + 2)*(int)test.at<uchar>(i, j + 2))) / (4 * (estimate_s.at(i).at(j)) + estimate_s.at(i - 2).at(j) + estimate_s.at(i + 2).at(j) + estimate_s.at(i).at(j - 2) + estimate_s.at(i).at(j + 2));
	estimate_green.at(i).at(j) = fuzzy_g;//no_color_diffrence
	if (estimate_green.at(i).at(j) < 0)
		estimate_green.at(i).at(j) = 0;
	if (estimate_green.at(i).at(j) > 255)
		estimate_green.at(i).at(j) = 255;
}

void tantative(vector < vector<int> > &estimate_green, vector < vector<int> > &estimate, Mat &test, int color, int win)//win只能是奇數 //color == 0處理紅色 color ==1 處理藍色
{
	if (win % 2 == 1)
		win += 1;

	vector <double>least_square_value_a;
	vector <double>least_square_value_b;
	vector <double>least_square_value_error;
	int row = estimate_green.size();
	int col = estimate_green.at(0).size();
	for (int i = 2 + (win / 2) + color; i < row - 2 - (win / 2); i = i + win + 1) //算紅色
		for (int j = 2 + (win / 2) + color; j < col - 2 - (win / 2); j = j + win + 1)
		{
			vector<int> a;
			vector<int> b;
			double result_a = 0;
			double result_b = 0;
			double error = 0;

			get_pair(a, b, estimate_green, test, i, j, win, 0);
			least_square(a, b, result_a, result_b);
			get_error(error, result_a, result_b, estimate_green, test, i, j, win, 0);

			if (isnan(result_a) || isnan(result_b) || isnan(error))
				continue;
			least_square_value_a.push_back(result_a);
			least_square_value_b.push_back(result_b);
			least_square_value_error.push_back(error);
		}
	double total_error = 0;
	double result_a_average = 0;
	double result_b_average = 0;

	for (int n = 0; n < least_square_value_error.size(); n++)
	{
		total_error += least_square_value_error.at(n);
		result_a_average += least_square_value_error.at(n)*least_square_value_a.at(n);
		result_b_average += least_square_value_error.at(n)*least_square_value_b.at(n);
	}

	result_a_average = result_a_average / total_error;
	result_b_average = result_b_average / total_error;

	push_estimate(result_a_average, result_b_average, estimate, estimate_green);
}

void get_pair(vector<int> &a, vector<int> &b, vector < vector<int> >  &estimate_green, Mat &test, int i, int j, int win, int color)//color == 0處理紅色 color ==1 處理藍色
{
	if ((i + j) % 2 == 0)
		if (i % 2 == color) //處理red
		{
			for (int k = i - (win / 2); k < i - 2 + win; k++)
				for (int m = j - (win / 2); m < j - 2 + win; m++)
					if ((k + m) % 2 == 0)
						if (k % 2 == color)
						{
							a.push_back(estimate_green.at(k).at(m));
							b.push_back((int)test.at<uchar>(k, m));
						}
		}
}

void get_error(double &error, double result_a, double result_b, vector < vector<int> >  &estimate_green, Mat &test, int i, int j, int win, int color)//color == 0處理紅色 color ==1 處理藍色
{
	int sum = 0;
	if ((i + j) % 2 == 0)
		if (i % 2 == color) //處理red
		{
			for (int k = i - 2; k < i - 2 + win; k++)
				for (int m = j - 2; m < j - 2 + win; m++)
					if ((k + m) % 2 == 0)
						if (k % 2 == color)
						{
							if (isnan(result_a) || isnan(result_b)) //least_square算det的時候det等於0不能inverse
								break;//這裡只是在回圈內break;
							error += pow(abs((int)test.at<uchar>(k, m) - (result_a*estimate_green.at(k).at(m) + result_b)), 2);
							sum++;
						}
			error = error / sum;
		}
}

void push_estimate(double result_a_average, double result_b_average, vector < vector<int> > &estimate_red, vector < vector<int> > &estimate_green)
{
	int row = estimate_red.size();
	int col = estimate_red.at(0).size();
	for (int i = 2; i < row - 2; i++)
	{
		for (int j = 2; j < col - 2; j++)
		{
			if ((int)(result_a_average * estimate_green.at(i).at(j) + result_b_average) > 255)
				estimate_red.at(i).at(j) = 255;
			else if ((int)(result_a_average * estimate_green.at(i).at(j) + result_b_average) < 0)
				estimate_red.at(i).at(j) = 0;
			else
				estimate_red.at(i).at(j) = (int)(result_a_average * estimate_green.at(i).at(j) + result_b_average);
		}
	}
}

void tantative_minus(vector < vector<int> > &tempatative, vector < vector<int> > &estimate, Mat &test, int color)
{
	int row = estimate.size();
	int col = estimate.at(0).size();
	for (int i = 2; i < row - 2; i++)
	{
		for (int j = 2; j < col - 2; j++)
		{
			if ((i + j) % 2 == 0)
				if (i % 2 == color) //處理red
					tempatative.at(i).at(j) = (int)test.at<uchar>(i, j) - estimate.at(i).at(j);
		}
	}
}

void deal_with_red(vector < vector<int> > &tempatative_r_r)
{
	int row = tempatative_r_r.size();
	int col = tempatative_r_r.at(0).size();
	for (int i = 2; i < row - 2; i++)
	{
		for (int j = 2; j < col - 2; j++)
		{
			if ((i + j) % 2 == 0)
			{
				if (i % 2 == 1) //處理藍色
					tempatative_r_r.at(i).at(j) = (tempatative_r_r.at(i - 1).at(j - 1) + tempatative_r_r.at(i - 1).at(j + 1) + tempatative_r_r.at(i + 1).at(j - 1) + tempatative_r_r.at(i + 1).at(j + 1)) / 4;
			}
			else
			{
				if (i % 2 == 0)//處理綠色
					tempatative_r_r.at(i).at(j) = (tempatative_r_r.at(i).at(j - 1) + tempatative_r_r.at(i).at(j + 1)) / 2;
				else
					tempatative_r_r.at(i).at(j) = (tempatative_r_r.at(i - 1).at(j) + tempatative_r_r.at(i + 1).at(j + 1)) / 2;
			}
		}
	}
}

void deal_with_blue(vector < vector<int> > &tempatative_b_b)
{
	int row = tempatative_b_b.size();
	int col = tempatative_b_b.at(0).size();

	for (int i = 2; i < row - 2; i++)
	{
		tempatative_b_b.at(i).at(1) = tempatative_b_b.at(i).at(3);
		tempatative_b_b.at(i).at(col - 1) = tempatative_b_b.at(i).at(col - 3);
	}

	for (int i = 2; i < col - 2; i++)
	{
		tempatative_b_b.at(1).at(i) = tempatative_b_b.at(3).at(i);
		tempatative_b_b.at(row - 1).at(i) = tempatative_b_b.at(row - 3).at(i);
	}
	tempatative_b_b.at(1).at(1) = tempatative_b_b.at(3).at(3);
	tempatative_b_b.at(row - 1).at(1) = tempatative_b_b.at(row - 3).at(3);
	tempatative_b_b.at(1).at(col - 1) = tempatative_b_b.at(3).at(col - 3);
	tempatative_b_b.at(row - 1).at(col - 1) = tempatative_b_b.at(row - 3).at(col - 3);  //以上是對稱用的

	for (int i = 2; i < row - 2; i++)
	{
		for (int j = 2; j < col - 2; j++)
		{
			if ((i + j) % 2 == 0)
			{
				if (i % 2 == 0) //處理紅色
					tempatative_b_b.at(i).at(j) = (tempatative_b_b.at(i - 1).at(j - 1) + tempatative_b_b.at(i - 1).at(j + 1) + tempatative_b_b.at(i + 1).at(j - 1) + tempatative_b_b.at(i + 1).at(j + 1)) / 4;
			}
			else
			{
				if (i % 2 == 0)
					tempatative_b_b.at(i).at(j) = (tempatative_b_b.at(i - 1).at(j) + tempatative_b_b.at(i + 1).at(j)) / 2;
				else
					tempatative_b_b.at(i).at(j) = (tempatative_b_b.at(i).at(j - 1) + tempatative_b_b.at(i).at(j + 1)) / 2;
			}
		}
	}
}

void tantative_add(vector < vector<int> > & tempatative, vector < vector<int> > &estimate)
{
	int row = estimate.size();
	int col = estimate.at(0).size();
	for (int i = 2; i < row - 2; i++)
		for (int j = 2; j < col - 2; j++)
			estimate.at(i).at(j) = estimate.at(i).at(j) + tempatative.at(i).at(j);
}

void rusult(vector < vector<int> >  &estimate_blue, vector < vector<int> >  &estimate_green, vector < vector<int> >  &estimate_red, Mat &result_map)
{
	for (int i = 0; i < result_map.rows; i++)
		for (int j = 0; j < result_map.cols; j++)
		{
			result_map.at<Vec3b>(i, j)[0] = estimate_blue.at(i + 2).at(j + 2);
			result_map.at<Vec3b>(i, j)[1] = estimate_green.at(i + 2).at(j + 2);
			result_map.at<Vec3b>(i, j)[2] = estimate_red.at(i + 2).at(j + 2);
		}
}