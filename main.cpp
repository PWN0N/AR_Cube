#include <iostream>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>

using namespace std;
using namespace cv;

RNG rng(12345);

struct Index
{
    int a1;
    int a2;
    int a3;
};




//获取轮廓的中心点
Point Center_cal(vector<vector<Point> > contours,int i)
{
    int centerx=0,centery=0,n=contours[i].size();
    //在提取的小正方形的边界上每隔周长个像素提取一个点的坐标，
    //求所提取四个点的平均坐标（即为小正方形的大致中心）
    centerx = (contours[i][n/4].x + contours[i][n*2/4].x + contours[i][3*n/4].x + contours[i][n-1].x)/4;
    centery = (contours[i][n/4].y + contours[i][n*2/4].y + contours[i][3*n/4].y + contours[i][n-1].y)/4;
    Point point1=Point(centerx,centery);
    return point1;
}
#define onepic 1
//#define DEBUG
Index in;
vector<Index> vin;
int main( int argc, char** argv[] )
{

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    string path[20];
    int k = 0;
    for(int i = 1;i<=5;i++)
    {
        for(int j = 7;j<=10;j++)
        {
            string s1 = "./qrcodepic/";

            string s2 ;
            stringstream ss;
            ss << i;
            s2 = ss.str();

            stringstream ss2;
            string s3;
            ss2 << j%10;
            s3 = ss2.str();

            string s4 = "0.png";

            path[k++] = s1+s2+s3+s4;
        }
    }
#if onepic == 1
    k = 1;
#endif
    for(int kk = 0;kk<k;kk++)
    {
#if onepic == 1
        //Mat src = imread( "qrcodepic\\590.png", 1 );
        Mat src = imread( "./pic/456.jpg", 1 );


        if(src.empty())
        {
            cout << "Can not load image！" << endl;
            return 0;
        }

        Mat src_all=src.clone();

        //彩色图转灰度图
        Mat src_gray;
        cvtColor( src, src_gray, CV_BGR2GRAY );

        //对图像进行平滑处理
        //blur( src_gray, src_gray, Size(3,3) );

        //使灰度图象直方图均衡化
        //equalizeHist( src_gray, src_gray );

#ifdef DEBUG
        namedWindow("src_gray");
        imshow("src_gray",src_gray);         //灰度图
#endif

        //指定112阀值进行二值化
        Mat threshold_output;
        threshold( src_gray, threshold_output, 112, 255, THRESH_BINARY );

#ifdef DEBUG
        //namedWindow("二值化后输出");
        //imshow("二值化后输出",threshold_output);   //二值化后输出
#endif

        //需要的变量定义
        Scalar color = Scalar(1,1,255 );
        vector<vector<Point>> contours,contours2;
        vector<Vec4i> hierarchy;
        Mat drawing = Mat::zeros( src.size(), CV_8UC3 );
        //Mat drawing2 = Mat::zeros( src.size(), CV_8UC3 );
        Mat drawingAllContours = Mat::zeros( src.size(), CV_8UC3 );

        //利用二值化输出寻找轮廓
        findContours(threshold_output, contours, hierarchy,  CV_RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0) );

        //寻找轮廓的方法
        int tempindex1 = 0;
        int tempindex2 = 0;

        for(int i = 0;i<contours.size();i++)
        {
            if(hierarchy[i][2] == -1)
                continue;
            else
                tempindex1 = hierarchy[i][2];                //第一个子轮廓的索引

            if(hierarchy[tempindex1][2] == -1)
                continue;
            else
            {
                tempindex2 = hierarchy[tempindex1][2];        //第二个子轮廓的索引
                //记录搜索到的有两个子轮廓的轮廓并把他们的编号存储
                in.a1 = i;
                in.a2 = tempindex1;
                in.a3 = tempindex2;
                vin.push_back(in);
            }
        }

        //按面积比例搜索
        vector<Index>::iterator it;
        for(it = vin.begin();it != vin.end();)
        {
            vector<Point> out1Contours = contours[it->a1];
            vector<Point> out2Contours = contours[it->a2];
            double lenth1 = arcLength(out1Contours,1);
            double lenth2 = arcLength(out2Contours,1);
            if(abs(lenth1/lenth2-2)>1)
            {
                it = vin.erase(it);
            }
            else
            {
                drawContours( drawing, contours, it->a1,  CV_RGB(255,255,255) , CV_FILLED, 8); //rng.uniform(0,255)
                it++;
            }
        }

        //填充的方式画出三个黑色定位角的轮廓
        //for(int i=0; i<contours2.size(); i++)
        //	drawContours( drawing2, contours2, i,  CV_RGB(rng.uniform(100,255),rng.uniform(100,255),rng.uniform(100,255)) , -1, 4, hierarchy[k][2], 0, Point() );

        //获取三个定位角的中心坐标
        Point point[3];
        int i = 0;
        vector<Point> pointthree;
        for(it = vin.begin(),i = 0;it != vin.end();i++,it++)
        {
            point[i] = Center_cal( contours, it->a1 );
            pointthree.push_back(point[i]);
        }

        if(pointthree.size() <3)
        {
            cout << "找到的定位角点不足3个"<<endl;
            return 0;
        }

        //计算轮廓的面积，计算定位角的面积，从而计算出边长
        double area = contourArea(contours[vin[0].a1]);
        int area_side = cvRound (sqrt (double(area)));
        for(int i=0; i<3; i++)
        {
            //画出三个定位角的中心连线
            line(drawing,point[i%3],point[(i+1)%3],color,area_side/10,8);
        }

        //清除找到的3个点,以便处理下一幅图片使用
        vin.clear();

        //由3个定位角校正图片
        //=========================================
        //找到角度最大的点
        double ca[2];
        double cb[2];

        ca[0] =  pointthree[1].x - pointthree[0].x;
        ca[1] =  pointthree[1].y - pointthree[0].y;
        cb[0] =  pointthree[2].x - pointthree[0].x;
        cb[1] =  pointthree[2].y - pointthree[0].y;
        double angle1 = 180/3.1415*acos((ca[0]*cb[0]+ca[1]*cb[1])/(sqrt(ca[0]*ca[0]+ca[1]*ca[1])*sqrt(cb[0]*cb[0]+cb[1]*cb[1])));
        double ccw1;
        if(ca[0]*cb[1] - ca[1]*cb[0] > 0) ccw1 = 0;
        else ccw1 = 1;

        ca[0] =  pointthree[0].x - pointthree[1].x;
        ca[1] =  pointthree[0].y - pointthree[1].y;
        cb[0] =  pointthree[2].x - pointthree[1].x;
        cb[1] =  pointthree[2].y - pointthree[1].y;
        double angle2 = 180/3.1415*acos((ca[0]*cb[0]+ca[1]*cb[1])/(sqrt(ca[0]*ca[0]+ca[1]*ca[1])*sqrt(cb[0]*cb[0]+cb[1]*cb[1])));
        double ccw2;
        if(ca[0]*cb[1] - ca[1]*cb[0] > 0) ccw2 = 0;
        else ccw2 = 1;

        ca[0] =  pointthree[1].x - pointthree[2].x;
        ca[1] =  pointthree[1].y - pointthree[2].y;
        cb[0] =  pointthree[0].x - pointthree[2].x;
        cb[1] =  pointthree[0].y - pointthree[2].y;
        double angle3 = 180/3.1415*acos((ca[0]*cb[0]+ca[1]*cb[1])/(sqrt(ca[0]*ca[0]+ca[1]*ca[1])*sqrt(cb[0]*cb[0]+cb[1]*cb[1])));
        double ccw3;
        if(ca[0]*cb[1] - ca[1]*cb[0] > 0) ccw3 = 0;
        else ccw3 = 1;

        CvPoint2D32f poly[4];
        if(angle3>angle2 && angle3>angle1)
        {
            if(ccw3)
            {
                poly[1] = pointthree[1];
                poly[3] = pointthree[0];
            }
            else
            {
                poly[1] = pointthree[0];
                poly[3] = pointthree[1];
            }
            poly[0] = pointthree[2];
        }
        else if(angle2>angle1 && angle2>angle3)
        {
            if(ccw2)
            {
                poly[1] = pointthree[0];
                poly[3] = pointthree[2];
            }
            else
            {
                poly[1] = pointthree[2];
                poly[3] = pointthree[0];
            }
            poly[0] = pointthree[1];

        }
        else if(angle1>angle2 && angle1 > angle3)
        {
            if(ccw1)
            {
                poly[1] = pointthree[1];
                poly[3] = pointthree[2];
            }
            else
            {
                poly[1] = pointthree[2];
                poly[3] = pointthree[1];
            }
            poly[0] = pointthree[0];
        }

        Point2f polyThrPts[3];
        polyThrPts[0]=poly[3];
        polyThrPts[1]=poly[0];
        polyThrPts[2]=poly[1];

        CvPoint2D32f trans[4];
        int temp = 50;
        trans[0] = Point2f(0+temp,0+temp);
        trans[1] = Point2f(0+temp,100+temp);
        trans[2] = Point2f(100+temp,100+temp);
        trans[3] = Point2f(100+temp,0+temp);

        Point2f transThrPts[3];
        transThrPts[0]=trans[3];
        transThrPts[1]=trans[0];
        transThrPts[2]=trans[1];

        //获取透视投影变换矩阵

        Mat transMat = getAffineTransform(polyThrPts, transThrPts);

        Mat transback = getAffineTransform(transThrPts , polyThrPts);


        poly[2].x = (int)(transback.at<double>(0, 0) * (double)trans[2].x
                          + transback.at<double>(0, 1) * (double)trans[2].y
                          + transback.at<double>(0, 2));
        poly[2].y = (int)(transback.at<double>(1, 0) * (double)trans[2].x
                          + transback.at<double>(1, 1) * (double)trans[2].y
                          + transback.at<double>(1, 2));


        //计算变换结果

        Mat src_img(src_all);
        Mat transedImg = Mat::zeros(src_img.rows, src_img.cols, src_img.type());
        warpAffine(src_img, transedImg, transMat, transedImg.size());

        //=========================================

#ifdef DEBUG
        namedWindow("透视变换后的图");
        imshow("透视变换后的图",transedImg);         //透视变换后的图

        drawContours( drawingAllContours, contours, -1,  CV_RGB(255,255,255) , 1, 8);
        namedWindow("DrawingAllContours");
        imshow( "DrawingAllContours", drawingAllContours );

        namedWindow(pathtemp);
        imshow(pathtemp , drawing );    //3个角点填充

#endif

        //接下来要框出这整个二维码
        Mat gray_all,threshold_output_all;
        vector<vector<Point> > contours_all;
        vector<Vec4i> hierarchy_all;
        cvtColor( drawing, gray_all, CV_BGR2GRAY );

        threshold( gray_all, threshold_output_all, 45, 255, THRESH_BINARY );

        findContours( threshold_output_all, contours_all, hierarchy_all,  RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0) );//RETR_EXTERNAL表示只寻找最外层轮廓

        Point2f fourPoint2f[4];
        //求最小包围矩形
        RotatedRect rectPoint = minAreaRect(contours_all[0]);

        //将rectPoint变量中存储的坐标值放到 fourPoint的数组中
        rectPoint.points(fourPoint2f);
        for (int i = 0; i < 4; i++)
        {
            line(src_all, fourPoint2f[i%4], fourPoint2f[(i + 1)%4],
                 Scalar(20,21,237), 3);
            circle(src_all, poly[i%4], 5, Scalar(0, 255, 0), 5, 8, 0);
        }
//        circle(src_all, poly[2], 5, Scalar(0, 255, 0), 5, 8, 0);
#ifdef DEBUG
        namedWindow(pathtemp);
        imshow(pathtemp , src_all );
#endif
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        std::cout<< "frame_times = "<<ttrack<<"s"<<std::endl;
        waitKey(0);

    }
    waitKey();
    return 0;
}