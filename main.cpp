
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace cv;
using namespace std;
using namespace Eigen;
//找到所提取轮廓的中心点
//在提取的中心小正方形的边界上每隔周长个像素提取一个点的坐标，求所提取四个点的平均坐标（即为小正方形的大致中心）
Point Center_cal(vector<vector<Point> > contours,int i);
void getQRCodeCorners(Mat src,vector<Point>& QRCodeCorners);

int main( int argc, char** argv[] )
{
    Mat src = imread( "/home/jonathan/QRCode/7.jpg", 1 );
    resize(src,src,Size(800,600));//标准大小
    vector<Point> QRcorner;

    getQRCodeCorners(src,QRcorner);



    line(src,QRcorner[0],QRcorner [1],Scalar(0,0,255),2);
    line(src,QRcorner [1],QRcorner [2],Scalar(0,255,0),2);
    line(src,QRcorner [2],QRcorner[3],Scalar(255,0,0),2);
    line(src,QRcorner [3],QRcorner[0],Scalar(255,0,255),2);
    circle(src,QRcorner[0],5,Scalar(0,0,255),3,8,0);
    circle(src,QRcorner[1],5,Scalar(0,0,255),3,8,0);
    circle(src,QRcorner[2],5,Scalar(0,0,255),3,8,0);
    circle(src,QRcorner[3],5,Scalar(0,0,255),3,8,0);
    imshow("sed",src);
    waitKey(0);

    Matrix<float,2,3> pr;
    pr << 1,0,0,
            0,1,0 ;
    Matrix<float,3,3> K;
    K << 3266.96772,0,2013.15663,
            0,3267.10869,1482.03236,
            0,0,1;
    Matrix<float,3,4> Qici ;
    Qici << QRcorner[0].x,QRcorner[1].x,QRcorner[2].x,QRcorner[3].x,
            QRcorner[0].y,QRcorner[1].y,QRcorner[2].y,QRcorner[3].y,
            1,1,1,1;
    Matrix<float,3,4> CameraPoint = K.inverse()*Qici;

    Matrix<float,4,2> p;
    Matrix<float,2,4> ptemp;
    ptemp = (pr*CameraPoint);
//    p = ptemp.transpose();
    p = ptemp.transpose();

    cout << p << endl;

    return 0;


//    Mat p = (pr*());K\qici
//                         H = est_homography(corner_pts,p);
//                         [proj_pts, pos{i}, rot{i}] = ar_cube(H,render_points,K);
//    % Draw a coordinate frame
//    frame = 0.05 * [0 0 0; 1 0 0; 0 1 0; 0 0 1];
//    frame = bsxfun(@plus,frame * rot{i}',pos{i}') * K';
//    frame(:, 1) = frame(:, 1) ./ frame(:, 3);
//    frame(:, 2) = frame(:, 2) ./ frame(:, 3);
//    frame = frame(:, 1:2);
//    generated_imgs{i} = insertShape(generated_imgs{i}, ...
//    'Line',[frame(1,:) frame(2,:)], ...
//    'Color','red','LineWidth',4);
//    generated_imgs{i} = insertShape(generated_imgs{i}, ...
//    'Line',[frame(1,:) frame(3,:)], ...
//    'Color','green','LineWidth',4);
//    generated_imgs{i} = insertShape(generated_imgs{i}, ...
//    'Line',[frame(1,:) frame(4,:)], ...
//    'Color','blue','LineWidth',4);
//
//    % Copy the RGB values from the logo_img to the video frame
//    generated_imgs{i} = draw_ar_cube(proj_pts,generated_imgs{i});
}
Point Center_cal(vector<vector<Point> > contours,int i)
{
    int centerx=0,centery=0,n=contours[i].size();
    centerx = (contours[i][n/4].x + contours[i][n*2/4].x + contours[i][3*n/4].x + contours[i][n-1].x)/4;
    centery = (contours[i][n/4].y + contours[i][n*2/4].y + contours[i][3*n/4].y + contours[i][n-1].y)/4;
    Point point1=Point(centerx,centery);
    return point1;
}
void getQRCodeCorners(Mat src,vector<Point>& QRCodeCorners){

    Mat src_gray;
    Mat src_all=src.clone();
    Mat threshold_output;
    vector<vector<Point> > contours,contours2;
    vector<Vec4i> hierarchy;
    //预处理
    cvtColor( src, src_gray, CV_BGR2GRAY );
    blur( src_gray, src_gray, Size(3,3) ); //模糊，去除毛刺
    threshold( src_gray, threshold_output, 100, 255, THRESH_OTSU );
    //寻找轮廓
    //第一个参数是输入图像 2值化的
    //第二个参数是内存存储器，FindContours找到的轮廓放到内存里面。
    //第三个参数是层级，**[Next, Previous, First_Child, Parent]** 的vector
    //第四个参数是类型，采用树结构
    //第五个参数是节点拟合模式，这里是全部寻找
    findContours( threshold_output, contours, hierarchy,  CV_RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0) );
    //轮廓筛选
    int c=0,ic=0,area=0;
    int parentIdx=-1;
    for( int i = 0; i< contours.size(); i++ )
    {
        //hierarchy[i][2] != -1 表示不是最外面的轮廓
        if (hierarchy[i][2] != -1 && ic==0)
        {
            parentIdx = i;
            ic++;
        }
        else if (hierarchy[i][2] != -1)
        {
            ic++;
        }
            //最外面的清0
        else if(hierarchy[i][2] == -1)
        {
            ic = 0;
            parentIdx = -1;
        }
        //找到定位点信息
        if ( ic >= 2)
        {
            contours2.push_back(contours[parentIdx]);
            ic = 0;
            parentIdx = -1;
        }
    }
    //填充定位点
    for(int i=0; i<contours2.size(); i++)

        drawContours( src_all, contours2, i,  CV_RGB(0,255,0) , 2 );

    //连接定位点
    Point point[3];

    for(int i=0; i<contours2.size(); i++)
    {
        point[i] = Center_cal( contours2, i );
        QRCodeCorners.push_back(point[i]);
    }

}

Matrix<float,3,3> getHomoMatrix(Matrix<float,4,2>  cube_pts,Matrix<float,4,2> world_pts){

    Matrix<float,8,9>  A ;
    Matrix<float,1,9> ax ;
    Matrix<float,1,9> ay ;

    for (int i = 0; i < 3 ; ++i) {
        ax << (-1)*cube_pts(i,1),(-1)*cube_pts(i,2),-1,0,0,0,cube_pts(i,1)*world_pts(i,1),cube_pts(i,2)*world_pts(i,1),world_pts(i,1);
        ay << 0,0,0,(-1)*cube_pts(i,1),(-1)*cube_pts(i,2),-1,cube_pts(i,1)*world_pts(i,2),cube_pts(i,2)*world_pts(i,2),world_pts(i,2);
        int m = 2*i-1;
        A(m:m+1,:) = [ax;ay];

    }

    JacobiSVD<Eigen::MatrixXf> svd(A, ComputeThinU | ComputeThinV );
    Matrix3f V = svd.matrixV(), U = svd.matrixU();
    Matrix3f  S = U.inverse() * A * V.transpose().inverse();

    Matrix<float,3,3> H = [reshape(V(:,end),[3,3])]';
    return H;
}

void getARcube(Matrix3f H,Matrix3f render_points,Matrix3f K){

                                                                                                                                                                                                                                                                                             %��Ϊÿ��ͼ��Ķ�ά�붼�����Ϊ��ģ����Ա��뱣֤t���һάΪ��H(3,3)ҪΪ��
    if H(3,3)<0
    H = -H;



            tempH = [H(:,1),H(:,2),cross(H(:,1),H(:,2))];

    JacobiSVD<Eigen::MatrixXf> svd(tempH, ComputeThinU | ComputeThinV );
    Matrix3f V = svd.matrixV(), U = svd.matrixU();
    Matrix3f S << 1,0,0,0,1,0,0,0,U*V.inverse();;

    [U,~,V] = svd(tempH);


    R = U*S*V';
    t = H(:,3)/norm(tempH(:,1));

    proj_points_homo = K*(R*render_points'+t);

    proj_points = [proj_points_homo(1,:)./proj_points_homo(3,:);proj_points_homo(2,:)./proj_points_homo(3,:)]';
}