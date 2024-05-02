/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

// #include <iostream>
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include "Converter.h"
#include "PointCloude.h"
#include "System.h"
#include <unistd.h>
#include "pointcloudmapping.h"


int currentloopcount = 0;

// 点云地图初始化
PointCloudMapping::PointCloudMapping(double resolution_,double meank_,double thresh_)
{
    this->resolution = resolution_;
    this->meank = thresh_;
    this->thresh = thresh_;
    statistical_filter.setMeanK(meank);
    statistical_filter.setStddevMulThresh(thresh);
    voxel.setLeafSize( resolution, resolution, resolution);
    globalMap = boost::make_shared< PointCloud >( );
    loopbusy=false;
    viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}

// 插入关键帧的函数，在这里进行修改
// 本身其功能包括：
// 1.向记录所有关键帧地址的keyframes中添加该关键帧地址
// 2.更新当前全部关键帧currentvpkfs(但貌似没有用到)
// 3.生成点云(利用generate函数生成作为返回值)，并添加点云到全局点云中
// 4.最后给关键帧更新提示(实际上是点云更新提示)，发送给本文件中的Viewer函数用于更新点云
// 解析后进行修改：
// 1.本身插入关键帧函数只负责插入关键帧，而不更新点云
// 2.生成点云改为接收深度图线程负责，并向Viewer发送提醒
// 3.currentvpkfs是在闭环检测线程中，全局ba时使用的，不需要动其加入关键帧地址
void PointCloudMapping::insertKeyFrame(KeyFrame* kf,int idk,vector<KeyFrame*> vpKFs)
{
    // cout<<"receive a keyframe, id = "<<idk<<" 第"<<kf->mnId<<"个"<<endl;
    //cout<<"vpKFs数量"<<vpKFs.size()<<endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );
    currentvpKFs = vpKFs;
    //colorImgs.push_back( color.clone() );
    //depthImgs.push_back( depth.clone() );
    // PointCloude pointcloude;
    // pointcloude.pcID = idk;
    // pointcloude.T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    // pointcloude.pcE = generatePointCloud(kf,color,depth);
    // pointcloud.push_back(pointcloude);
    // keyFrameUpdated.notify_one();
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)//,Eigen::Isometry3d T
{
    PointCloud::Ptr tmp( new PointCloud() );
    // point cloud is null ptr
    for ( int m=0; m<depth.rows; m+=3 )
    {
        for ( int n=0; n<depth.cols; n+=3 )
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d>5)
                continue;
            PointT p;
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;
            
            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];
                
            tmp->points.push_back(p);
        }
    }
    
    //Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    //PointCloud::Ptr cloud(new PointCloud);
    //pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    //cloud->is_dense = false;
    
    //cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return tmp;
}


void PointCloudMapping::viewer()
{
    pcl::visualization::CloudViewer viewer("viewer");
    while(1)
    {
        
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            // 这里接受更新全局点云信号
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }
        cout << "reveive signal and begin to process" << endl;
        // keyframe is updated 
        uint16_t TN = mNumDepthFs;
        size_t N = 0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }
        cout << TN << "  " << N << endl;
        if(loopbusy || bStop)
        {
          //cout<<"loopbusy || bStop"<<endl;
            cout << "loopbusy" << loopbusy << endl;
            cout << "bStop" << bStop << endl;            
            continue;
        }
        // 这里判断是否全部更新完了点云
        // 但是没用呀，这一行....
        // 改成了如果没有新的关键帧就continue
        cout << "lastKeyframeSize: " <<lastKeyframeSize << endl;
        if(lastKeyframeSize == TN)
        { 
            // 延时0.05s
            usleep(50000);
            continue;
        }

        // 这里设置一个局部变量表示是否需要后面更新点云
        // bool bNeedTrans = false;
        cloudbusy = true;
        for ( size_t i = lastKeyframeSize; i < TN ; i++ )
        {
          
            // 这里修改关键是i的取值，他的目的是取出对应的点云idx
            // 因此增加真实重建idx 
            PointCloud::Ptr p (new PointCloud);
            pcl::transformPointCloud( *(pointcloud[i].pcE), *p, pointcloud[i].T.inverse().matrix());
            *globalMap += *p;
 
        }
      
        // depth filter and statistical removal 
        PointCloud::Ptr tmp1 ( new PointCloud );
        
        statistical_filter.setInputCloud(globalMap);
        statistical_filter.filter( *tmp1 );

        PointCloud::Ptr tmp(new PointCloud());
        voxel.setInputCloud( tmp1 );
        voxel.filter( *globalMap );
        //globalMap->swap( *tmp );
        viewer.showCloud( globalMap );
        std::cout<<"show global map, size="<<TN<<"   "<<globalMap->points.size()<<endl;
        lastKeyframeSize = TN;
        cloudbusy = false;
        //*globalMap = *tmp1;
        
        //if()
        //{
	    
	//}
    }
}
void PointCloudMapping::save()
{
	pcl::io::savePCDFile( "result.pcd", *globalMap );
	std::cout <<"globalMap save finished" << endl;
}

// 全局BA后更新全局点云函数
// 这里增加需要检查关键帧是否为ifend，因为只有这些关键帧有对应的点云，，，吗？
// 并不需要，因为很幸运，他增加了遍历判断只有帧id和点云id相同才会进行校正，因此这个函数不需要修改
void PointCloudMapping::updatecloud()
{
	if(!cloudbusy)
	{
		loopbusy = true;
		cout<<"start loop mappoint"<<endl;
        PointCloud::Ptr tmp1(new PointCloud);
		for (int i=0;i<currentvpKFs.size();i++)
		{
		    for (int j=0;j<pointcloud.size();j++)
		    {   
				if(pointcloud[j].pcID==currentvpKFs[i]->mnFrameId) 
				{   
					Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat(currentvpKFs[i]->GetPose() );
					PointCloud::Ptr cloud(new PointCloud);
					pcl::transformPointCloud( *pointcloud[j].pcE, *cloud, T.inverse().matrix());
					*tmp1 +=*cloud;

					//cout<<"第pointcloud"<<j<<"与第vpKFs"<<i<<"匹配"<<endl;
					continue;
				}
			}
		}
        cout<<"finishloopmap"<<endl;
        PointCloud::Ptr tmp2(new PointCloud());
        voxel.setInputCloud( tmp1 );
        voxel.filter( *tmp2 );
        globalMap->swap( *tmp2 );
        //viewer.showCloud( globalMap );
        loopbusy = false;
        //cloudbusy = true;
        loopcount++;

        //*globalMap = *tmp1;
	}
}
