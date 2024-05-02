/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<string.h>
#include<stdio.h>
#include<stdlib.h>
#include<sys/socket.h>
#include<string>
#include<map>
#include<arpa/inet.h>
// #include<errno.h>
// #include<sys/types.h>
// #include<netinet/in.h>
// #include<unistd.h>
#include "pointcloudmapping.h"
#include <cstdio>
#include<opencv2/core/core.hpp>
#include<KeyFrame.h>
#include<System.h>

using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);
int get_client_connnect(int host);

int main(int argc, char **argv)
{
    // 通信建立后才能进行程序(debug期间注释掉)
    // 端口1
    my_server1 = get_client_connnect(5001);
    if (my_server1)
        std::cout << "port1 connected! server1:" << my_server1  << std::endl;
    else
    {
        std::cout << "port1 failed" << std::endl;
        return 2;
    }
    usleep(1000000);
    // 这里记得改全局
    my_server2 = get_client_connnect(5002);
    if (my_server2)
        std::cout << "port2 connected! server2:" << my_server2 << std::endl;
    else
    {
        std::cout << "port2 failed" << std::endl;
        return 2;
    }

    // while(1);
    
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3])+"/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;
    std::string imagePath;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        imagePath = string(argv[3]) + "/" + vstrImageFilenames[ni];
        im = cv::imread(imagePath,CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im,tframe,imagePath);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }
    // 等待点云和闭环都加载完再结束
    while(SLAM.mpPointCloudMapping->loopbusy || SLAM.mpPointCloudMapping->cloudbusy)
    {
        cout<<"";
    }
    // Tracking time statistics
    SLAM.mpPointCloudMapping->bStop = true;

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;
    cout << "-------" << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("CameraTrajectory2.txt");
    cout << "Exit successfully" << endl;

    SLAM.save();
    // Stop all threads
    SLAM.Shutdown();
    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}

int get_client_connnect(int host)
{
    int my_server;
    struct sockaddr_in  servaddr;
    if( (my_server = socket(AF_INET, SOCK_STREAM, 0)) < 0){
        printf("create socket error: %s(errno: %d)\n", strerror(errno),errno);
        return 0;
    }
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(host);
    if( inet_pton(AF_INET, "127.0.0.1", &servaddr.sin_addr) <= 0){
        printf("inet_pton error for 127.0.0.1:%d\n",host);
        return 0;
    }

    if( connect(my_server, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0){
        printf("connect error: %s(errno: %d)\n",strerror(errno),errno);
        return 0;
    }
    return my_server;
}
