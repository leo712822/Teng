
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow,QDialog
from PyQt5.QtWidgets import QMessageBox
from qwidget import paramui
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from qwidget import msgui,paramui
import re
import numpy as np
from ctypes import *
import ctypes
from matplotlib import pyplot as plt
import xlrd
import xlwt
from xlutils import copy
from time import sleep, ctime 
import threading
import time
import cv2
#import core.utils as utils
#import tensorflow as tf
from PIL import Image
import os
import random
import csv
from datetime import datetime
import serial                    #串口依赖
import binascii                  #串口数据解析依赖
# 云端数据依赖
import socket
import hashlib
import configparser
import websocket
import json
import pickle
#动态路径规划
from path_planning_lib import *
#三维目标识别与图像识别by龚章鹏
use_3D_lidar = False
use_2D_image = False
Lock_rader = threading.Lock()


if use_3D_lidar:
    from predict_module import lidar_object_recognition,box2Dcorner_from_ret_box3d_score   
    RECV_BUF_SIZE = 16*1248
    msg_stop=b'\xA5\x4F\x00\x40\x0F\x00\x00\x00'
    msg_start_5Hz=b'\xA5\x56\x00\x40\x0F\x00\x00\x07'
    msg_start_10Hz=b'\xA5\xA6\x00\x40\x0F\x00\x00\x57'
    msg_start_20Hz=b'\xA5\x46\x00\x40\x0F\x00\x00\xF7'
    s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)#socket.SOCK_DGRAM制定是UDP
    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    s.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_RCVBUF,
        RECV_BUF_SIZE
    )
    s.bind(('192.168.0.1',2014))
    w=0.0018     #5Hz,0.0018;10HZ,0.0036
    delt_t_table=[0,     13.32, 3.33,  16.65,
                  6.66,  19.98, 9.99,  23.31,
                  26.64, 39.96, 29.97, 43.29,
                  33.3,  46.62, 36.63, 49.95,
                  53.28, 66.6,  56.61, 69.93,
                  59.94, 73.26, 63.27, 76.59,
                  79.92, 93.24, 83.25, 96.57,
                  86.58, 99.9,  89.91, 103.23]
    H_beta_table=[6.01, 3.377, 6.01, 3.377,
                  6.01, 3.377, 6.01, 3.377,
                  6.01, 3.377, 6.01, 3.377,
                  6.01, 3.377, 6.01, 3.377,
                  6.01, 3.377, 6.01, 3.377,
                  6.01, 3.377, 6.01, 3.377,
                   6.01, 3.377, 6.01, 3.377,
                  6.01, 3.377, 6.01, 3.377]
    V_theta_table=[-15, -13, -11, -9,
                  -7,  -5,  -3,  -1,
                   1,  3,    5,  7,
                   9,  11,  13,  15,
                  -15, -13, -11, -9,
                  -7,  -5,  -3,  -1,
                   1,  3,    5,  7,
                   9,  11,  13,  15]

    def get_start():#启动
        s.sendto(msg_start_10Hz,('192.168.0.3',2015)) 
        time.sleep(5)

    def get_stop():
        s.sendto(msg_stop,('192.168.0.3',2015)) #停止
        time.sleep(5)
    
    def getpoints():
        frame=np.zeros([129024,4])
        num_index = 0
        for k in range(168):
            data_c,addr = s.recvfrom(1248)
            for i in range(12):
                A_angle = (data_c[i*100+2]+data_c[i*100+3]*256)*0.01
                for j in range(32):
                    Range = (data_c[i*100+4+j*3]+data_c[i*100+4+j*3+1]*256)*4/1000
                    if Range==0:
                        continue
                    Intensity = data_c[i*100+4+j*3+2]
                    Angle = (-A_angle - w*delt_t_table[j]-H_beta_table[j])*np.pi/180
                    X= Range*math.cos(V_theta_table[j]*np.pi/180)*math.cos(Angle)
                    Y= Range*math.cos(V_theta_table[j]*np.pi/180)*math.sin(Angle)
                    Z= Range*math.sin(V_theta_table[j]*np.pi/180)
                    frame[num_index,:]=X,Y,Z,Intensity
                    num_index += 1
        tmp = frame[0:num_index,:]
        tmp[:,3]=tmp[:,3]/255
        tmp[:,0]=-tmp[:,0]
        tmp[:,1]=-tmp[:,1]
        return  tmp
        
    get_start()
    
elif use_2D_image:
    from image_object_recognition import camera_recg
    from core.config import cfg
    cap = cv2.VideoCapture(0)



#########################动态轨迹规划参数设置by龚章鹏
area = 10 #20
simulink_flag = False
show_animation = False
error_sum = 0
GPS_t_yaw = None
x_set = []
y_set = []
yaw_set = []
time_dur_set = []
time_record_set = []
delta_set = []
error_set = []
error_set2 = []
error_set3 = []
path_set = []
dc_ob_set = []
df_ob_set = []
dr_ob_set = []
ob_set = []
c_speed_set = []
s0 = 0
c_speed = 0/3.6
v_state = 0
c_a = 0
c_d = 0
c_d_d = 0
c_d_dd = 0
time_record = 0
target_ind_last = 0
target_ind = 0
error_c= 0
ob = np.array([]).reshape([-1,2])
ob_lidar = np.array([]).reshape([-1,2])
ob_tmp8 = []
ob_tmp7 = []
ob_tmp6 = []
ob_tmp5 = []
ob_tmp4 = []
ob_tmp3 = []
ob_tmp2 = []
ob_tmp1 = []
state = None
delta = 0
frenet_init_flag = False
path = None
path_last = None
#########################动态轨迹规划参数设置by龚章鹏
#~~~~~~~~~可设置数据~~~~~~~~~~~
ex_AEB_w=30 #AEB识别区域宽度 单位：cm ID=1102 下位机对应参数:float break_w=30;
ex_AEB_h=50 #AEB识别区域长度 单位：cm ID=1103 下位机对应参数:float break_l=50;
ex_v=0#0 #设定车速 单位km/h ID=1104 
ex_ob=150 #主动避障避让程度系数 ID=1105 下位机对应参数:int ang_k=150;
ex_camer_d=900 #摄像头识别信号灯判停像素面积阈值 
ex_AEB_f=100 #AEB判定执行周期 单位：ms ID=1106 
ex_flag=5 #所有参数输入确定后置1,停止置2
ex_turn_mid=1024 #方向盘中位位置 ID=1107 下位机对应参数:main函数中int ang_k=150;
ex_turn_cp=30 #方向盘空行程补偿 ID=1108 下位机对应参数:main函数中int ang_k=150;
ex_leaveout_flag=0;
ex_record=0
ex_dataInput_start_flag=0
ex_dataInput_enpty_flag=0
ex_dataInput_start_flag_tmp=0
ex_GPS_direction_flag=1
tc_turn_mid=1024  # 方向盘中位位置
tc_AEB_h=50  # AEB识别区域长度 单位：dm
tc_AEB_w=30  # AEB识别区域宽度 单位：dm
tc_AEB_f=100  # AEB判定执行周期 单位：ms
tc_v=0  # 设定车速 单位km/h
tc_turn_cp=60  # 方向盘空行程补偿
tc_ob=150  # 主动避障避让程度系数
tc_camer_d=900  # 摄像头识别信号灯判停像素面积阈值
tc_stop_pwm=0  # 刹车pwm
hf_turn_mid=2  # 方向盘中位位置
hf_AEB_h=2  # AEB识别区域长度 单位：dm
hf_AEB_w=2  # AEB识别区域宽度 单位：dm
hf_AEB_f=2  # AEB判定执行周期 单位：ms
hf_v=2  # 设定车速 单位km/h
hf_turn_cp=2  # 方向盘空行程补偿
hf_ob=2  # 主动避障避让程度系数
hf_camer_d=2  # 摄像头识别信号灯判停像素面积阈值
hf_stop_pwm=2  # 刹车pwm
hf_AEB =2
hf_ac=2
hf_brake=2
hf_ob_start=2
fault_state=''
#~~~~~~~~~云端控制~~~~~~~~~~~
ex_start_direction =0 #1：倒车 其他：前进
ex_start_state = 2#2:不控，0:停止，1:启动
ex_traffic_signal = 1
pre_GPS_xc=0
pre_GPS_yc=0
#~~~~~~~~~交互数据~~~~~~~~~~~
ex_car_number = 'WIN19071607'
cloud_time_num=0
vehicle_ID=ex_car_number
longitude_int=0
longitude_frac=0
longitude_flag=1
longitude=0
latitude_int=0
latitude_frac=0
latitude_flag=1
latitude=0
vehicle_xita=0
soc=63
error='11000'
EPS_angle = 0
state_direction=0
ac_pwm = 0
brake_pwm = 0
#~~~~~~~~~执行指令~~~~~~~~~~~
brake=0
speed=0
brake_flag = False
GPS_xita=0            #方向转角
#~~~~~~~~错误诊断标识~~~~~~~~~
error_EPS =1
error_EPS_miss = 1
error_brake =1
error_ac = 1
error_haomibo = 1
error_camera =1
error_gps_move = 1

error_AEB = 1
error_capture = 1
error_GPS_xy = 1
error_GPS_ll = 1
error_GPS_yaw = 1
error_speed_miss = 1
error_ser = 1
error_can_send = 1
error_traffic =1 
error_battery = 1
#~~~~~~~~~标识位~~~~~~~~~~~~~~
ac_flag = 0           #AEB作用标识，非设置。置1，表示需制动
brake_signal = 0      #红绿灯判断刹车标识
cam_flag = 0          #红绿灯检测标识，非设置。置1，表示需制动
GPS_move_flag=0       #GPS寻迹功能开启标识，设置。置1开启
#~~~~~~~~全局数据变量存储~~~~~~~~
GPS_send_flag = 1
GPS_yaw=0             #车身角度
rwheel_vel_100=0      #轮速
GPS_x=np.zeros([100]) #寻迹坐标位置X存储
GPS_y=np.zeros([100]) #寻迹坐标位置y存储
GPS_s=np.zeros([100])
GPS_yawarray = np.zeros([100])
GPS_x_tmp=np.zeros([100]) #寻迹坐标位置X存储
GPS_y_tmp=np.zeros([100]) #寻迹坐标位置y存储
GPS_xc_int=0          #车身位置x整型部分
GPS_xc_frac=0         #车身位置x小数部分
GPS_xc=0             #车身位置x
GPS_xc_last=0
GPS_yc_int=0          #车身位置y整型部分
GPS_yc_frac=0         #车身位置y小数部分
GPS_yc=0              #车身位置y
GPS_yc_last=0
radar_range=np.zeros([64])
radar_speed=np.zeros([64])
radar_angle=np.zeros([64])
radar_power=np.zeros([64])
obstacle_range =np.zeros([120])
obstacle_speed =np.zeros([120])
break_d = np.ones([120])*500        #当前时刻采取紧急制动，最终与目标障碍物距离，已转换成d*sinθ
break_d_inten =np.ones([12])*500
obstacle_list=np.zeros((256,4))
obstacle_listh=np.zeros((256,4))
recv_radar = np.zeros((256,8))
gps_vel_kmh = 0
label=0 #红绿灯通讯检测信号
data_2=np.zeros([60])#红绿灯通讯串口数据
#~~~~~~~~~~其他全局变量~~~~~~~~~~
GPS_k=0               #寻迹轨迹序列索引
cam_area=0
#~~~~~~~~~全局常量~~~~~~~~~~~~~~~
ex_traffic_p_x = 507158.70 #红绿灯x
ex_traffic_p_y =14325935.83  #红绿灯y
ex_traffic_p_yaw = 262.10
d_to_traffic_light_thresh=10
area_thresh=ex_camer_d       #红绿灯识别面积阈值
t1=1                  #AEB反应时间
GPS_d=2.5
GPS_L=1.3
ac = 50               #制动加速度，单位分米/s
d_AEB = 30            #制动距离安全值阈值，单为分米
d_tmp_last=10000      #寻迹判断位置序列点，足够大就行
#~~~~~~~~~canID~~~~~~~~~~~~~~~~~~
EPS_ID=1129
send_brakeID=1127
send_accelerateID=1128
send_turnID=1130
soc_ID=1131
soc_error_ID=1132
gps_vel_ID=1164  # by Tre
gps_x_int_ID=1168
gps_float_ID=1169
gps_y_int_ID=1170
gps_yaw_ID=1171
gps_j_w_int_ID=1172
gps_w_frac_ID=1173
gps_j_frac_ID=1174
time_ID=1175
speedID=1176
state_diagnose_ID=1177
radarID_min=1280
radarID_max=1343
radar_trackID = 1539
radar_rawID = 1283
error_can_init = 1

def get_lidar_data():
    #旧版固态激光雷达,需搭配demo2.dll使用
    valid_num = dll_gzp.get_num_interface()
    frame_data =  np.array(np.fromiter(pointer_gzp, dtype=np.float32, count=valid_num*4))
    frame_data_reshape=frame_data.reshape(valid_num,4)   
    return frame_data_reshape

def points2pcd(points,name):
    #点云保存成pcd格式到本地
    PCD_FILE_PATH=name #存放路径  
    handle = open(PCD_FILE_PATH, 'a')   #写文件句柄
    point_num=points.shape[0]    #得到点云点数
    #pcd头部（重要）
    handle.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')
    # 依次写入点
    for i in range(point_num):
        string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2])
        handle.write(string)
    handle.close()

# 载入npy版地图数据，若没有该文件则为上面初始化的值
#~~~~~~~~~~~图像识别参数设置~~~~~~~~~~~~~

#~~~~~~~~~~~参数表加载~~~~~~~~~~~~~
break_d_thresh=np.zeros([12])
for i in range(6):
    a=ex_AEB_w*0.5/max(np.cos(np.pi*(5+i)/18),1e-10)
    b=ex_AEB_h/max(np.sin(np.pi*(5+i)/18),1e-10)
    break_d_thresh[6+i]=min(a,b)
    break_d_thresh[5-i]=min(a,b)
#串口初始化及启动
#ser = serial.Serial("COM5",57600,timeout=30)
#~~~~~~~~~~~界面定义~~~~~~~~~~~~~
myGlobalStart = 2 #停止是2 启动是1
myDirection = 0 #0是正向，1是反向
myFrontAndMid = 0 #0是从原点，1是从中间 输出1 ,2,3,4
lastUiLoacl = 0 #初始在第0页,2是参数录入的第二页
lastUiLoacl2 = 3 #初始在第3页,4是参数调试的第二页

#参数校验是否通过
def checkParam(param, reg, left, right, msg):
    check = param
    isCheck = 1
    if re.match(reg, check):
        if left == -1 and right == -1:
            pass
        else:
            if int(check) >= left and int(check) <= right:
                pass
            else:
                isCheck = 0
                initMsgUi(msg)
    else:
        isCheck = 0
        initMsgUi(msg)
    return isCheck
class MyWindow(QMainWindow, paramui.Ui_Form):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        # 定时器
        self.timer = QTimer(self)  # 初始化一个定时器
        self.timer.timeout.connect(self.operate)  # 计时结束调用operate()方法
        self.timer.start(500)  # 设置计时间隔并启动

    # 定时器信号
    def operate(self):
        global fault_state
        self.label_guzhang.setText(str(fault_state))
        pass

    ###########页面切换#############
    # 切换为参数录入界面
    def changeParamInput(self):
        global lastUiLoacl
        self.StackedWidget.setCurrentIndex(lastUiLoacl)
        self.StackedWidget.setStyleSheet("background-image: url(:/img/img_canshu.png);")

    # 切换为坐标录入界面
    def changeLocalInput(self):
        if myGlobalStart == 2:
            source = self.sender().objectName()
            self.StackedWidget.setCurrentIndex(1)
            self.StackedWidget.setStyleSheet("background-image: url(:/img/img_zuobiao.png);")

    # 切换到调试页面
    def changeTiaoshiInput(self):
        global lastUiLoacl2
        self.StackedWidget.setCurrentIndex(lastUiLoacl2)
        self.StackedWidget.setStyleSheet("background-image: url(:/img/img_tiaoshi.png);")

    # 切换寻迹方向
    def changeDirection(self):
        global myDirection
        # 启动状态
        if myGlobalStart != 1:
            if myDirection == 0:
                myDirection = 1
                self.gpsDInput2.setStyleSheet(
                    "background: transparent;background-image: url(:/img/fangxiang2.png);border:0;font: 24px \"思源黑体 CN Regular\";color:#cedce3;padding-right: 90px;")
                self.fanxiangView.setStyleSheet(
                    "background: transparent;border:0;font: 24px \"思源黑体 CN Regular\";color:#5c6572;")
            else:
                myDirection = 0
                self.gpsDInput2.setStyleSheet(
                    "background: transparent;background-image: url(:/img/fangxiang1.png);border:0;font: 24px \"思源黑体 CN Regular\";color:#5c6572;padding-right: 90px;")
                self.fanxiangView.setStyleSheet(
                    "background: transparent;border:0;font: 24px \"思源黑体 CN Regular\";color:#cedce3;")
        pass

    ###########按钮切换#############
    # 切换从原点还是从中间点
    def changeFrontAndMid(self):
        global myFrontAndMid
        # 启动状态
        if myGlobalStart != 1:
            if myFrontAndMid == 0:
                myFrontAndMid = 1
                self.gpsDInput1.setStyleSheet(
                    "background: transparent;background-image: url(:/img/fangxiang2.png);border:0;font: 24px \"思源黑体 CN Regular\";color:#cedce3;padding-right: 90px;")
                self.zhongjianView.setStyleSheet(
                    "background: transparent;border:0;font: 24px \"思源黑体 CN Regular\";color:#5c6572;")
            else:
                myFrontAndMid = 0
                self.gpsDInput1.setStyleSheet(
                    "background: transparent;background-image: url(:/img/fangxiang1.png);border:0;font: 24px \"思源黑体 CN Regular\";color:#5c6572;padding-right: 90px;")
                self.zhongjianView.setStyleSheet(
                    "background: transparent;border:0;font: 24px \"思源黑体 CN Regular\";color:#cedce3;")
        pass

    ###########页码切换#############
    # 上一页 参数设置
    def changePageUp(self):
        if self.StackedWidget.currentIndex() == 2:
            self.StackedWidget.setCurrentIndex(0)

    # 下一页 参数设置
    def changePageDown(self):
        if self.StackedWidget.currentIndex() == 0:
            self.StackedWidget.setCurrentIndex(2)

    # 上一页 调试参数
    def changePageUpCanshu(self):
        if self.StackedWidget.currentIndex() == 4:
            self.StackedWidget.setCurrentIndex(3)
        pass

    # 下一页 调试参数
    def changePageDownCanshu(self):
        if self.StackedWidget.currentIndex() == 3:
            self.StackedWidget.setCurrentIndex(4)
        pass

    ###########程序#############
    # 执行按钮被按下
    def zhixingPutton(self):
        name = self.sender().objectName()

        global tc_turn_mid  # 方向盘中位位置
        global tc_AEB_h  # AEB识别区域长度 单位：dm
        global tc_AEB_w  # AEB识别区域宽度 单位：dm
        global tc_AEB_f  # AEB判定执行周期 单位：ms
        global tc_v  # 设定车速 单位km/h
        global tc_turn_cp  # 方向盘空行程补偿
        global tc_ob  # 主动避障避让程度系数
        global tc_camer_d  # 摄像头识别信号灯判停像素面积阈值

        global hf_turn_mid  # 方向盘中位位置
        global hf_AEB_h  # AEB识别区域长度 单位：dm
        global hf_AEB_w  # AEB识别区域宽度 单位：dm
        global hf_AEB_f  # AEB判定执行周期 单位：ms
        global hf_v  # 设定车速 单位km/h
        global hf_turn_cp  # 方向盘空行程补偿
        global hf_ob  # 主动避障避让程度系数
        global hf_camer_d  # 摄像头识别信号灯判停像素面积阈值
        global hf_brake  # 刹车pwm
        global hf_ac  # 油门
        global hf_AEB  # aeb
        global hf_ob_start  # 动态避障

        if name == "pushButton_fxzx":
            if re.match(r'^[1-9]\d*$', self.turnMidInpur_fx.displayText()):
                pass
            else:
                initMsgUi("参数为数字，请重新录入！")
                return
            tc_turn_mid = self.turnMidInpur_fx.displayText()
            hf_turn_mid = 1
        if name == "pushButton_cdzx":
            if re.match(r'^[1-9]\d*$', self.turnMidInpur_cd.displayText()):
                pass
            else:
                initMsgUi("参数为数字，请重新录入！")
                return
            tc_AEB_h = self.turnMidInpur_cd.displayText()
            hf_AEB_h = 1
        if name == "pushButton_kdzx":
            if re.match(r'^[1-9]\d*$', self.turnMidInpur_kd.displayText()):
                pass
            else:
                initMsgUi("参数为数字，请重新录入！")
                return
            tc_AEB_w = self.turnMidInpur_kd.displayText()
            hf_AEB_w = 1
        if name == "pushButton_zqzx":
            if re.match(r'^[1-9]\d*$', self.turnMidInpur_zq.displayText()):
                pass
            else:
                initMsgUi("参数为数字，请重新录入！")
                return
            tc_AEB_f = self.turnMidInpur_zq.displayText()
            hf_AEB_f = 1
        if name == "pushButton_cszx":
            if re.match(r'^[1-9]\d*$', self.turnMidInpur_cs.displayText()):
                pass
            else:
                initMsgUi("参数为数字，请重新录入！")
                return
            tc_v = self.turnMidInpur_cs.displayText()
            hf_v = 1
        if name == "pushButton_bczx":
            if re.match(r'^[1-9]\d*$', self.turnMidInpur_bc.displayText()):
                pass
            else:
                initMsgUi("参数为数字，请重新录入！")
                return
            tc_turn_cp = self.turnMidInpur_bc.displayText()
            hf_turn_cp = 1
        if name == "pushButton_xszx":
            if re.match(r'^[1-9]\d*$', self.turnMidInpur_xs.displayText()):
                pass
            else:
                initMsgUi("参数为数字，请重新录入！")
                return
            tc_ob = self.turnMidInpur_xs.displayText()
            hf_ob = 1
        if name == "pushButton_yzzx":
            if re.match(r'^[1-9]\d*$', self.turnMidInpur_yz.displayText()):
                pass
            else:
                initMsgUi("参数为数字，请重新录入！")
                return
            tc_camer_d = self.turnMidInpur_yz.displayText()
            hf_camer_d = 1
        if name == "pushButton_sczx":
            hf_brake = 1
        if name == "pushButton_ymzx":
            hf_ac = 1
        if name == "pushButton_aebzx":
            hf_AEB = 1
        if name == "pushButton_dtzx":
            hf_ob_start = 1
        initMsgUi("执行成功！")

    # 恢复按钮被按下
    def huifuPutton(self):
        name = self.sender().objectName()

        global hf_turn_mid  # 方向盘中位位置
        global hf_AEB_h  # AEB识别区域长度 单位：dm
        global hf_AEB_w  # AEB识别区域宽度 单位：dm
        global hf_AEB_f  # AEB判定执行周期 单位：ms
        global hf_v  # 设定车速 单位km/h
        global hf_turn_cp  # 方向盘空行程补偿
        global hf_ob  # 主动避障避让程度系数
        global hf_camer_d  # 摄像头识别信号灯判停像素面积阈值
        global hf_GPS_direction_flag  # 寻迹方向
        global hf_brake  # 刹车pwm
        global hf_ac  # 油门
        global hf_AEB  # aeb
        global hf_ob_start  # 动态避障

        if name == "pushButton_fxhf":
            hf_turn_mid = 0
            initMsgUi("恢复成功！")
        if name == "pushButton_schf":
            hf_brake = 0
            initMsgUi("恢复成功！")
        if name == "pushButton_ymhf":
            hf_ac = 0
            initMsgUi("恢复成功！")
        if name == "pushButton_aebhf":
            hf_AEB = 0
            initMsgUi("停止成功！")
        if name == "pushButton_dthf":
            hf_ob_start = 0
            initMsgUi("停止成功！")

    # 程序全部退出
    def closeEvent(self, *args, **kwargs):
        global ex_leaveout_flag  # 程序终止
        ex_leaveout_flag = 1

    # 关闭程序
    def leaveOut(self):
        self.close()
        global ex_leaveout_flag  # 程序终止
        ex_leaveout_flag = 1
        pass

    # 最小化程序
    def showMinWin(self):
        self.showMinimized()

    # 坐标录入启动
    def localStart(self):
        global ex_dataInput_start_flag  # 开始采集数据
        ex_dataInput_start_flag = 1
        initMsgUi("开始成功！")
        tmp = self.sender()
        print(tmp.objectName())

    def localOver(self):
        global ex_dataInput_start_flag  # 结束采集数据
        ex_dataInput_start_flag = 0
        # QMessageBox.information(self, "提示", "结束成功", QMessageBox.Yes)
        initMsgUi("停止成功！")

    def localEmpty(self):
        global ex_dataInput_enpty_flag  # 清空采集的数据
        ex_dataInput_enpty_flag = 1
        initMsgUi("清空成功！")

    # 参数录入启动
    def startParam(self):
        # 自己的全局变量
        global myGlobalStart
        global myDirection
        # 全局变量声明
        global ex_turn_mid  # 方向盘中位位置
        global ex_AEB_h  # AEB识别区域长度 单位：dm
        global ex_AEB_w  # AEB识别区域宽度 单位：dm
        global ex_AEB_f  # AEB判定执行周期 单位：ms
        global ex_v  # 设定车速 单位km/h
        global ex_turn_cp  # 方向盘空行程补偿
        # global ex_ob  # 主动避障避让程度系数
        global ex_camer_d  # 摄像头识别信号灯判停像素面积阈值
        global ex_GPS_direction_flag  # 寻迹方向
        global ex_traffic_p_x  # 红绿灯x
        global ex_traffic_p_y  # 红绿灯y
        global ex_traffic_p_yaw  # 红绿灯角度

        global ex_flag  # 所有参数输入确定后置1  停止2

        # 点击启动
        if myGlobalStart == 2:
            # 校验
            isCheck = 1
            # 方向盘中位位置
            if isCheck == 1:
                isCheck = checkParam(self.turnMidInpur.displayText(), r'^[1-9]\d*$', 0, 2048,
                                     "方向盘中位位置为正整数（0~2048），请重新录入！")
            # AEB识别区域长度
            if isCheck == 1:
                isCheck = checkParam(self.aebHInput.displayText(), r'^[1-9]\d*$', 10, 100,
                                     "AEB识别区域长度为正整数（10~100），请重新录入！")
            # AEB识别区域宽度
            if isCheck == 1:
                isCheck = checkParam(self.aebWInput.displayText(), r'^[1-9]\d*$', 20, 50, "AEB识别区域宽度为正整数（20~50），请重新录入！")
            # AEB判定执行周期
            if isCheck == 1:
                isCheck = checkParam(self.aebFInput.displayText(), r'^[1-9]\d*$', 50, 200,
                                     "AEB判定执行周期为正整数（50~200），请重新录入！")
            # 设定车速
            if isCheck == 1:
                isCheck = checkParam(self.vInput.displayText(), r'^[1-9]\d*$', 1, 4, "车速为正整数（1~4），请重新录入！")
            # 方向盘空行程补偿
            if isCheck == 1:
                isCheck = checkParam(self.turnCpInput.displayText(), r'^[0-9]\d*$', 0, 30, "方向盘空行程补偿为正整数（0~30），请重新录入！")
            # 摄像头识别信号灯判停像素面积阈值
            if isCheck == 1:
                isCheck = checkParam(self.camerDInput.displayText(), r'^[1-9]\d*$', 500, 2000,
                                     "信号灯判停像素面积阈值为正整数（500~2000），请重新录入！")
            # 红绿灯x
            if isCheck == 1:
                isCheck = checkParam(self.honglvInput_x.displayText(), r'^[0-9]+(.[0-9]{1,10})?$', -1, -1,
                                     "红绿灯x为正数，请重新录入！")
            # 红绿灯y
            if isCheck == 1:
                isCheck = checkParam(self.honglvInput_y.displayText(), r'^[0-9]+(.[0-9]{1,10})?$', -1, -1,
                                     "红绿灯y为正数，请重新录入！")
            # 红绿灯角度
            if isCheck == 1:
                isCheck = checkParam(self.honglvInput_j.displayText(), r'^[0-9]+(.[0-9]{1,10})?$', -1, -1,
                                     "红绿灯角度为正数，请重新录入！")

            # 参数植入 启动程序
            if isCheck == 1:
                ex_turn_mid = self.turnMidInpur.displayText()
                ex_AEB_h = self.aebHInput.displayText()
                ex_AEB_w = self.aebWInput.displayText()
                ex_AEB_f = self.aebFInput.displayText()
                ex_turn_cp = self.turnCpInput.displayText()
                # ex_ob = inputParam[5]
                ex_camer_d = self.camerDInput.displayText()
                ex_v = self.vInput.displayText()
                ex_traffic_p_x = self.honglvInput_x.displayText()
                ex_traffic_p_y = self.honglvInput_y.displayText()
                ex_traffic_p_yaw = self.honglvInput_j.displayText()

                if myDirection == 0:
                    if myFrontAndMid == 0:
                        ex_GPS_direction_flag = 1
                    else:
                        ex_GPS_direction_flag = 2
                else:
                    if myFrontAndMid == 0:
                        ex_GPS_direction_flag = 3
                    else:
                        ex_GPS_direction_flag = 4

                # 输入框编程只读
                self.turnMidInpur.setReadOnly(bool(1))
                self.aebHInput.setReadOnly(bool(1))
                self.aebWInput.setReadOnly(bool(1))
                self.aebFInput.setReadOnly(bool(1))
                self.turnCpInput.setReadOnly(bool(1))
                # self.obInput.setReadOnly(bool(1))
                self.camerDInput.setReadOnly(bool(1))
                self.vInput.setReadOnly(bool(1))
                self.honglvInput_x.setReadOnly(bool(1))
                self.honglvInput_y.setReadOnly(bool(1))
                self.honglvInput_j.setReadOnly(bool(1))

                # 启动
                ex_flag = 1  # 启动
                myGlobalStart = 1
                self.qidong.setStyleSheet("background: transparent;background-image: url(:/img/img_ting.png);")
                self.qidong2.setStyleSheet("background: transparent;background-image: url(:/img/img_ting.png);")
                initMsgUi("启动成功！")

        else:  # 准备停止
            # 输入框取消只读
            self.turnMidInpur.setReadOnly(bool(0))
            self.aebHInput.setReadOnly(bool(0))
            self.aebWInput.setReadOnly(bool(0))
            self.aebFInput.setReadOnly(bool(0))
            self.turnCpInput.setReadOnly(bool(0))
            # self.obInput.setReadOnly(bool(0))
            self.camerDInput.setReadOnly(bool(0))
            self.vInput.setReadOnly(bool(0))
            self.honglvInput_x.setReadOnly(bool(0))
            self.honglvInput_y.setReadOnly(bool(0))
            self.honglvInput_j.setReadOnly(bool(0))

            # 启动
            ex_flag = 2  # 停止
            myGlobalStart = 2
            self.qidong.setStyleSheet("background: transparent;background-image: url(:/img/img_start.png);")
            self.qidong2.setStyleSheet("background: transparent;background-image: url(:/img/img_start.png);")
            initMsgUi("停止成功！")
###########窗口启动程序#############
#主窗口
def initUi():
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    #myWin.showFullScreen()
    #子窗口
    global myWinMsg
    myWinMsg = MyMessage()
    # 阻塞
    sys.exit(app.exec_())

#信息弹出框
class MyMessage(QDialog, msgui.Ui_Dialog):
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        # self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        # self.setWindowFlags(QtCore.Qt.CustomizeWindowHint)
    #关闭按钮
    def leaveOut(self):
        self.close()

def initMsgUi(msg):
    #app = QApplication(sys.argv)
    global myWinMsg
    myWinMsg.show()
    myWinMsg.activateWindow()
    myWinMsg.messageLaber.setText(msg)
    #sys.exit(app.exec_())

#~~~~~~~~~~~云端定义~~~~~~~~~~~~~
#需要更改：服务器ip
ADDR = '39.97.48.113'
#需要更改：服务器ip端口
PORT = 80
#需要更改：请求行
URL_HEADER = 'http://39.97.48.113:80/demo/setdata'
URL_HEADER_ONE = 'http://39.97.48.113:80/demo/setdata'
URL_HEADER_TWO = 'http://39.97.48.113:80/demo/setdata'
#数据下载接口
URL_HEADER_WS = 'http://39.97.48.113:80/demo/setdata'

# 封装数据的类
class MyClassOne:
    # 初始化
    def __init__(self,CloudDaTa0,CloudDaTa1,CloudDaTa2,CloudDaTa3,CloudDaTa4,CloudDaTa5,CloudDaTa6):
        self.CloudDaTa0 = CloudDaTa0
        self.CloudDaTa1 = CloudDaTa1
        self.CloudDaTa2 = CloudDaTa2
        self.CloudDaTa3 = CloudDaTa3
        self.CloudDaTa4 = CloudDaTa4
        self.CloudDaTa5 = CloudDaTa5
        self.CloudDaTa6 = CloudDaTa6

class MyClassTwo:
    # 初始化
    def __init__(self,CloudDaTa0,CloudDaTa1,CloudDaTa2,CloudDaTa3,CloudDaTa4,CloudDaTa5):
        self.CloudDaTa0 = CloudDaTa0
        self.CloudDaTa1 = CloudDaTa1
        self.CloudDaTa2 = CloudDaTa2
        self.CloudDaTa3 = CloudDaTa3
        self.CloudDaTa4 = CloudDaTa4
        self.CloudDaTa5 = CloudDaTa5


# 读取配置文件，后期可以优化
def initConfig():
    try:
        test_cfg = "./mycfg.ini"
        config_raw = configparser.RawConfigParser()
        config_raw.read(test_cfg)
        global ADDR
        global PORT
        global URL_HEADER
        global URL_HEADER_ONE
        global URL_HEADER_TWO
        global URL_HEADER_WS

        ADDR = config_raw.get('DEFAULT', 'ADDR')
        PORT = config_raw.getint('DEFAULT', 'PORT')
        URL_HEADER = config_raw.get('DEFAULT', 'URL_HEADER')
        URL_HEADER_WS = config_raw.get('DEFAULT', 'URL_HEADER_WS')

        URL_HEADER_ONE = URL_HEADER + "One.do"
        URL_HEADER_TWO = URL_HEADER + "Two.do"
    except BaseException:
        print('配置文件读取失败')
#数据上传接口1 最快3s/次
# CloudDaTa0    车辆编号
# CloudDaTa1    时间
# CloudDaTa2    经度
# CloudDaTa3    纬度
# CloudDaTa4    横摆角
# CloudDaTa5    车速
# CloudDaTa6    车辆故障
def dataUploadByHttpOne(CloudDaTa0,CloudDaTa1,CloudDaTa2,CloudDaTa3,CloudDaTa4,CloudDaTa5,CloudDaTa6):
    # 使用全局变量
    global ADDR
    global PORT
    global URL_HEADER_ONE

    if CloudDaTa2 <= 1:
        CloudDaTa2 = 0.0
        CloudDaTa3 = 0.0
    if CloudDaTa3 <= 1:
        CloudDaTa2 = 0.0
        CloudDaTa3 = 0.0
    #初始化数据
    dataSource = MyClassOne(str(CloudDaTa0), str(CloudDaTa1), str(CloudDaTa2), str(CloudDaTa3), str(CloudDaTa4), str(CloudDaTa5),str(CloudDaTa6)).__dict__
    #构建向服务器发送的数据
    tmp = hashlib.md5()
    tmp.update(json.dumps(dataSource).encode("utf-8"))
    str0="data=" + json.dumps(dataSource) + "&&md5=" + tmp.hexdigest()
    #构建http头
    str1 ="POST " + URL_HEADER_ONE + " HTTP/1.1\n" + "Host: www.webxml.com.cn\n" + "Content-Type: application/x-www-form-urlencoded\n" + "Content-Length: "
    str1 = str1 + str(len(str0)) + "\n\n" + str0 + "\r\n\r\n"
    # 创建一个socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # 建立连接:
        s.connect((ADDR, PORT))
        #设置连接超时
        s.settimeout(3)
        # 发送数据:
        s.send(bytes(str1, encoding='utf8'))
        #设置接收
        a = s.recv(1000)
    except BaseException:
        print('链接1未建立')
    # 关闭连接:
    s.close()

#数据上传接口2  最快1s/次
# CloudDaTa0    车辆编号
# CloudDaTa1    车速
# CloudDaTa2    电量
# CloudDaTa3    红绿灯状态
# CloudDaTa4    运行方向
# CloudDaTa5    底盘状态
def dataUploadByHttpTwo(CloudDaTa0,CloudDaTa1,CloudDaTa2,CloudDaTa3,CloudDaTa4,CloudDaTa5):
    # 使用全局变量
    global ADDR
    global PORT
    global URL_HEADER_TWO
    #初始化数据
    dataSource = MyClassTwo(str(CloudDaTa0), str(CloudDaTa1), str(CloudDaTa2), str(CloudDaTa3), str(CloudDaTa4), str(CloudDaTa5)).__dict__
    #构建向服务器发送的数据
    tmp = hashlib.md5()
    tmp.update(json.dumps(dataSource).encode("utf-8"))
    str0="data=" + json.dumps(dataSource) + "&&md5=" + tmp.hexdigest()
    #构建http头
    str1 ="POST " + URL_HEADER_TWO + " HTTP/1.1\n" + "Host: www.webxml.com.cn\n" + "Content-Type: application/x-www-form-urlencoded\n" + "Content-Length: "
    str1 = str1 + str(len(str0)) + "\n\n" + str0 + "\r\n\r\n"
    # 创建一个socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # 建立连接:
        s.connect((ADDR, PORT))
        #设置连接超时
        s.settimeout(1)
        # 发送数据:
        s.send(bytes(str1, encoding='utf8'))
        #设置接收
        a = s.recv(1000)
    except BaseException:
        print('链接2未建立')
    # 关闭连接:
    s.close()

# 小车接收云端消息
def dataDownLoad(ws):
    try:
        ws.on_open = on_open
        ws.run_forever()
        # time.sleep(1)
        print("服务器连接失败")
    except BaseException as e:
        print(e)

def on_message(ws, message):
    print('message=')
    print(message)
    result = json.loads(message)
    #变量赋值
    global vehicle_ID            #用于唯一标识,与参数上传编号一致
    global ex_v                  # 设定车速km/h ，浮点数
    global ex_start_direction    # 0代表前进，1代表倒车
    global ex_start_state        # 1代表启动，0代表停止
    global ex_traffic_signal        # 1代表识别，0代表不识别

    ex_start_direction = result['direction']
    ex_v = result['speed']
    ex_start_state = result['start']
    ex_traffic_signal = result['trafficLight']
# 出现错误
def on_error(ws, error):
    print(error)
# 连接关闭
def on_close(ws):
    global URL_HEADER_WS
    global thred_flag
    if thred_flag:
        print("重新连接服务器...")
        ws = websocket.WebSocketApp(URL_HEADER_WS,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close)
def on_open(ws):
    global vehicle_ID  # 用于唯一标识,与参数上传编号一致
    global thred_flag
    if thred_flag:
        print("服务器连接成功")
        ws.send(vehicle_ID) # 需要修改

#~~~~~~~~~~~~~can收发配置定义by龚章鹏~~~~~~~~~~~~~~
VCI_USBCAN2A = 4
class VCI_INIT_CONFIG(Structure):  
    _fields_ = [("AccCode", c_ulong),
                ("AccMask", c_ulong),
                ("Reserved", c_ulong),
                ("Filter", c_ubyte),
                ("Timing0", c_ubyte),
                ("Timing1", c_ubyte),
                ("Mode", c_ubyte)
                ]

class VCI_CAN_OBJ(Structure):
    _fields_=[("ID", c_uint),
                ("TimeStamp", c_uint),
                ("TimeFlag", c_ubyte),
                ("SendType", c_ubyte),
                ("RemoteFlag", c_ubyte),
                ("ExternFlag", c_ubyte),
                ("DataLen", c_ubyte),
                ("Data", c_ubyte*8),
                ("Reserved", c_ubyte*3)]

def exportList(ret):
    pass
    structList = []
    for i in range(0,ret):
        j = str(i)
        a = ("ID"+j, c_uint)
        b = ("TimeStamp"+j, c_uint)
        d = ("TimeFlag"+j, c_ubyte)
        e = ("SendType"+j, c_ubyte)
        f = ("RemoteFlag"+j, c_ubyte)
        g = ("ExternFlag"+j, c_ubyte)
        h = ("DataLen"+j, c_ubyte)
        q = ("Data"+j, c_ubyte*8)
        w = ("Reserved"+j, c_ubyte*3)
        structList.append(a)
        structList.append(b)
        #structList.append()
        structList.append(d)
        structList.append(e)
        structList.append(f)
        structList.append(g)
        structList.append(h)
        structList.append(q)
        structList.append(w)
    return(structList) 

class VCI_CAN_OBJ2(Structure):
    res = exportList(2000)
    _fields_=res
#can初始化及启动
canLib = windll.LoadLibrary('./ControlCAN.dll') 
vic = VCI_INIT_CONFIG(0x80000008, 0xFFFFFFFF, 0,2, 0x00, 0x1C, 0)
can_sata_1=canLib.VCI_OpenDevice(4, 0, 0)
can_sata_2=canLib.VCI_InitCAN(4, 0, 0, pointer(vic))
can_sata_3=canLib.VCI_StartCAN(4, 0, 0)
can_sata_4=canLib.VCI_ClearBuffer(4, 0, 0)
print('打开设备: %d' % (can_sata_1)) #4代表CANalyst-II型号，0代表设备1，0代表通道1
print('初始化: %d' % (can_sata_2))
print('启动: %d' % (can_sata_3))
print('清空缓冲区: %d' % (can_sata_4))
v2=VCI_CAN_OBJ2()
if can_sata_1==1 and can_sata_2==1 and can_sata_3==1 and can_sata_4==1:
    error_can_init=0
else:
    error_can_init=1

def ascii_to_int(a):
    if a==0:
        b=0
    else:
        if a==46:
            b=10
        else:
            b=a-48
    return b

def data_to_can(a):
    a0=0
    a1=0
    a2=0
    a3=0
    a4=0
    a5=0
    a6=0
    a7=0
    a7=a%256
    a_tmp=(a-a7)/256
    a6=a_tmp%256
    a_tmp=(a_tmp-a6)/256
    a5=a_tmp%256
    a_tmp=(a_tmp-a5)/256
    a4=a_tmp%256
    a_tmp=(a_tmp-a4)/256
    a3=a_tmp%256
    a_tmp=(a_tmp-a3)/256
    a2=a_tmp%256
    a_tmp=(a_tmp-a2)/256
    a1=a_tmp%256
    a_tmp=(a_tmp-a1)/256
    a0=a_tmp%256
    return a0,a1,a2,a3,a4,a5,a6,a7

def can_send(id,a0,a1,a2,a3,a4,a5,a6,a7):
    ubyte_array = c_ubyte*8
    ubyte_3array = c_ubyte*3
    b_can = ubyte_3array(0, 0 , 0)
    a_can = ubyte_array(int(a0),int(a1),int(a2),int(a3),int(a4),int(a5),int(a6),int(a7))
    vco = VCI_CAN_OBJ(id, 0, 0, 0, 0, 0,  8, a_can, b_can)
    canLib.VCI_Transmit(4, 0, 0, pointer(vco), 1)

def capture():
    #can接收函数,一次接收2000条can报文
    global GPS_xc_int
    global GPS_xc_frac
    global GPS_yc_int
    global GPS_yc_frac
    global GPS_xc
    global GPS_yc
    global GPS_yaw
    global rwheel_vel_100
    global radarID_min
    global radarID_max
    global gps_x_int_ID
    global gps_float_ID
    global gps_y_int_ID
    global gps_vel_kmh
    global gps_yaw_ID
    global EPS_ID
    global state_diagnose_ID
    global canLib
    global v2
    global longitude_int
    global longitude_frac
    global longitude_flag
    global latitude_frac
    global latitude_int
    global latitude_flag
    global longitude
    global latitude
    global error_brake
    global error_ac 
    global error_capture
    global error_GPS_xy
    global error_GPS_ll
    global error_GPS_yaw
    global error_speed_miss
    global error_EPS
    global error_haomibo
    global error_EPS_miss
    global EPS_angle
    global state_direction
    global ac_pwm
    global brake_pwm
    global soc_ID
    global soc_error_ID
    global soc
    global error_battery
    # by Tre
    global recv_radar
    global radar_trackID
    global radar_rawID
    global gps_vel_ID
    global cell_size
    global front
    ret = canLib.VCI_Receive(VCI_USBCAN2A, 0, 0, byref(v2), 2000, 0)
    #print('ret=%d' % ret)
    if ret>0:
        for i in range(0,ret):
            loc = locals()
            exec('ID=v2.ID{}'.format(i))
            ID=loc['ID']
            exec('Data=v2.Data{}'.format(i))
            Data=loc['Data']
            # if ID >= radarID_min and ID <= radarID_max:
            #     haomibo(Data,ID)
            if ID == radar_trackID:
                # a = int(Data[0]) % 128
                a = int(Data[0])
                for k_radar in range(8):
                    recv_radar[a][k_radar] = Data[k_radar]
                #print('a=%d' % a)
            else:
                if ID==gps_x_int_ID:
                    GPS_xc_int=ascii_to_int(Data[5])+ascii_to_int(Data[4])*10+ascii_to_int(Data[3])*100+ascii_to_int(Data[2])*1000+ascii_to_int(Data[1])*10000+ascii_to_int(Data[0])*100000
                else:
                    if ID==gps_float_ID:
                        GPS_xc_frac=ascii_to_int(Data[0])*0.1+ascii_to_int(Data[1])*0.01
                        GPS_xc=GPS_xc_int+GPS_xc_frac
                        GPS_yc_frac=ascii_to_int(Data[2])*0.1+ascii_to_int(Data[3])*0.01
                        #print('GPS_xc = %.2f '%(GPS_xc))
                        error_GPS_xy+=1
                    else:
                        if ID==gps_y_int_ID:
                            GPS_yc_int=ascii_to_int(Data[7])*1+ascii_to_int(Data[6])*10+ascii_to_int(Data[5])*100+ascii_to_int(Data[4])*1000+ascii_to_int(Data[3])*10000+ascii_to_int(Data[2])*100000+ascii_to_int(Data[1])*1000000+ascii_to_int(Data[0])*10000000
                            GPS_yc=GPS_yc_int+GPS_yc_frac
                            #print('GPS_yc = %.2f '%(GPS_yc))
                            error_GPS_xy+=1
                        else:
                            if ID==gps_yaw_ID:
                                tmp=0
                                tmp_2=0
                                for k in range(8):
                                    if ascii_to_int(Data[k])==10:
                                        tmp_2=1
                                    else:
                                        if tmp_2>=1:
                                            tmp=tmp+ascii_to_int(Data[k])*0.1**int(tmp_2)
                                            tmp_2=tmp_2+1
                                        else:
                                            tmp=tmp*10+ascii_to_int(Data[k])
                                            
                                GPS_yaw=tmp
                                #print('GPS_yaw %.2f' % GPS_yaw)
                                error_GPS_yaw+=1
                            else:
                                if ID== speedID or ID==gps_vel_ID:
                                    if ID== speedID:
                                        rwheel_vel_100=Data[7]+Data[6]*16*16 #单位：厘米/s
                                        #c_speed = rwheel_vel_100
                                        #print('转速=%d'%rwheel_vel_100)
                                        error_speed_miss+=1
                                    else:
                                        gps_vel_kmh=(Data[7]+Data[6]*16*16)/10
                                        if gps_vel_kmh>6000:
                                            gps_vel_kmh=0
                                        #print('车速=%.1f'%gps_vel_kmh)
                                else:
                                    if ID==gps_j_w_int_ID:
                                        if Data[0]==1:
                                            latitude_flag=1
                                        else:
                                            latitude_flag=-1
                                        latitude_int=Data[1]
                                        if Data[2]==1:
                                            longitude_flag=1
                                        else:
                                            longitude_flag=-1
                                        longitude_int=Data[3]
                                    else:
                                        if ID==gps_j_frac_ID:
                                            longitude_frac=ascii_to_int(Data[0])*0.1+ascii_to_int(Data[1])*0.01+ascii_to_int(Data[2])*0.001+ascii_to_int(Data[3])*0.0001+ascii_to_int(Data[4])*0.00001+ascii_to_int(Data[5])*0.000001+ascii_to_int(Data[6])*0.0000001+ascii_to_int(Data[7])*0.00000001
                                            longitude=(longitude_int+longitude_frac)*longitude_flag
                                            #print('longitude = %.8f '%(longitude))
                                            error_GPS_ll+=1
                                        else:
                                            if ID==gps_w_frac_ID:
                                                latitude_frac=ascii_to_int(Data[0])*0.1+ascii_to_int(Data[1])*0.01+ascii_to_int(Data[2])*0.001+ascii_to_int(Data[3])*0.0001+ascii_to_int(Data[4])*0.00001+ascii_to_int(Data[5])*0.000001+ascii_to_int(Data[6])*0.0000001+ascii_to_int(Data[7])*0.00000001
                                                latitude=(latitude_int+latitude_frac)*latitude_flag
                                                #print('latitude = %.8f '%(latitude))
                                                error_GPS_ll+=1
                                            else:
                                                if ID == EPS_ID:
                                                    if Data[2] == 0  and Data[6]== 0:
                                                        error_EPS+=1
                                                        EPS_angle=Data[3]*256+Data[4]
                                                else:
                                                    if ID == state_diagnose_ID:
                                                        if Data[7] == 0:
                                                            error_brake+=1
                                                        if Data[6] == 0:
                                                            error_ac+=1
                                                        if Data[5] ==1:
                                                            state_direction=1
                                                        else:
                                                            if Data[5] ==2:
                                                                state_direction=2
                                                            else:
                                                                state_direction=0
                                                        if Data[4] ==0:
                                                            error_haomibo+=1
                                                        if Data[3] == 0:
                                                            error_EPS_miss+=1
                                                        ac_pwm = Data[2] 
                                                        brake_pwm = Data[1]
                                                    else:
                                                        if ID == soc_ID :
                                                            soc = (Data[4]*256+Data[5])/100
                                                        else:
                                                            if ID == soc_error_ID:
                                                                if Data[5] == 0 and Data[6] == 0:
                                                                    error_battery+=1
        error_capture+=1
#~~~~~~~~~~~~~can收发配置定义结束by龚章鹏~~~~~~~~~~~~~~
#~~~~~~~~~~~~~excel数据读写定义by龚章鹏~~~~~~~~~~~~~~
workbook=xlwt.Workbook(encoding='utf-8')
booksheet=workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
GPS_record_num=0
def load_path_gzp():
    #加载寻迹点文件,如果一切正常GPS_move_flag=4，否则=0
    global GPS_k
    global GPS_move_flag
    global GPS_x
    global GPS_y
    global GPS_xc
    global GPS_yc
    global GPS_d
    global speed
    global brake_signal
    global GPS_yaw
    global GPS_L
    global ex_GPS_direction_flag
    global d_tmp_last
    global GPS_x_tmp
    global GPS_y_tmp
    global GPS_xita
    global error_gps_move
    global GPS_send_flag
    global simulink_flag
    global state
    ####确定正向/反向寻迹(start)####
    
    #print('ex_GPS_direction_flag={}'.format(ex_GPS_direction_flag))
    if ex_GPS_direction_flag == 1 or ex_GPS_direction_flag == 3:
        if os.path.exists('./GPS_save.xls'):
            x = xlrd.open_workbook('./GPS_save.xls')
            sheet1 = x.sheet_by_name('Sheet 1')
            GPS_x_tmp=sheet1.col_values(0)
            GPS_y_tmp=sheet1.col_values(1)
            GPS_x_tmp_num = len(GPS_x_tmp)-2
            GPS_x_tmp=GPS_x_tmp[0:GPS_x_tmp_num]
            GPS_y_tmp=GPS_y_tmp[0:GPS_x_tmp_num]
            print('GPS_trajectory_load')
        if ex_GPS_direction_flag == 1:
            ex_GPS_direction_flag = '1' 
            print('Start at begin')    
        else:
            ex_GPS_direction_flag = '3'
            print('Start at mid')
        GPS_x=GPS_x_tmp
        GPS_y=GPS_y_tmp
        print('~~~~~~~~正向路径加载~~~~~~~~') 
    else:
        if ex_GPS_direction_flag == 2 or ex_GPS_direction_flag == 4:
            if os.path.exists('.\GPS_save.xls'):
                x = xlrd.open_workbook('.\GPS_save.xls')
                sheet1 = x.sheet_by_name('Sheet 1')
                GPS_x_tmp=sheet1.col_values(0)
                GPS_y_tmp=sheet1.col_values(1)
                GPS_x_tmp_num = len(GPS_x_tmp)-2
                GPS_x_tmp=GPS_x_tmp[0:GPS_x_tmp_num]
                GPS_y_tmp=GPS_y_tmp[0:GPS_x_tmp_num]
                print('GPS_trajectory_load')
            GPS_x=GPS_x_tmp[::-1]
            GPS_y=GPS_y_tmp[::-1]
            if ex_GPS_direction_flag == 2:
                ex_GPS_direction_flag = '2'
                print('Start at begin') 
            else:
                ex_GPS_direction_flag = '4'
                print('Start at mid') 
            print('~~~~~~~~反向路径加载~~~~~~~~')
    #GPS_x=[GPS_x[i]-GPS_x[0] for i in range(len(GPS_x))]
    #GPS_y=[GPS_y[i]-GPS_y[0] for i in range(len(GPS_y))]
    ####确定正向/反向寻迹(end)####
    ####确定中间/从头寻迹(start)####
    if ex_GPS_direction_flag == '3' or ex_GPS_direction_flag == '4':
        GPS_move_flag=2  #从最近点下个点开始
        ex_GPS_direction_flag = '5'
    else:
        if ex_GPS_direction_flag == '1' or ex_GPS_direction_flag == '2':
            GPS_move_flag=3 #从零开始
            ex_GPS_direction_flag = '5'
    ####确定中间/从头寻迹(end)####
    ####执行计算GPS_k(start)####
    if simulink_flag:
        if state != None:
            GPS_xc = state.x
            GPS_yc = state.y
            GPS_yaw = state.yaw
        else:
            GPS_xc = 2*GPS_x[0]-GPS_x[1]
            GPS_yc = 2*GPS_y[0]-GPS_y[1]
            GPS_yaw = math.atan2(GPS_y[1]-GPS_y[0],GPS_x[1]-GPS_x[0])
    if GPS_move_flag ==2: #中间寻迹
        GPS_k=0
        k_num= len(GPS_x)
        d_tmp_last = 10000
        for i in range(k_num):
            d_tmp= np.sqrt(((GPS_x[i] - GPS_xc)**2 + (GPS_y[i] - GPS_yc)**2))
            if d_tmp<d_tmp_last:
                GPS_k=i               
                d_tmp_last=d_tmp
        while np.sqrt(((GPS_x[GPS_k] - GPS_xc)**2 + (GPS_y[GPS_k] - GPS_yc)**2))<3 and np.sqrt(((GPS_x[GPS_k] - GPS_xc)**2 + (GPS_y[GPS_k] - GPS_yc)**2))!=0 :#防止点间距过小
            GPS_k=GPS_k+1
            if GPS_k > k_num-2:
                GPS_move_flag =0
                break 
        #print('k_num={}'.format(k_num))
        #print('GPS_k={}'.format(GPS_k))
        if GPS_k<k_num-1:
            GPS_move_flag =1
        else:
            GPS_move_flag =0
            print('~~~~~~~~中间寻迹离目标轨迹太远！！！！！')
    if GPS_move_flag ==3: #从头寻迹，默认情况
        
        GPS_k=0
        d_tmp= np.sqrt(((GPS_x[GPS_k] - GPS_xc)**2 + (GPS_y[GPS_k] - GPS_yc)**2))
        if d_tmp<d_tmp_last:
            GPS_move_flag =1
        else:
            GPS_move_flag =0
            print('GPS_xc={},GPS_yc={},GPS_k={}'.format(GPS_xc,GPS_yc,GPS_k))
            print('~~~~~~~~从头寻迹离目标轨迹太远！！！！！')
    ####执行计算GPS_k(end)####
    if GPS_move_flag==1:
        GPS_x = GPS_x[GPS_k::]
        GPS_y = GPS_y[GPS_k::]
        if not simulink_flag:
            GPS_x = [GPS_xc] + GPS_x
            GPS_y = [GPS_yc] + GPS_y
        GPS_k = 0
        can_send(2794,0,0,0,0,0,0,0,1)
        can_send(512,3,0,0,0,0,0,0,0)
        GPS_move_flag = 4

def GPS_record():
    global GPS_xc
    global GPS_yc
    global GPS_xc_last
    global GPS_yc_last
    global workbook
    global booksheet
    global GPS_record_num
    global ex_dataInput_start_flag
    global ex_dataInput_enpty_flag
    global ex_dataInput_start_flag_tmp
    if ex_dataInput_start_flag == 1: 
        ex_dataInput_start_flag_tmp=1
        print('~~~~~~~~开始采点~~~~~~~~')
        print(GPS_xc)
        GPS_record_d_tmp=np.sqrt((GPS_xc-GPS_xc_last)**2+(GPS_yc-GPS_yc_last)**2)
        if GPS_xc != 0 and GPS_yc != 0 and (GPS_record_num==0 or (GPS_record_d_tmp<500 and GPS_record_d_tmp>2.0)) :
            booksheet.write(GPS_record_num,0,float(GPS_xc))
            booksheet.write(GPS_record_num,1,float(GPS_yc))
            GPS_record_num=GPS_record_num+1
            GPS_xc_last=GPS_xc
            GPS_yc_last=GPS_yc
    else:
        if ex_dataInput_start_flag == 0 and ex_dataInput_start_flag_tmp == 1:
            workbook.save('.\GPS_save.xls')
            ex_dataInput_start_flag_tmp = 0
            GPS_record_num=0
            print('~~~~~~~~采点结束~~~~~~~~')
        else:
            if ex_dataInput_enpty_flag ==1:
                if os.path.exists('.\GPS_save.xls'):
                    os.remove('.\GPS_save.xls')
                    print('~~~~~~~~数据清除~~~~~~~~')
                GPS_record_num = 0
                ex_dataInput_enpty_flag = 0

#~~~~~~~~~~~~~excel数据读写定义结束by龚章鹏~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~函数定义~~~~~~~~~~~~~~~~~
def cacu_angle_from_p(x1,y1,x2,y2):
    #工具函数:x1,y1为车身坐标；x2,y2为目标坐标，输出位置方向角
    global GPS_yaw
    GPS_x_difference=x1 - x2 
    GPS_y_difference = y1 - y2
    if GPS_y_difference==0:#目标点在正东或正西
        if GPS_x_difference>0:
            tan_tmp=np.pi/2
        else:
            tan_tmp=3*np.pi/2
    else:
        if GPS_x_difference == 0:#目标点在正南或正北
            if GPS_y_difference > 0:
                tan_tmp=0
            else:
                tan_tmp=np.pi
        else:
            if GPS_y_difference==0:#目标点在正东或正西
                if GPS_x_difference>0:
                    tan_tmp=np.pi/2
                else:
                    tan_tmp=3*np.pi/2
            else:
                if GPS_x_difference == 0:#目标点在正南或正北
                    if GPS_y_difference > 0:
                        tan_tmp=0
                    else:
                        tan_tmp=np.pi
                else:
                    if GPS_y_difference>0 and GPS_x_difference>0:#第一象限
                        tan_tmp=np.arctan(GPS_x_difference/GPS_y_difference)
                    else:
                        if GPS_y_difference<0 and GPS_x_difference>0:#第四象限
                            tan_tmp=np.pi-np.arctan(-GPS_x_difference/GPS_y_difference)
                        else:
                            if GPS_y_difference<0 and GPS_x_difference<0:#第三象限
                                tan_tmp=np.pi+np.arctan(GPS_x_difference/GPS_y_difference)
                            else:
                                if GPS_y_difference>0 and GPS_x_difference<0:#第二象限
                                    tan_tmp=2*np.pi-np.arctan(-GPS_x_difference/GPS_y_difference)                       
            tan_tmp=tan_tmp/np.pi*180-GPS_yaw
            while tan_tmp<0:
                tan_tmp=tan_tmp+360
            while tan_tmp>360:
                tan_tmp=tan_tmp-360
    return tan_tmp

def arctan2_gzp(x):
    #工具函数：arctan泰勒展开（暂不用）
    a=0
    if x>0.7:
        a = 0.6193067 + 0.5536375*(x-0.7)
    else:
        if x<-0.7:
            a = -0.6193067 + 0.5536375*(0.7+x)
        else:
            a = x - 0.3333333*x**3 + 0.2*x**5
    return a

def sin_gzp(x):
    #工具函数：sin泰勒展开（暂不用）
    a = x - 0.1666667*x**3 + 0.0083333*x**5
    return a
    
def generate_box_four_points(x,y,yaw):
    #工具函数：由车身的坐标与航向角获取车身四角点坐标
    ru_x=x+math.cos(yaw)*l_a+math.sin(yaw)*w_half
    rd_x=x-math.cos(yaw)*l_b+math.sin(yaw)*w_half
    lu_x=x+math.cos(yaw)*l_a-math.sin(yaw)*w_half
    ld_x=x-math.cos(yaw)*l_b-math.sin(yaw)*w_half
    
    ru_y=y+math.sin(yaw)*l_a-math.cos(yaw)*w_half
    rd_y=y-math.sin(yaw)*l_b-math.cos(yaw)*w_half
    lu_y=y+math.sin(yaw)*l_a+math.cos(yaw)*w_half
    ld_y=y-math.sin(yaw)*l_b+math.cos(yaw)*w_half
    return ru_x,ru_y,rd_x,rd_y,lu_x,lu_y,ld_x,ld_y
    
def  coordinate_trans_lidar2world(x_lidar,y_lidar,GPS_xc,GPS_yc,GPS_yaw,b):
    #工具函数：对点进行车身坐标系到世界坐标系转换，b为传感器到后轴中心安装距离
    x = b *math.cos(GPS_yaw)+ x_lidar * math.sin(GPS_yaw) +y_lidar*math.cos(GPS_yaw)+GPS_xc
    y = b *math.sin(GPS_yaw)-x_lidar * math.cos(GPS_yaw) +y_lidar*math.sin(GPS_yaw)+GPS_yc
    return x,y

def oblist_to_world_cordination(oblist,GPS_xcc,GPS_ycc,GPS_yawc,l=1.5):
    #工具函数：对障碍物表进行车身坐标系到世界坐标系转换，l为传感器到后轴中心安装距离
    oblist = np.array(oblist)
    num = oblist.shape[0]
    #l=1.14#1.59
    oblist_world = []
    oblist_keep = np.zeros([256,3])
    for i in range(num):
        x,y,v_x,v_y = oblist[i,0:4]
        if x>-10 and x<10 and y>0 and y<20:
            x_world = (l+y)*math.cos(GPS_yawc) + x*math.sin(GPS_yawc)+GPS_xcc
            y_world = (l+y)*math.sin(GPS_yawc) - x*math.cos(GPS_yawc)+GPS_ycc
            #x_world = np.floor(x_world*10)/10 #0.05m误差
            #y_world = np.floor(y_world*10)/10
            oblist_world.append([x_world,y_world])
    return oblist_world

def cal_dmin_to_ob(pc,pf,pr,ob):
    #工具函数：计算最近障碍物分别到轴线前中后的最短距离
    if len(ob)==0:
        return 20,20,20
    d1= [np.sqrt((a[0]-pc[0])**2+(a[1]-pc[1])**2) for a in ob]
    d2= [np.sqrt((a[0]-pf[0])**2+(a[1]-pf[1])**2) for a in ob]
    d3= [np.sqrt((a[0]-pr[0])**2+(a[1]-pr[1])**2) for a in ob]
    dmin1 = np.min(d1)
    dmin2 = np.min(d2)
    dmin3 = np.min(d3)
    return np.minimum(dmin1,20),np.minimum(dmin2,20),np.minimum(dmin3,20)
    
#~~~~~~~~~~~~~~~~~函数定义结束~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~新版毫米波采集函数by龚章鹏~~~~~~~~~~~~~~
def new_haomibo():
    #毫米波数据生成障碍物表obstacle_list
    global recv_radar
    global obstacle_list
    global obstacle_listh
    global ex_AEB_h
    global ex_AEB_w
    global ac_flag
    global dwa_block_flag
    ob_num = 256
    obstacle_listh = np.zeros((ob_num,4))
    p_world = np.zeros(2)
    #毫米波原始数据recv_radar(256,8)中的有效数据存入障碍列表，然后放进地图
    for k_3 in range(ob_num):
        if recv_radar[k_3] is [0,0,0,0,0,0,0,0]:
            continue
        templeData = (int(recv_radar[k_3][1])<<4 | (int(recv_radar[k_3][2])&0xF0)>>4)  # x/m
        obstacle_listh[k_3][0] = float(templeData-2048)/10
        templeData = ((int(recv_radar[k_3][2])&0x0F)<<8 | int(recv_radar[k_3][3]))  # y/m
        obstacle_listh[k_3][1] =  float(templeData)/10
        templeData = ((int(recv_radar[k_3][5])&0xE0)>>5 | int(recv_radar[k_3][4])<<3)  # V_x/m/s
        obstacle_listh[k_3][2] = float(templeData-1024)/10
        templeData = ((int(recv_radar[k_3][5])&0x07)<<8 | int(recv_radar[k_3][6]))  # V_y/m/s
        obstacle_listh[k_3][3] = float(templeData-1024)/10
        recv_radar[k_3] = [0,0,0,0,0,0,0,0]
        if ((obstacle_listh[k_3][0]!=0) or (obstacle_listh[k_3][1]!=0)):
            if (abs(obstacle_listh[k_3][1])<=float(ex_AEB_h)/10) and (abs(obstacle_listh[k_3][0])<=float(ex_AEB_w)/20):
                #ac_flag = 1  # AEB反应区域内看见障碍标志
                pass
    obstacle_list = obstacle_listh
#~~~~~~~~~~~~~新版毫米波采集函数结束by龚章鹏~~~~~~~~~~~~~~
#~~~~~~~~~~~~~动态轨迹规划函数by龚章鹏~~~~~~~~~~~~~~
def frenet_path_planning_init():
    #动态轨迹规划初始化
    global GPS_xc#计算用变量定义
    global GPS_yc
    global GPS_x
    global GPS_y
    global GPS_s
    global GPS_yawarray
    global GPS_t_yaw
    global s0
    global c_speed
    global c_a
    global c_d
    global c_d_d
    global c_d_dd
    global target_ind_last
    global target_ind
    global csp
    global state#仿真用变量
    global time_record
    global last_time_record
    global simulink_flag #标识符
    global error_c #记录用非必须变量
    global x_set
    global y_set
    global yaw_set
    global time_dur_set
    global time_record_set
    global delta_set
    global error_set
    global error_set2
    global error_set3
    global path_set
    global dc_ob_set
    global df_ob_set
    global dr_ob_set
    global ob_set
    global c_speed_set
    x_set = []
    y_set = []
    yaw_set = []
    time_dur_set = []
    time_record_set = []
    delta_set = []
    error_set = []
    error_set2 = []
    error_set3 = []
    path_set = []
    dc_ob_set = []
    df_ob_set = []
    dr_ob_set = []
    ob_set = []
    c_speed_set = []
    s0 = 0
    c_speed =0/3.6
    c_a =0
    c_d = 0
    c_d_d = 0
    c_d_dd = 0
    time_record = 0
    target_ind_last = 0
    target_ind = 0
    error_c= 0
    #if GPS_xc!=0 and GPS_yc!=0:
    #    GPS_x = [GPS_xc] + GPS_x
    #    GPS_y = [GPS_yc] + GPS_y
    if simulink_flag:
        state = VehicleState(l_a=1.0,l_b=0.8,x=GPS_x[0],y=GPS_y[0]-1,yaw=np.pi/2,v=c_speed)#yaw是正东逆时针角
        GPS_xc = GPS_x[0]
        GPS_yc = GPS_y[0]
    last_time_record = time.time()
    tx, ty, tyaw, tc, ts, csp = generate_target_course(GPS_x,GPS_y)
    GPS_x = tx
    GPS_y = ty
    GPS_s = ts
    GPS_yawarray = tyaw
    print('Gnerate_target_course_sucessed, point_num = {}'.format(len(GPS_x)))
    GPS_t_yaw = tyaw

def frenet_path_planning(path_planning_flag):
    #动态轨迹规划执行函数
    global GPS_x#计算用变量定义
    global GPS_y
    global GPS_s
    global GPS_yawarray
    global GPS_xc
    global GPS_yc
    global s0
    global c_speed
    global c_a
    global c_d
    global c_d_d
    global c_d_dd
    global GPS_yaw
    global csp
    global rwheel_vel_100
    global ob
    global ob_new
    global obstacle_list
    global target_ind
    global target_ind_last
    global v_state
    global GPS_tx
    global GPS_ty
    global delta
    global target_v
    global path
    global path_last
    global state#仿真用变量
    global last_time_record
    global time_record
    global send_turnID#执行控制canID
    global send_accelerateID
    global send_brakeID
    global thred_flag#标识符
    global move_thred_flag
    global show_animation
    global ac_flag
    global simulink_flag
    global GPS_move_flag
    global ex_flag
    global error_c #记录用非必须变量
    global x_set
    global y_set
    global yaw_set
    global time_dur_set
    global time_record_set
    global delta_set
    global error_set
    global error_set2
    global error_set3
    global dc_ob_set
    global df_ob_set
    global dr_ob_set
    global ob_set
    global c_speed_set
    global path_set
    global dc_ob_set
    global df_ob_set
    global dr_ob_set
    min_c_speed = 1 #越大越有可能得到最佳路线，但过大无法得到较小分辨率
    max_c_speed = 1.5
    ob_new=np.copy(ob)
    path = frenet_optimal_planning(csp, s0, v_state , c_d, c_d_d, c_d_dd, ob_new,time_record)
    if path == None:
        print('Can not avoid!')
        c_speed = 0
        can_send(send_accelerateID,0,0,0,0,0,0,0,0) #停车指令
        can_send(send_brakeID,0,0,0,0,0,0,0,1)  #制动指令
        sleep(1)
        can_send(send_brakeID,0,0,0,0,0,0,0,0)
        if simulink_flag:
            dt = time.time()-last_time_record
            update(state,c_speed,delta,dt)
            GPS_xcc = state.x
            GPS_ycc = state.y
            GPS_xcf = state.fx
            GPS_ycf = state.fy
            GPS_xcr = state.rx
            GPS_ycr = state.ry
            GPS_yawc = state.yaw
            
        else:
            dt = time.time()-last_time_record
            GPS_xcc = GPS_xc+0.8*math.cos(np.pi/2-GPS_yaw*np.pi/180)
            GPS_ycc = GPS_yc+0.8*math.sin(np.pi/2-GPS_yaw*np.pi/180)
            GPS_xcf = GPS_xc+1.59*math.cos(np.pi/2-GPS_yaw*np.pi/180)
            GPS_ycf = GPS_yc+1.59*math.sin(np.pi/2-GPS_yaw*np.pi/180)
            GPS_xcr = GPS_xc
            GPS_ycr = GPS_yc
            GPS_yawc = np.pi/2-GPS_yaw*np.pi/180
            
        #数据记录，车身状态部分
        last_time_record = time.time()
        x_set.append(GPS_xcc)
        y_set.append(GPS_ycc)
        yaw_set.append(GPS_yawc)
        time_dur_set.append(dt)
        time_record_set.append(time_record)
        delta_set.append(delta)
        #数据记录，轨迹部分
        if path_last != None:
            target_ind_tmp = calc_target_index(GPS_xcc,GPS_ycc,path_last.x,path_last.y)
            flag = sign_l_or_r(GPS_xcc,GPS_ycc,path_last.x[target_ind_tmp],path_last.y[target_ind_tmp],path_last.yaw[target_ind_tmp])
            error_c = -flag*cacl_min_d(GPS_xcc,GPS_ycc,path_last.x,path_last.y,target_ind_tmp)
            target_ind_tmp = calc_target_index(GPS_xcr,GPS_ycr,path_last.x,path_last.y)
            flag = sign_l_or_r(GPS_xcr,GPS_ycr,path_last.x[target_ind_tmp],path_last.y[target_ind_tmp],path_last.yaw[target_ind_tmp])
            error_c2 = -flag*cacl_min_d(GPS_xcr,GPS_ycr,path_last.x,path_last.y,target_ind_tmp)
            target_ind_tmp = calc_target_index(GPS_xcf,GPS_ycf,path_last.x,path_last.y)
            flag = sign_l_or_r(GPS_xcf,GPS_ycf,path_last.x[target_ind_tmp],path_last.y[target_ind_tmp],path_last.yaw[target_ind_tmp])
            error_c3 = -flag*cacl_min_d(GPS_xcf,GPS_ycf,path_last.x,path_last.y,target_ind_tmp)
            error_set.append(error_c)
            error_set2.append(error_c2)
            error_set3.append(error_c3)
            dc_ob,df_ob,dr_ob = cal_dmin_to_ob([GPS_xcc,GPS_ycc],[GPS_xcf,GPS_ycf],[GPS_xcr,GPS_ycr],ob)
            dc_ob_set.append(dc_ob)
            df_ob_set.append(df_ob)
            dr_ob_set.append(dr_ob)
            ob_set.append(ob_new)
            path_set.append(path)
            c_speed_set.append(c_speed)
        return path_planning_flag+1
    if path_last == None:
        path_last = path
    #仿真车辆模型状态更新
    if simulink_flag:
        dt = time.time()-last_time_record
        update(state,c_speed,delta,dt)
        GPS_xcc = state.x
        GPS_ycc = state.y
        GPS_xcf = state.fx
        GPS_ycf = state.fy
        GPS_xcr = state.rx
        GPS_ycr = state.ry
        GPS_yawc = state.yaw
        v_state = state.v
        print(dt)      
        
    else:
     #车辆状态更新
        dt = time.time()-last_time_record
        GPS_xcc = GPS_xc+0.8*math.cos(np.pi/2-GPS_yaw*np.pi/180)
        GPS_ycc = GPS_yc+0.8*math.sin(np.pi/2-GPS_yaw*np.pi/180)
        GPS_xcf = GPS_xc+1.59*math.cos(np.pi/2-GPS_yaw*np.pi/180)
        GPS_ycf = GPS_yc+1.59*math.sin(np.pi/2-GPS_yaw*np.pi/180)
        GPS_xcr = GPS_xc
        GPS_ycr = GPS_yc
        GPS_yawc = np.pi/2-GPS_yaw*np.pi/180
        v_state = c_speed   #rwheel_vel_100/100
        print(dt)
    last_time_record = time.time()
    target_ind = calc_target_index(GPS_xcc,GPS_ycc,path.x,path.y)
    #print('到目标距离={}'.format(np.sqrt((GPS_xcc-GPS_x[-1])**2+(GPS_ycc-GPS_y[-1])**2)))
    delta,target_ind,error_c = stanley_control(GPS_xcc,GPS_ycc,GPS_yawc,max(v_state,min_c_speed),path.x,path.y,path.yaw,target_ind,error_c) #计算的delta向左为正
    if delta <0:
        ot = int((-delta+0.8075)*39114)  # 输出的前轮delta角到canID执行报文数据，需标定，左右单独标定
    else:
        ot = int((-delta+1.1147)*28333)
    ot_low = int(ot%256)
    ot_high = int((ot-ot_low)/256)
    can_send(send_turnID,0,0,0,0,0,0,ot_high,ot_low)#转角执行发给下位机
    target_v= math.sqrt(path.s_d[1]**2+path.d_d[1]**2)
    c_speed = max(target_v,min_c_speed)
    c_speed = min(c_speed,max_c_speed)
    vt = int(c_speed*10)
    can_send(send_accelerateID,0,0,0,0,0,0,1,vt) #速度执行发给下位机,dm/s
    #数据记录部分
    target_ind_tmp = calc_target_index(GPS_xcc,GPS_ycc,path_last.x,path_last.y)
    flag = sign_l_or_r(GPS_xcc,GPS_ycc,path_last.x[target_ind_tmp],path_last.y[target_ind_tmp],path_last.yaw[target_ind_tmp])
    error_c = -flag*cacl_min_d(GPS_xcc,GPS_ycc,path_last.x,path_last.y,target_ind_tmp)
    target_ind_tmp = calc_target_index(GPS_xcr,GPS_ycr,path_last.x,path_last.y)
    flag = sign_l_or_r(GPS_xcr,GPS_ycr,path_last.x[target_ind_tmp],path_last.y[target_ind_tmp],path_last.yaw[target_ind_tmp])
    error_c2 = -flag*cacl_min_d(GPS_xcr,GPS_ycr,path_last.x,path_last.y,target_ind_tmp)
    target_ind_tmp = calc_target_index(GPS_xcf,GPS_ycf,path_last.x,path_last.y)
    flag = sign_l_or_r(GPS_xcf,GPS_ycf,path_last.x[target_ind_tmp],path_last.y[target_ind_tmp],path_last.yaw[target_ind_tmp])
    error_c3 = -flag*cacl_min_d(GPS_xcf,GPS_ycf,path_last.x,path_last.y,target_ind_tmp)
    error_set.append(error_c)
    error_set2.append(error_c2)
    error_set3.append(error_c3)
    dc_ob,df_ob,dr_ob = cal_dmin_to_ob([GPS_xcc,GPS_ycc],[GPS_xcf,GPS_ycf],[GPS_xcr,GPS_ycr],ob)
    dc_ob_set.append(dc_ob)
    df_ob_set.append(df_ob)
    dr_ob_set.append(dr_ob)
    time_record += dt
    x_set.append(GPS_xcc)
    y_set.append(GPS_ycc)
    yaw_set.append(GPS_yawc)
    time_record_set.append(time_record)
    time_dur_set.append(dt)
    delta_set.append(delta)
    ob_set.append(ob_new)
    path_set.append(path)
    c_speed_set.append(c_speed)
    path_last = path
    if simulink_flag:
        c_speed_real =c_speed+(np.random.randint(8)-5)*0.01
    else:
        c_speed_real = max(rwheel_vel_100/100,min_c_speed)#path.s_d[target_ind]#
    #轨迹规划函数输入状态更新
    c_speed = max(c_speed_real,min_c_speed)
    target_ind_tmp = calc_target_index(GPS_xcc,GPS_ycc,GPS_x,GPS_y)
    flag = sign_l_or_r(GPS_xcc,GPS_ycc,GPS_x[target_ind_tmp],GPS_y[target_ind_tmp],GPS_yawarray[target_ind_tmp])
    c_d_tmp = -flag*np.sqrt((GPS_xcc-GPS_x[target_ind_tmp])**2+(GPS_ycc-GPS_y[target_ind_tmp])**2)#cacl_min_d(GPS_xcc,GPS_ycc,GPS_x,GPS_y,target_ind_tmp)#左负右正
    s0 = GPS_s[target_ind_tmp]
    #flag = sign_l_or_r(GPS_xcc,GPS_ycc,path.x[target_ind],path.y[target_ind],path.yaw[target_ind])
    #c_d_tmp = -flag*np.sqrt((GPS_xcc-path.x[target_ind])**2+(GPS_ycc-path.y[target_ind])**2)
    c_d = c_d_tmp
    c_d_d = path.d_d[target_ind]
    c_d_dd = path.d_dd[target_ind]
    #函数终止条件
    if np.hypot(path.x[1] - GPS_x[-1], path.y[1] - GPS_y[-1]) <= 1.0: 
        print('Arrived!')
        delta = 0 
        if delta <0:
            ot = int((-delta+0.8075)*39114)  # 输出的前轮delta角，左最大16749：30°,0.5236,，11104,0x2b 60 中位31584，0x7b 60 右边最大 52064,0xcb 60
        else:
            ot = int((-delta+1.1147)*28333)
        ot_low = int(ot%256)
        ot_high = int((ot-ot_low)/256)
        can_send(send_turnID,0,0,0,0,0,0,ot_high,ot_low)#转角执行发给下位机
        can_send(send_accelerateID,0,0,0,0,0,0,1,0) #速度执行发给下位机
        can_send(send_brakeID,0,0,0,0,0,0,0,1)  #制动指令
        time.sleep(5)
        can_send(send_brakeID,0,0,0,0,0,0,0,0)
        GPS_move_flag = 0
        move_thred_flag = False
        ex_flag = 2
        return 10001
    return 0
#~~~~~~~~~~~~~动态轨迹规划函数结束by龚章鹏~~~~~~~~~~~~~~
#~~~~~~~~~~~~~旧版毫米波雷达by龚章鹏(暂不用)~~~~~~~~~~~~~~
def haomibo(Data,ID):
    global radar_range
    global radar_speed
    global radar_angle
    global radar_power
    global obstacle_range
    global obstacle_speed
    global break_d
    global rwheel_vel_100
    global ac
    global sin_value
    global t1
    # 旧版毫米波
    a=int(ID-1280)
    templeData = int(Data[0] | (Data[1] & 127)*2**8)
    radar_range[a] = templeData/10 #分米
    templeData = int(((Data[3] & 63) *2**8) | Data[2])
    if templeData>8191:
        radar_speed[a]= (templeData - 16384)/10 #分米/s
    else:
        templeData/10
    templeData = int(((Data[5] & 252) //2**2) | ((Data[6] & 31) *2**8))
    if templeData > 1023:
        radar_angle[a]=(templeData- 2048)/10
    else:
        radar_angle[a]=templeData /10  #单位°
    if radar_angle[a]<-60:
        radar_angle[a]=-60
    else:
        if radar_angle[a]>59:
            radar_angle[a]=59
    if radar_range[a] > 19:
        templeData =int(((Data[6] & 224) //2**5) | ((Data[7] & 127) *2**3))
        if templeData > 511:
            radar_power[a]=(templeData - 1024) - 400
        else:
            radar_power[a]=templeData - 400
    else:
        radar_power[a] = 0
    for k in range(64):
        if radar_range[k]>0:
            obstacle_range[int(radar_angle[k])+60]=radar_range[k]
            obstacle_speed[int(radar_angle[k])+60]=radar_speed[k]
            vc= (rwheel_vel_100)/10
            break_d[int(radar_angle[k])+60] = obstacle_range[int(radar_angle[k])+60]+ obstacle_speed[int(radar_angle[k])+60]*t1/10 

def lvbo_inten(a,b):
    tmp_0=200;
    tmp_1=200;
    tmp_2=200;
    for k in range(10):
        if a[10*b+k]!=0 and a[10*b+k] <tmp_0:
            tmp_2=tmp_1
            tmp_1=tmp_0
            tmp_0=a[10*b+k]
    return tmp_0

def AEB():
    global break_d_inten
    global break_d
    global break_d_thresh
    global obstacle_range
    global ac_flag
    global obstacle_speed
    global radar_range
    global radar_angle
    global radar_speed
    global error_AEB
    ac_flag_tmp=0
    for k in range(6):
        break_d_inten[6+k] = lvbo_inten(break_d,6+k)
        break_d_inten[5-k] = lvbo_inten(break_d,5-k)
        if (break_d_inten[6+k]<break_d_thresh[6+k] or break_d_inten[5-k]<break_d_thresh[5-k]):
            ac_flag_tmp = 1
    if ac_flag_tmp==1:
        ac_flag = 1;    #刹车
    else:
        ac_flag = 0;    #不刹车
    obstacle_range=np.zeros([120])
    obstacle_speed=np.zeros([120])
    radar_range=np.ones([64])*200
    radar_angle=np.ones([64])*200
    radar_speed=np.ones([64])*200
    error_AEB+=1
    return ac_flag
#~~~~~~~~~~~~~旧版毫米波雷达结束by龚章鹏(暂不用)~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~线程设计by龚章鹏~~~~~~~~~~~~~~~~
def capture_thred():
    #can总线接收线程
    global thred_flag
    while thred_flag:
        capture()
        sleep(0.02)
    print('1退出 capture_thred')

def can_send_thred():
    #制动控制+云端控制线程
    global thred_flag
    global brake_signal
    global GPS_xita
    global ac_flag
    global send_brakeID
    global send_accelerateID
    global send_turnID
    global ex_flag
    global error_can_send
    global brake
    global ex_v                  # 设定车速km/h ，浮点数
    global ex_start_direction    # 0代表前进，1代表倒车
    global ex_start_state        # 1代表启动，0代表停止
    global ex_traffic_signal
    global cloud_time_num
    global rwheel_vel_100
    global label
    global GPS_xc
    global GPS_yc
    global pre_GPS_xc
    global pre_GPS_yc
    global brake_flag
    while thred_flag: 
        #制动控制部分，有brake_signal（红灯）和ac_flag（AEB有障碍物）来判定是否执行制动指令
        if brake_signal == 1 or ac_flag == 1:
            speed = 0
            brake = 1
            can_send(send_brakeID,0,0,0,0,0,0,0,brake)
            brake_flag = True
        elif brake_flag == True:
            speed=int(float(ex_v)*10/3.6)
            brake = 0
            can_send(send_brakeID,0,0,0,0,0,0,0,brake)
            brake_flag = False
        if ex_start_direction ==1:
            start_direction_tmp =2
        else:
            start_direction_tmp = 1
        
        #云端控制部分
        if cloud_time_num>=1:
            cloud_time_num+=1
        if ex_start_state==1 and rwheel_vel_100<=5:
            if start_direction_tmp ==2:
                can_send(send_accelerateID,0,0,0,0,0,0,start_direction_tmp,0)#后退
                sleep(1.5)
                can_send(send_accelerateID,0,0,0,0,0,0,start_direction_tmp,speed)
            else:
                can_send(send_accelerateID,0,0,0,0,0,0,start_direction_tmp,0)#后退
                sleep(1.5)
                can_send(send_accelerateID,0,0,0,0,0,0,start_direction_tmp,speed)#前进
            ex_start_state = 2
            cloud_time_num=1
            print('云端控制：启动')
        else:
            if ex_start_state==0:
                pre_GPS_xc=GPS_xc
                pre_GPS_yc=GPS_yc
                #can_send(send_accelerateID,0,0,0,0,0,0,start_direction_tmp,speed)
                ex_start_state = 2
                start_direction_tmp = 0
                ex_start_direction = 0
                can_send(send_accelerateID,0,0,0,0,0,0,start_direction_tmp,0)
                cloud_time_num=0
                can_send(send_brakeID,0,0,0,0,0,0,0,1)
                sleep(2)
                print('云端控制：停止')
                can_send(send_accelerateID,0,0,0,0,0,0,0,0)
                can_send(send_brakeID,0,0,0,0,0,0,0,0)
                can_send(512,0,0,0,0,0,0,0,0)
                can_send(send_turnID,0,0,0,0,0,1,109,96)
                print('云端控制：退出')
        if cloud_time_num>1000 or np.sqrt((GPS_yc-pre_GPS_yc)**2+(GPS_xc-pre_GPS_xc)**2)>10: #失联100*0.05=50s，或距离移动超过10m停止
            #ex_start_state=0
            pass
        error_can_send+=1
        sleep(0.05)

def send_can_tc_thred():
    #初始可调参数发送线程
    global tc_flag
    global tc_AEB_w
    global tc_AEB_h
    global tc_v
    global tc_ob
    global tc_AEB_f
    global tc_turn_mid
    global tc_turn_cp
    global hf_flag
    global hf_AEB_w
    global hf_AEB_h
    global hf_v
    global hf_ob
    global hf_AEB_f
    global hf_turn_mid
    global hf_turn_cp
    global hf_AEB
    global hf_ac
    global hf_brake
    global hf_ob_start
    global send_brakeID
    global send_turnID
    global send_accelerateID
    while thred_flag:
        if hf_AEB_w==1:
            v_tmp=min(max(int(tc_AEB_w),20),200)
            a0,a1,a2,a3,a4,a5,a6,a7=data_to_can(int(v_tmp))
            can_send(1102,a0,a1,a2,a3,a4,a5,a6,a7)
            hf_AEB_w=2
        if hf_AEB_h==1:
            v_tmp=min(max(int(tc_AEB_h),20),200)
            a0,a1,a2,a3,a4,a5,a6,a7=data_to_can(int(v_tmp))
            can_send(1103,a0,a1,a2,a3,a4,a5,a6,a7)
            hf_AEB_w=2
        if hf_v==1:
            tc_v_tmp=float(tc_v)*10/3.6
            v_tmp=min(max(int(tc_v_tmp),0),100)
            a0,a1,a2,a3,a4,a5,a6,a7=data_to_can(int(v_tmp))
            can_send(1104,a0,a1,a2,a3,a4,a5,a6,a7)
            #执行
            hf_v=2
        if hf_ob==1:
            v_tmp=min(max(int(tc_ob),0),1000)
            a0,a1,a2,a3,a4,a5,a6,a7=data_to_can(int(v_tmp))
            can_send(1105,a0,a1,a2,a3,a4,a5,a6,a7)
            #执行
            hf_ob=2
        if hf_AEB_f==1:
            v_tmp=min(max(int(tc_AEB_f),50),1000)
            a0,a1,a2,a3,a4,a5,a6,a7=data_to_can(int(v_tmp))
            can_send(1106,a0,a1,a2,a3,a4,a5,a6,a7)
            hf_AEB_f=2
        if hf_turn_mid ==1:
            v_tmp=min(max(int(tc_turn_mid),800),1200)
            a0,a1,a2,a3,a4,a5,a6,a7=data_to_can(int(v_tmp))
            can_send(1107,a0,a1,a2,a3,a4,a5,a6,a7)
            hf_turn_mid=2
        if hf_turn_mid ==0:
            can_send(send_turnID,0,0,0,0,0,1,109,96)
            hf_turn_mid=2
        if hf_turn_cp ==1:
            v_tmp=min(max(int(tc_turn_cp),0),100)
            a0,a1,a2,a3,a4,a5,a6,a7=data_to_can(int(v_tmp))
            can_send(1108,a0,a1,a2,a3,a4,a5,a6,a7)
            hf_turn_cp=2
        if hf_brake == 1:
            can_send(send_brakeID,0,0,0,0,0,0,0,1)
            hf_brake =2
        if hf_brake == 0:
            can_send(send_brakeID,0,0,0,0,0,0,0,0)
            hf_brake =2 
        if hf_ac == 1:
            can_send(send_accelerateID,0,0,0,0,0,0,0,1)
            hf_ac =2
        if hf_ac == 0:
            can_send(send_accelerateID,0,0,0,0,0,0,0,0)
            hf_ac =2 
        if hf_AEB == 1:
            can_send(1110,0,0,0,0,0,0,0,1)
            hf_AEB =2
        if hf_AEB == 0:
            can_send(1110,0,0,0,0,0,0,0,0)
            hf_AEB =2
        if hf_ob_start == 1:
            can_send(1111,0,0,0,0,0,0,0,1)
            hf_ob_start =2
        if hf_ob_start == 0:
            can_send(1111,0,0,0,0,0,0,0,0)
            hf_ob_start =2 
        sleep(0.5)

def send_can_to_micro_thred():
    #单步调试线程
    global ex_flag
    global ex_AEB_w
    global ex_AEB_h
    global ex_v
    global ex_ob
    global ex_AEB_f
    global ex_turn_mid
    global ex_turn_cp
    global GPS_move_flag
    global ex_GPS_direction_flag
    global thred_flag
    while thred_flag:
        if ex_flag==1:
            v_tmp=min(max(int(ex_AEB_w),20),200)
            a0,a1,a2,a3,a4,a5,a6,a7=data_to_can(int(v_tmp))
            can_send(1102,a0,a1,a2,a3,a4,a5,a6,a7)
            v_tmp=min(max(int(ex_AEB_h),20),200)
            a0,a1,a2,a3,a4,a5,a6,a7=data_to_can(int(v_tmp))
            can_send(1103,a0,a1,a2,a3,a4,a5,a6,a7)
            ex_v_tmp=float(ex_v)*10/3.6
            v_tmp=min(max(int(ex_v_tmp),0),100)
            a0,a1,a2,a3,a4,a5,a6,a7=data_to_can(int(v_tmp))
            print('ex_v={}'.format(v_tmp))
            can_send(1104,a0,a1,a2,a3,a4,a5,a6,a7)
            v_tmp=min(max(int(ex_ob),0),300)
            a0,a1,a2,a3,a4,a5,a6,a7=data_to_can(int(v_tmp))
            can_send(1105,a0,a1,a2,a3,a4,a5,a6,a7)
            v_tmp=min(max(int(ex_AEB_f),50),1000)
            a0,a1,a2,a3,a4,a5,a6,a7=data_to_can(int(v_tmp))
            can_send(1106,a0,a1,a2,a3,a4,a5,a6,a7)
            v_tmp=min(max(int(ex_turn_mid),800),1200)
            a0,a1,a2,a3,a4,a5,a6,a7=data_to_can(int(v_tmp))
            can_send(1107,a0,a1,a2,a3,a4,a5,a6,a7)
            v_tmp=min(max(int(ex_turn_cp),0),100)
            a0,a1,a2,a3,a4,a5,a6,a7=data_to_can(int(v_tmp))
            can_send(1108,a0,a1,a2,a3,a4,a5,a6,a7)
            print('paraments_set_succeed')
            break
        sleep(0.2)

def gps_move_thred():
    #动态路径规划线程
    global move_thred_flag
    global GPS_move_flag
    global frenet_init_flag
    path_planning_flag = 0
    while move_thred_flag:
        load_path_gzp()
        if GPS_move_flag == 4:
            if frenet_init_flag == False:
                frenet_path_planning_init()
                frenet_init_flag = True
                sleep(1)
            path_planning_flag = frenet_path_planning(path_planning_flag)
            if path_planning_flag>10000:
                GPS_move_flag = 0
                print('无法避障，已等待100s，退出寻迹线程')
                break
        sleep(0.01)
    
def lidar_thred():
    #雷达采集+三维目标识别线程，生成激光雷达障碍物表ob_lidar
    global thred_flag
    global ob
    global ob_lidar
    global GPS_xc
    global GPS_yc
    global GPS_yaw
    while thred_flag:
        raw_lidar = getpoints()#get_lidar_data()
        #lidar_path = 'E:\\zip_voxelnet\\data\\object\\validation\\velodyne\\000013.bin'
        #raw_lidar = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
        GPS_xcc=GPS_xc
        GPS_ycc=GPS_yc
        GPS_yawc=GPS_yaw
        ret_box3d_score = lidar_object_recognition(raw_lidar)
        corner_points = box2Dcorner_from_ret_box3d_score(ret_box3d_score)
        ob_lidar = np.array(oblist_to_world_cordination(corner_points,GPS_xcc,GPS_ycc,GPS_yawc,1.14)).reshape([-1,2])
        sleep(0.01)
    
def ridar_thred():
    #障碍物表生成线程，毫米波啊雷达障碍物表obstacle_list，激光雷达障碍物表ob_lidar，生成障碍物表ob
    global thred_flag
    global ob
    global ob_lidar
    global ob_tmp8
    global ob_tmp7
    global ob_tmp6
    global ob_tmp5
    global ob_tmp4
    global ob_tmp3
    global ob_tmp2
    global ob_tmp1
    global simulink_flag
    global GPS_x
    global GPS_y
    global GPS_move_flag
    global obstacle_list
    global GPS_xc
    global GPS_yc
    global GPS_yaw
    while thred_flag:
        if simulink_flag and GPS_move_flag == 4:
            noisy_k=0.01
            ob_tmp8 = ob_tmp7
            ob_tmp7 = ob_tmp6
            ob_tmp6 = ob_tmp5
            ob_tmp5 = ob_tmp4
            ob_tmp4 = ob_tmp3
            ob_tmp3 = ob_tmp2
            ob_tmp2 = ob_tmp1
            ob_tmp1 = [np.array([444489.63+0.5+(np.random.randint(100)-50)*noisy_k ,14430211.98+(np.random.randint(100)-50)*noisy_k]),
            np.array([444489.12-0.25+(np.random.randint(100)-50)*noisy_k,14430221.57+(np.random.randint(100)-50)*noisy_k]),
            np.array([444488.76-0.25+(np.random.randint(100)-50)*noisy_k,14430227.76+(np.random.randint(100)-50)*noisy_k]),
            np.array([444499.43+(np.random.randint(100)-50)*noisy_k,14430248.99-0.25+(np.random.randint(100)-50)*noisy_k]),
            np.array([444512.56+(np.random.randint(100)-50)*noisy_k,14430249.6+0.5+(np.random.randint(100)-50)*noisy_k])]
            ob = np.array(ob_tmp8+ob_tmp7+ob_tmp6+ob_tmp5+ob_tmp4+ob_tmp3+ob_tmp2+ob_tmp1)#ob_tmp8+ob_tmp7+ob_tmp6+ob_tmp5+ob_tmp4+ob_tmp3+ob_tmp2+
        elif not simulink_flag:
            #ob = np.array([[GPS_x[0]+1,GPS_y[0]+5],[GPS_x[0]-1,GPS_y[0]+10]])
            ob_tmp8 = ob_tmp7
            ob_tmp7 = ob_tmp6
            ob_tmp6 = ob_tmp5
            ob_tmp5 = ob_tmp4
            ob_tmp4 = ob_tmp3
            ob_tmp3 = ob_tmp2
            ob_tmp2 = ob_tmp1
            ob_tmp1 = oblist_to_world_cordination(obstacle_list,GPS_xc,GPS_yc,np.pi/2-GPS_yaw*np.pi/180)
            ob = np.array(ob_tmp8+ob_tmp7+ob_tmp6+ob_tmp5+ob_tmp4+ob_tmp3+ob_tmp2+ob_tmp1)
        ob = np.concatenate([ob.reshape([-1,2]),ob_lidar],axis=0)
        sleep(0.04)

def radar_thred():
    #毫米波雷达生成毫米波障碍物表线程
    global thred_flag
    global recv_radar
    while thred_flag:
        Lock_rader.acquire()
        new_haomibo()
        Lock_rader.release()
        sleep(0.04)

def GPS_record_thred():
    #路径采集线程
    global thred_flag
    while thred_flag:
        GPS_record()
        sleep(0.5)

def ser_receive():
    #RFID串口数据接收线程(暂不用)
    global thred_flag
    global data_2
    global label
    global ex_traffic_p_x
    global ex_traffic_p_y
    global GPS_xc
    global GPS_yc
    global GPS_yaw
    global ex_traffic_p_yaw
    global d_to_traffic_light_thresh
    global error_ser
    while thred_flag:
        d_to_traffic_light=np.sqrt((float(ex_traffic_p_x)-GPS_xc)**2+(float(ex_traffic_p_y)-GPS_yc)**2)
        #print('d_to_traffic_light%.2f'%d_to_traffic_light)
        data= str(binascii.b2a_hex(ser.read()))[2:-1]
        if data=='55':
            label_tmp=0
            data= str(binascii.b2a_hex(ser.read()))[2:-1]
            if data=='55':
                data_2= '55'+str(binascii.b2a_hex(ser.read(60)))[2:-1]
                print('data_2=') 
                print(data_2)
        if data_2[16]=='d' and d_to_traffic_light<d_to_traffic_light_thresh:
            label_tmp=1 #白色低电平
        else:
            if data_2[16]=='e'and d_to_traffic_light<d_to_traffic_light_thresh:
                label_tmp=3 #黄色低电平
            else:
                if data_2[16]=='f' and d_to_traffic_light<d_to_traffic_light_thresh:
                    label_tmp=2 #绿色低电平
                else:
                    label_tmp=0 #未知
        traffic_angle=cacu_angle_from_p(float(ex_traffic_p_x),float(ex_traffic_p_y),GPS_xc,GPS_yc)
        ex_traffic_p_yaw_tmp=float(ex_traffic_p_yaw)-GPS_yaw
        while ex_traffic_p_yaw_tmp<0:
            ex_traffic_p_yaw_tmp=ex_traffic_p_yaw_tmp+360
        while ex_traffic_p_yaw_tmp>360:
            ex_traffic_p_yaw_tmp=ex_traffic_p_yaw_tmp-360
        if  ((traffic_angle<90 and traffic_angle>0) or (traffic_angle>270 and traffic_angle<360)) and((ex_traffic_p_yaw_tmp<60 and ex_traffic_p_yaw_tmp>0) or (ex_traffic_p_yaw_tmp>300 and ex_traffic_p_yaw_tmp<360)):
            label =label_tmp
        else:
            label =0
        error_ser+=1
        sleep(0.1)

def AEB_thred():
    #AEB线程(暂不用)
    global thred_flag
    global ac_flag
    while 0:
        ac_flag=AEB()
        sleep(0.1)

def brake_thred():
    #信号灯识别制动信号生成线程，由（label，cam_flag，cam_area）改变brake_signal
    global thred_flag
    global ac_flag
    global cam_area
    global area_thresh
    global brake_signal
    global cam_flag
    global label
    global error_traffic
    global ex_traffic_signal
    while thred_flag:
        if (label==1 or (cam_flag==1 and cam_area>area_thresh)) and ex_traffic_signal ==1:
            brake_signal = 1
        else:
            brake_signal = 0
        error_traffic+=1
        sleep(0.02)

def diagnose():
    #故障诊断线程
    # 11000    故障码发送正常
    # 11001    转向失灵
    # 11002    制动失灵
    # 11003    油门失灵
    # 11005    毫米波故障
    # 11007    摄像头故障
    # 11008    控制系统故障
    # 11009    CAN通讯故障
    # 11010    数据采集故障
    # 11011    GPS数据缺失
    # 11012    速度数据缺失
    # 11013    串口通讯故障
    # 11014    控制指令发送故障
    global thred_flag
    global error_EPS
    global error_brake
    global error_ac
    global error_haomibo
    global error_camera
    global error_gps_move
    global error_can_init
    global error_AEB
    global error_capture
    global error_GPS_xy
    global error_GPS_ll
    global error_GPS_yaw
    global error_speed_miss
    global error_ser
    global error_can_send 
    global error_traffic
    global error_EPS_miss
    global fault_state
    global GPS_xc
    global GPS_yc
    global GPS_yaw
    global error_battery
    global rwheel_vel_100
    global error
    global delta
    while thred_flag:
        error_tmp='11000'
        rwheel_vel_100_tmp = rwheel_vel_100/100
        fault_state_tmp='GPS_x='+str(GPS_xc)+'\t'+'GPS_y='+str(GPS_yc)+'\t'+'GPS_yaw='+str(GPS_yaw)+'\t'+'v='+str(rwheel_vel_100_tmp)+'m/s'+'\t'+'delta='+str(delta)+'\n'
        str_k=0
        if error_EPS==0 or error_EPS_miss ==0:
            error_tmp=error_tmp+',11001'
            fault_state_tmp=fault_state_tmp+'转向失灵'
            str_k=str_k+1
            if str_k>3:
                str_k=0
                fault_state_tmp=fault_state_tmp+'\n'
            else:
                fault_state_tmp=fault_state_tmp+'\t'
            #print('转向失灵')
        if error_brake==0:
            error_tmp=error_tmp+',11002'
            fault_state_tmp=fault_state_tmp+'制动失灵'
            str_k=str_k+1
            if str_k>3:
                str_k=0
                fault_state_tmp=fault_state_tmp+'\n'
            else:
                fault_state_tmp=fault_state_tmp+'\t'
            #print('制动失灵')
        if error_ac==0:
            error_tmp=error_tmp+',11003'
            fault_state_tmp=fault_state_tmp+'油门失灵'
            str_k=str_k+1
            if str_k>3:
                str_k=0
                fault_state_tmp=fault_state_tmp+'\n'
            else:
                fault_state_tmp=fault_state_tmp+'\t'
            #print('油门失灵')
        if error_haomibo==0:
            error_tmp=error_tmp+',11005'
            fault_state_tmp=fault_state_tmp+'毫米波雷达故障'
            str_k=str_k+1
            if str_k>3:
                str_k=0
                fault_state_tmp=fault_state_tmp+'\n'
            else:
                fault_state_tmp=fault_state_tmp+'\t'
            #print('毫米波雷达故障')
        if error_camera==0:
            error_tmp=error_tmp+',11007'
            fault_state_tmp=fault_state_tmp+'摄像头故障'
            str_k=str_k+1
            if str_k>3:
                str_k=0
                fault_state_tmp=fault_state_tmp+'\n'
            else:
                fault_state_tmp=fault_state_tmp+'\t'
            #print('摄像头故障')
        if  error_AEB == 0:
            error_tmp=error_tmp+',11008'
            fault_state_tmp=fault_state_tmp+'AEB故障'
            str_k=str_k+1
            if str_k>3:
                str_k=0
                fault_state_tmp=fault_state_tmp+'\n'
            else:
                fault_state_tmp=fault_state_tmp+'\t'
            #print('AEB故障')
        if error_can_init==1:
            error_tmp=error_tmp+',11009'
            fault_state_tmp=fault_state_tmp+'CAN通讯故障'
            str_k=str_k+1
            if str_k>3:
                str_k=0
                fault_state_tmp=fault_state_tmp+'\n'
            else:
                fault_state_tmp=fault_state_tmp+'\t'
            #print('CAN通讯故障')
        if error_capture==0:
            error_tmp=error_tmp+',11010'
            fault_state_tmp=fault_state_tmp+'数据采集故障'
            str_k=str_k+1
            if str_k>3:
                str_k=0
                fault_state_tmp=fault_state_tmp+'\n'
            else:
                fault_state_tmp=fault_state_tmp+'\t'
            #print('数据采集故障')
        if error_GPS_xy==0 or error_GPS_ll == 0 or error_GPS_yaw == 0 or error_traffic == 0:
            fault_state_tmp=fault_state_tmp+'没有GPS信号'
            str_k=str_k+1
            if str_k>3:
                str_k=0
                fault_state_tmp=fault_state_tmp+'\n'
            else:
                fault_state_tmp=fault_state_tmp+'\t'
            print('没有GPS信号')
            error_tmp=error_tmp+',11011'
        if error_speed_miss==0:
            error_tmp=error_tmp+',11012'
            fault_state_tmp=fault_state_tmp+'没有速度数据'
            str_k=str_k+1
            if str_k>3:
                str_k=0
                fault_state_tmp=fault_state_tmp+'\n'
            else:
                fault_state_tmp=fault_state_tmp+'\t'
            #print('没有速度数据')
        if error_ser==0:#常故障
            error_tmp=error_tmp+',11013'
            fault_state_tmp=fault_state_tmp+'没有串口数据'
            str_k=str_k+1
            if str_k>3:
                str_k=0
                fault_state_tmp=fault_state_tmp+'\n'
            else:
                fault_state_tmp=fault_state_tmp+'\t'
            #print('没有串口数据')
        if error_can_send==0:
            error_tmp=error_tmp+',11014'
            fault_state_tmp=fault_state_tmp+'控制指令下发线程故障'
            str_k=str_k+1
            if str_k>3:
                str_k=0
                fault_state_tmp=fault_state_tmp+'\n'
            else:
                fault_state_tmp=fault_state_tmp+'\t'
            #print('控制指令下发线程故障')
        # if error_battery == 0:
            # fault_state_tmp=fault_state_tmp+'电池系统故障'
            # str_k=str_k+1
            # if str_k>3:
                # str_k=0
                # fault_state_tmp=fault_state_tmp+'\n'
            # else:
                # fault_state_tmp=fault_state_tmp+'\t'
        error =error_tmp
        fault_state=fault_state_tmp
        error_EPS_miss=0
        error_EPS =0
        error_brake =0
        error_ac = 0
        error_haomibo = 0
        error_camera =0
        error_gps_move = 0
        error_AEB = 0
        error_capture = 0
        error_GPS_xy = 0
        error_GPS_ll = 0
        error_GPS_yaw = 0
        error_speed_miss = 0
        error_ser = 0
        error_can_send = 0
        error_traffic = 0
        error_battery = 0
        #print('故障码:')
        #print(error)
        sleep(4)

def data_save():
    #云端数据上传线程，慢速
    global thred_flag
    global longitude
    global latitude
    global vehicle_xita
    global GPS_yaw
    global rwheel_vel_100
    global error
    global ex_car_number
    while thred_flag:
        vehicle_xita=int(GPS_yaw)
        rwheel_vel_100_to_pms_tmp=rwheel_vel_100/100*3.6
        rwheel_vel_100_to_pms=round(rwheel_vel_100_to_pms_tmp,2)
        now_time=np.zeros([6])
        now = datetime.now()
        now_time[0]=now.year
        now_time[1]=now.month
        now_time[2]=now.day
        now_time[3]=now.hour
        now_time[4]=now.minute
        now_time[5]=now.second
        now_time_total=int(now_time[5]+now_time[4]*100+now_time[3]*10000+now_time[2]*1000000+now_time[1]*100000000+now_time[0]*10000000000)
        longitude_tmp=round(longitude,6)
        latitude_tmp=round(latitude,6)
        #慢速上传，最快3s/次
        # 车辆编号,时间,经度,纬度,横摆角,车速,车辆故障
        dataUploadByHttpOne(ex_car_number, now_time_total, longitude_tmp, latitude_tmp, vehicle_xita, rwheel_vel_100_to_pms, error)
        #print (str(now.year)+ '/' + str(now.month) + '/' + str(now.day) ,str(now.hour) +':'+ str(now.minute) + ':' + str(now.second)+':data_save_fine')
        # print('~~~~~~~~~~~~~~~~')
        # print(ex_car_number)
        # print(now_time_total)
        # print(longitude_tmp)
        # print(latitude_tmp)
        # print(vehicle_xita)
        # print(rwheel_vel_100_to_KMs)
        # print(error)
        # print('~~~~~~~~~~~~~~~~')
        sleep(4)

def data_save_fast():
    #云端数据上传线程，快速
    global thred_flag
    global rwheel_vel_100
    global soc
    global error
    global ex_car_number
    global ex_v
    global cam_flag #未检测0红灯1绿灯2黄灯3
    global label #未检测0红灯1绿灯2黄灯3
    global speed
    global brake
    global EPS_angle
    global ex_turn_mid
    global state_direction
    global ac_pwm
    global brake_pwm
    while thred_flag:
        EPS_angle_tmp=0.0788*(EPS_angle-int(ex_turn_mid))-2.7215
        EPS_angle_tmp=round(EPS_angle_tmp,2)
        rwheel_vel_100_to_mps_tmp=rwheel_vel_100/100*3.6
        rwheel_vel_100_to_mps=round(rwheel_vel_100_to_mps_tmp,2)
        # 高速上传，最快1s/次
        # # 车辆编号，车速，电量，红绿灯状态，运行方向，底盘状态
        chassis_state=str(EPS_angle_tmp)+','+str(brake_pwm)+','+str(ac_pwm)
        if cam_flag==1 or label==1:
            cam_flag_tmp=1
        else:
            if cam_flag==2 or label==2:
                cam_flag_tmp=2
            else:
                if cam_flag==3 or label==3:
                    cam_flag_tmp=3
                else:
                    cam_flag_tmp=0
        dataUploadByHttpTwo(ex_car_number, rwheel_vel_100_to_mps, soc, cam_flag_tmp, state_direction, chassis_state)
        # print('~~~~~~~~~~~~~~~~')
        # print(ex_car_number)
        # print(rwheel_vel_100_to_KMs)
        # print(soc)
        # print(cam_flag)
        # print(state_direction)
        #print(chassis_state)
        # print('~~~~~~~~~~~~~~~~')
        sleep(1)
    # print('12退出 data_save_fast')

def communication_from_cloud():
    #云端服务器建立连接线程
    global thred_flag
    global URL_HEADER_WS
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(URL_HEADER_WS,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    while thred_flag:
        dataDownLoad(ws)
        sleep(1)

def camera_thred():
    #摄像头识别线程
    global thred_flag
    global cam_flag
    global cam_area
    global error_camera
    global cap
    while thred_flag:
        cam_area,cam_flag,thred_flag,error_camera=camera_recg(cap,error_camera,thred_flag)
        sleep(0.01)
#~~~~~~~~~~~~~~~~~线程设计结束by龚章鹏~~~~~~~~~~~~~~~~
        
#~~~~~~~~~~~~~~~~~~线程执行~~~~~~~~~~~~~~~~~

threads=[]
thred_flag=True
move_thred_flag=True
t = threading.Thread(target=capture_thred)
threads.append(t)
t = threading.Thread(target=AEB_thred)
threads.append(t)
# t = threading.Thread(target=data_save)
# threads.append(t)
# t = threading.Thread(target=data_save_fast)
# threads.append(t)
t = threading.Thread(target=can_send_thred)
threads.append(t)
t = threading.Thread(target=brake_thred)
threads.append(t)
t = threading.Thread(target=GPS_record_thred)
threads.append(t)
t = threading.Thread(target=send_can_to_micro_thred)
threads.append(t)
t = threading.Thread(target=diagnose)
threads.append(t)
if use_3D_lidar:
    t = threading.Thread(target=lidar_thred)
    threads.append(t)
elif use_2D_image:
    t = threading.Thread(target=camera_thred)
    threads.append(t)
t = threading.Thread(target=ridar_thred)
threads.append(t)
t = threading.Thread(target=radar_thred)
threads.append(t)
t = threading.Thread(target=send_can_tc_thred)
threads.append(t)
#t_ser = threading.Thread(target=ser_receive) #串口线程如果没有收到数，会保持几秒钟，因此关闭会比较慢
#t_ser.setDaemon(True)
t_cloud = threading.Thread(target=communication_from_cloud)
t_cloud.setDaemon(True)
t_ui = threading.Thread(target=initUi)
initConfig()
for tt in threads:
    tt.isRunning = True
    tt.start()
print('start threading')
t_ui.isRunning = True
t_ui.start()
#t_ser.isRunning = True
#t_ser.start()
t_cloud.isRunning = True
t_cloud.start()

while True:
    if ex_flag == 1 :
        #点击启动
        d_tmp_last=10000  
        GPS_move_flag = 0
        ex_GPS_direction_flag = 3
        frenet_init_flag = False
        t_move = threading.Thread(target=gps_move_thred)
        move_thred_flag = True
        t_move.isRunning = True
        t_move.start()
        print('start move')
        ex_flag=0
        
    if ex_leaveout_flag==1:
        #点击关闭界面
        thred_flag = False
        break
    if ex_flag == 2:
        #点击停止
        move_thred_flag = False
        t_move.isRunning = False
        t_move.join()
        print('stop move')
        can_send(send_accelerateID,0,0,0,0,0,0,0,0)
        can_send(send_brakeID,0,0,0,0,0,0,0,0)
        can_send(512,0,0,0,0,0,0,0,0)
        can_send(send_turnID,0,0,0,0,0,1,109,96)
        ex_flag=0
    if show_animation and frenet_init_flag == True and path != None and move_thred_flag == True:
        #可视化，各别主机上开启会使程序卡死
        plt.clf()
        plt.axis("equal")
        plt.grid(True)
        plt.plot(GPS_x, GPS_y)
        if simulink_flag:
            GPS_xcc = state.x
            GPS_ycc = state.y
            GPS_yawc = state.yaw
        else:
            GPS_xcc = GPS_xc+0.8*math.cos(np.pi/2-GPS_yaw*np.pi/180)
            GPS_ycc = GPS_yc+0.8*math.sin(np.pi/2-GPS_yaw*np.pi/180)
            GPS_yawc = np.pi/2-GPS_yaw*np.pi/180
        try:
            plt.plot(ob_new[:, 0], ob_new[:, 1], "ob")
        except:
            print('no obstacal')
        try:
            plt.plot(path.x[1:], path.y[1:], "-or")
            ru_x,ru_y,rd_x,rd_y,lu_x,lu_y,ld_x,ld_y = generate_box_four_points(GPS_xcc,GPS_ycc,GPS_yawc)
            plt.plot(x_set,y_set,"-g")
            plt.plot(ru_x, ru_y, "og")
            plt.plot(rd_x, rd_y, "og")
            plt.plot(lu_x, lu_y, "og")
            plt.plot(ld_x, ld_y, "og") 
            plt.plot(path.x[1], path.y[1], "vc")
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("c[m]:" + str(path.c[0])[0:6]+" v[m/s]:" + str(state.v)[0:6] + "  delta[rad]:" + str(delta)[0:6]) #delta右转为负
            plt.pause(0.001)
        except:
            print('no path')
    sleep(0.5)

#程序结束关闭剩余线程
thred_flag=False
move_thred_flag = False
for tt in threads:
    tt.isRunning = False
    tt.join()
if ex_flag == 0:
    t_move.isRunning = False
    t_move.join()
if use_2D_image:
    cap.release()
cv2.destroyAllWindows
can_send(512,0,0,0,0,0,0,0,0)
can_send(send_accelerateID,0,0,0,0,0,0,0,0)
can_send(send_brakeID,0,0,0,0,0,0,0,0)
sleep(2)
can_send(send_turnID,0,0,0,0,0,1,109,96)
#数据保存
def data_process_save(x_set,y_set,yaw_set,c_speed_set,delta_set,error_set,error_set2,error_set3,dc_ob_set,df_ob_set,dr_ob_set,time_record_set,time_dur_set,filename):
    data_list = []
    data_num=len(x_set)
    for i in range(data_num):
        data_list.append([x_set[i],y_set[i],yaw_set[i],c_speed_set[i],delta_set[i],error_set[i],error_set2[i],error_set3[i],dc_ob_set[i],df_ob_set[i],dr_ob_set[i],time_record_set[i],time_dur_set[i]])
    np.save(filename,data_list)
timestr = time.strftime("%Y%m%d_%H%M%S")
data_file_name = './data_process_save' + timestr+'.npy'
ob_file_name = './ob'+timestr+'.data'
path_file_name = './path'+timestr+'.data'
gps_x_file_name = './gps_x'+timestr+'.data'
gps_y_file_name = './gps_y'+timestr+'.data'
data_process_save(x_set,y_set,yaw_set,c_speed_set,delta_set,error_set,error_set2,error_set3,dc_ob_set,df_ob_set,dr_ob_set,time_record_set,time_dur_set,data_file_name)
with open(ob_file_name,'wb') as filehandle:
    pickle.dump(ob_set,filehandle)
with open(path_file_name,'wb') as filehandle:
    pickle.dump(path_set,filehandle)
with open(gps_x_file_name,'wb') as filehandle:
    pickle.dump(GPS_x,filehandle)
with open(gps_y_file_name,'wb') as filehandle:
    pickle.dump(GPS_y,filehandle)
#关闭界面线程
t_ui.isRunning = False
t_ui.join()
#关闭激光雷达
if use_3D_lidar:
    get_stop()
time.sleep(1)
print('finish')

