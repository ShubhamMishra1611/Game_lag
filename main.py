import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import savgol_filter
from scipy.signal import butter, lfilter
from scipy.signal import hilbert


class game_lag:

    guass_kernel=(5,5)
    screenx1y1=(504,0)
    screenx2y2=(1812,760)
    rectanglex1y1=(680,850)
    max_lag=500
    
    rectanglex2y2=(1610,1050)
    order=5
    fs=60
    cutoff=3.667


    def __init__(self,video_path,frame_rate):
        print("Initializing...")
        if not os.path.exists(video_path):
            print("Video file does not exist!")
            sys.exit()
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.y,self.x,self.inputgiven=self.get_frame()
        self.inputgiven.pop()
        self.y_t=self.process()
        #------------------------------------
        plt.plot(self.x,self.y_t)
        plt.plot(self.x,self.inputgiven)
        plt.title("Input given")
        plt.show()
        #------------------------------------
        coor_mat=[]
        for i in range(self.max_lag):
            input_given_shifted=self.shift(self.inputgiven,i)
            coor_mat.append((i,self.coorelation(self.y_t,input_given_shifted)))
        max_coor=max(coor_mat,key=lambda x:x[1])
        print("Lag is: ",max_coor[0]/self.frame_rate," seconds")
        
    def get_frame(self):
        y=[]
        input_given=[]
        print("Getting frames...")
        cap = cv2.VideoCapture(self.video_path)
        frame_count =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total frames: ",frame_count)
        frame1=None
        for i in range(frame_count):
            ret,frame2=cap.read()
            if ret==False:
                print("---------------------------------------------------")
                print("Error reading frame!")
                print("---------------------------------------------------")
                continue
            if self.input_Given(frame2):
                input_given.append(1)
            else:
                input_given.append(0)
            if i==0:
                frame1=frame2
                continue
            frame1gray=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
            frame2gray=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            frame1guass=cv2.GaussianBlur(frame1gray,self.guass_kernel,0)
            frame2guass=cv2.GaussianBlur(frame2gray,self.guass_kernel,0)
            screen1=frame1guass[self.screenx1y1[1]:self.screenx2y2[1],self.screenx1y1[0]:self.screenx2y2[0]]
            screen2=frame2guass[self.screenx1y1[1]:self.screenx2y2[1],self.screenx1y1[0]:self.screenx2y2[0]]
            distance=np.sum(np.abs(screen1-screen2))
            y.append(distance)
            frame1=frame2
        x=np.arange(len(y))
        y=np.array(y)
        y=y/np.max(y)
        return y,x,input_given
    
    def process(self):
        print("Processing...")
        y=self.y
        x=self.x
        y=savgol_filter(y, 51, 3)
        #------------------------------------
        plt.plot(x,y)
        plt.title("Savgol filter")
        plt.show()
        #------------------------------------
        b, a = self.butter_lowpass(self.cutoff, self.fs, self.order)
        y_l = self.butter_lowpass_filter(y, self.cutoff, self.fs, self.order)
        #------------------------------------
        plt.plot(x,y_l)
        plt.title("Butterworth filter")
        plt.show()
        #------------------------------------
        analytic_signal = hilbert(y_l)
        amplitude_envelope = np.abs(analytic_signal)
        #------------------------------------
        plt.plot(x,amplitude_envelope)
        plt.title("Amplitude envelope")
        plt.show()
        #------------------------------------
        threshold=amplitude_envelope.mean()
        y_t=np.zeros(len(amplitude_envelope))
        for i in range(len(amplitude_envelope)):
            if amplitude_envelope[i]>threshold:
                y_t[i]=1
        #------------------------------------
        plt.plot(x,y_t)
        plt.title("Thresholding")
        plt.show()
        #------------------------------------
        return y_t


    def butter_lowpass(self,cutoff,fs,order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_lowpass_filter(self,data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    def input_Given(self,y):
        y=y[self.rectanglex1y1[1]:self.rectanglex2y2[1],self.rectanglex1y1[0]:self.rectanglex2y2[0]]
        y=cv2.cvtColor(y,cv2.COLOR_BGR2RGB)
        if np.any(y[:,:,2]>200):
            return True
        return False
    
    def coorelation(self,y1,y2):
        res=y1*y2
        return np.sum(res)
    
    def shift(self,x,n):
        if n==0:
            return x
        l=[0]*n
        l.extend(x[:-n])
        return l


if __name__=="__main__":
    video_path="videos\Crouch.MP4"
    frame_rate=240
    game_lag(video_path,frame_rate)


            
