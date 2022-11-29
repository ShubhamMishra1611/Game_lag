import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import savgol_filter
from scipy.signal import butter, lfilter
from scipy.signal import hilbert


class game_lag:

    guass_kernel = (5, 5)
    screenx1y1 = (504, 0)
    screenx2y2 = (1812, 760)
    rectanglex1y1 = (680, 850)
    rectanglex2y2 = (1610, 1050)
    MAXLAG = 500

    order = 5
    fs = 60
    cutoff = 3.667
    latency = []

    def __init__(self, video_path: str, frame_rate: int,visual:bool = False) -> None:
        '''
        video_path: path to the video file
        frame_rate: frame rate of the video
        visual: if True, plots will be shown.Default value is False

        For Example:
        game_lag("video.mp4",60,visual=True)
        '''
        print("Initializing...")
        self.visual=visual
        if not os.path.exists(video_path):
            print("Video file does not exist!")
            sys.exit()
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.y, self.x, self.inputgiven = self.get_frame()
        self.inputgiven.pop()
        self.y_t = self.process()
        # ------------------------------------
        if self.visual:
            plt.plot(self.x, self.y_t)
            plt.plot(self.x, self.inputgiven)
            plt.title("Input given")
            plt.show()
        # ------------------------------------
        i = 0
        while i < len(self.inputgiven):
            if self.inputgiven[i] == 1:
                length = 0
                for j in range(i, len(self.inputgiven)):
                    if self.inputgiven[j] == 1:
                        length += 1
                    else:
                        break
                lag = []
                for k in range(self.MAXLAG):
                    input_shifted = self.shift(
                        self.inputgiven[i:i+self.MAXLAG], k)
                    lag.append((k, self.coorelation(
                        self.y_t[i:i+self.MAXLAG], input_shifted)))
                max_latency = max(lag, key=lambda item: item[1])
                print("Action performed at frame: ", i,end=" ")
                print("Latency: ", max_latency[0]/self.frame_rate, "s",end="\n")
                self.latency.append(max_latency)
                with open("output_ADS_HG.csv", "a") as f:
                    f.write(str(i)+","+str(i+max_latency[0])+","+str(max_latency[0]/self.frame_rate)+"\n")
                i += length
            i += 1
        print("Done!")

    def get_frame(self):
        y = []
        input_given = []
        print("Getting frames...")
        cap = cv2.VideoCapture(self.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total frames: ", frame_count)
        frame1 = None
        screen1=None
        screen2=None
        for i in range(frame_count):
            ret, frame2 = cap.read()
            print("Frame: ", i,end="\r")
            if ret == False:
                print("---------------------------------------------------")
                print("Error reading frame!")
                print("---------------------------------------------------")
                continue
            box=frame2[self.rectanglex1y1[1]:self.rectanglex2y2[1],
              self.rectanglex1y1[0]:self.rectanglex2y2[0]]
            input_given.append(1 if np.any(box[:, :, 0] > 200) else 0)
            if i == 0:
                screen2=frame2[self.screenx1y1[1]:self.screenx2y2[1],self.screenx1y1[0]:self.screenx2y2[0]]
                screen_gray=cv2.cvtColor(screen2,cv2.COLOR_BGR2GRAY)
                screen_blur=cv2.GaussianBlur(screen_gray,self.guass_kernel,0)
                screen1=screen_blur
                continue
            screen2=frame2[self.screenx1y1[1]:self.screenx2y2[1],self.screenx1y1[0]:self.screenx2y2[0]]
            screen_gray=cv2.cvtColor(screen2,cv2.COLOR_BGR2GRAY)
            screen_blur=cv2.GaussianBlur(screen_gray,self.guass_kernel,0)
            distance=np.sum(np.abs(screen1-screen_blur))
            y.append(distance)
            screen1=screen_blur
        x = np.arange(len(y))
        y = np.array(y)
        y = y/np.max(y)
        return y, x, input_given

    def process(self):
        print("Processing...")
        y = self.y
        x = self.x
        y = savgol_filter(y, 51, 3)
        # ------------------------------------
        if self.visual:
            plt.plot(x, y)
            plt.title("Savgol filter")
            plt.show()
        # ------------------------------------
        b, a = self.butter_lowpass(self.cutoff, self.fs, self.order)
        y_l = self.butter_lowpass_filter(y, self.cutoff, self.fs, self.order)
        # ------------------------------------
        if self.visual:
            plt.plot(x, y_l)
            plt.title("Butterworth filter")
            plt.show()
        # ------------------------------------
        analytic_signal = hilbert(y_l)
        amplitude_envelope = np.abs(analytic_signal)
        # ------------------------------------
        if self.visual:
            plt.plot(x, amplitude_envelope)
            plt.title("Amplitude envelope")
            plt.show()
        # ------------------------------------
        threshold = amplitude_envelope.mean()
        y_t = np.zeros(len(amplitude_envelope))
        for i in range(len(amplitude_envelope)):
            if amplitude_envelope[i] > threshold:
                y_t[i] = 1
        # ------------------------------------
        if self.visual:
            plt.plot(x, y_t)
            plt.title("Thresholding")
            plt.show()
        # ------------------------------------
        return y_t

    def butter_lowpass(self, cutoff: float, fs: float, order: int = 5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def coorelation(self, y1, y2):
        '''
        Gets the coorelation in y1 and y2 with the below formula.
        summation(y1[i]*y2[i]) for i goes from 0 to len of y1.
        '''
        if len(y1) != len(y2):
            if len(y1) < len(y2):
                while len(y1) != len(y2):
                    y2 = y2[:-1]
            else:
                while len(y1) != len(y2):
                    y1 = y1[:-1]
        res = y1*y2
        return np.sum(res)

    def shift(self, x: np.array, n: int):
        '''
        This shift the given array x to n index.
        '''
        if n == 0:
            return x
        l = [0]*n
        l.extend(x[:-n])
        return l


if __name__ == "__main__":
    video_path = "videos\ADS HG.MP4"  # Path to video
    frame_rate = 240  # Frame rate of video
    game_lag(video_path, frame_rate)  # Initialize class with video path and frame rate
