import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import List
import os
from csv import writer
from tqdm import tqdm
"""
import game_lag
gl=game_lag(0) # 0 to work using the webcam
               # 1 to work using the video

g1.get_data(name) # file will be stored 
"""


class game_lag:

    screenx1y1=(504,0)
    screenx2y2=(1812,760)
    __relu=lambda x: np.maximum(0,x)
    frame_rate=239
    time_lag=[]


    def __init__(self,type_opr:int,path=None) -> None:
        print("Class initialized")
        self.path=path
        if type_opr==0 or (path!=None and os.path.isfile(path)) :
            self.__process(type_opr)
        else:
            if type_opr!=1 or type_opr!=0:
                raise "Invalid operation given."
            else:
                raise "File not found!!!"
    
    def __process(self,type_opr:int):
        print("Start Processing")
        camera=cv2.VideoCapture(self.path) if type_opr==1 else cv2.VideoCapture(1)
        frame1,frame2=None,None
        y=[]
        x=[]
        input_list=[]
        try:
            print("extracting frame...")
            ret,frame2=camera.read()
            for i in tqdm(range(10000)):
                print(i,end="\r")
                if not ret:
                    break
                input_list.append(True) if game_lag.__input_given(frame2) else input_list.append(False)
                if  frame1 is not None:
                    screen1=frame1[self.screenx1y1[1]:self.screenx2y2[1],self.screenx1y1[0]:self.screenx2y2[0]]
                    screen2=frame2[self.screenx1y1[1]:self.screenx2y2[1],self.screenx1y1[0]:self.screenx2y2[0]]
                    distance=np.sum(np.abs(screen1-screen2))
                    y.append(distance)
                else:
                    frame1=frame2
            print("frame extracted")
            x=list(range(len(y)))
            y=np.array(y)/max(y)
            print("Applying filter and smoothing")
            yhat=savgol_filter(y,51,3)
            new_y_hat=yhat-np.mean(yhat)
            new_y_hat=new_y_hat[50:2333]
            new_y_hat=game_lag.__smooth(game_lag.__relu(game_lag.__relu(new_y_hat)-np.mean(game_lag.__relu(new_y_hat))),0.95)#Smoothing using exponential moving average
            x=x[50:2333]
            print("Filter and smoothing applied")
            self.save_to_json(x,new_y_hat)

            for i in range(len(new_y_hat)):
                if new_y_hat[i]>0 and input_list[i] == True:
                    j=i
                    try:
                        while input_list[j]:
                            j-=1
                    except:
                        j=0
                    sel f.time_lag.append(abs(i-j)*self.frame_rate)
                    if self.time_lag>240:
                        continue
                    List=[j,i,self.time_lag]
                    with open('action.csv','a') as f:
                        writer_obj=writer(f)
                        writer_obj.writerow(List)
                        f.close()
            print("done with all data...")
            average=sum(self.time_lag)/len(self.time_lag)
            print(f"Average latency in this: {average}")

        except Exception as e:
            print("Oops Something went wrong :(")
            print(e)

    def __smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)                        # Save it
            last = smoothed_val                                  # Anchor the last smoothed value
            
        return smoothed


    def save_to_json(self,x:list,y:np.array)-> None:
        with open('y.json','w') as f:
            json.dump(y,f)
        with open('x.json','w') as f:
            json.dump(x,f)
        print("Data saved")

    def __input_given(frame):
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        try:
            x_box=cv2.imread("x_box.jpg")
            w, h = x_box.shape[::-1]
            res = cv2.matchTemplate(frame,x_box,cv2.TM_CCOEFF_NORMED)
            threshold=0.8
            loc=np.where(res>=threshold)
            pt=loc[::-1]
            x_box=frame[pt[0]:pt[0]+w,pt[1]:pt[1]+h]
        except:
            rectanglex1y1=(680,850)
            rectanglex2y2=(1610,1050)
            x_box=frame[rectanglex1y1[1]:rectanglex2y2[1],rectanglex1y1[0]:rectanglex2y2[0]]

        if np.any(x_box[:,:,2]>200):
            return True
        return False


if __name__=="__main__":
    g=game_lag(1,"videos\Crouch.MP4")

print("\a")



