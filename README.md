# Game_lag

# Results
## Results are uploaded in [test_final.ipynb](https://github.com/ShubhamMishra1611/Game_lag/blob/main/test_final.ipynb) on the data provided from same site.
## Requirements

* Python 3.6
* OpenCV
* Numpy
* Matplotlib
* Scipy

## Usage

* Run the [main.py](https://github.com/ShubhamMishra1611/Game_lag/blob/main/main.py) file
* Enter the video path and frame rate
* The code will run and display the graphs and latency in the terminal
* Or import the game_lag class from main.py create instance of the class providing video path and frame rate as argument.For example
```python 
from main import game_lag
video_path=video.mp4                              #Your video path.
frame_rate=240                                    #Frame rate of video file.
game = game_lag(video_path, frame_rate)
```
Also this is specific to given set of videos provided in the dataset but one can change the screen location and ben heck box in the code and it will work for that video.
### For example:
You can change these attributes in main.py file.
```python
class game_lag:

    guass_kernel=(5,5)
    screenx1y1=(504,0)                             #here you can change the screen top left and bottom right corner
    screenx2y2=(1812,760)
    rectanglex1y1=(680,850)                        #here you can change the ben heck top left and bottom right corner
    rectanglex2y2=(1610,1050)
    MAXLAG=500
    
    order=5
    fs=60
    cutoff=3.667
    latency=[]
```
As an output it save the latency for each input event in a csv (here [output.py](https://github.com/ShubhamMishra1611/Game_lag/blob/main/output.csv)) file and displays the average latency in the game in terminal.
It also shows different graphs being processed in the code to get a better insight.
These graphs are:
1. Input event vs Frame number
2. Savgol filter applied on input event vs Frame number
3. Butterworth filter applied on input event vs Frame number
4. Amplitude envelope of the input event vs Frame number
5. Thresholding of the amplitude envelope vs Frame number

## Error
If you find your output coming out as 

'''---------------------------------------------------</br>
Error reading frame!</br>
'''---------------------------------------------------</br>
'''---------------------------------------------------</br>
Error reading frame!</br>
'''---------------------------------------------------</br>


This might come out if some frame of the videos are no readble are being returned as NoneType.
However,this is not issue,the code with continue reading file and will give output.
