# Bottle-Defect-Detection
A deep learning model which is able to identify bottles, predict their material and find out whether they are defective(crushed, cut etc.) or not
This work was done during the course of BITS Pilani PS-1 by team comprising of Ishaan, Jishnu, Javin & Shivank as a project for Plastic Water Labs Prvt. Ltd.

<img src="https://github.com/Jarvis-BITS/bottle-defect-detection/blob/main/Sample-pictures/sample_gif.gif" height=250px align=center>
<img src="https://github.com/Jarvis-BITS/bottle-defect-detection/blob/main/Sample-pictures/bottle_detection.PNG" height=250px align=left><img src="https://github.com/Jarvis-BITS/bottle-defect-detection/blob/main/Sample-pictures/Color_detection.png" height=250px align=right><img src="https://github.com/Jarvis-BITS/bottle-defect-detection/blob/main/Sample-pictures/Material_detection.PNG" height=250px align=left><img src="https://github.com/Jarvis-BITS/bottle-defect-detection/blob/main/Sample-pictures/defect_detection.PNG" height=250px>

```
server_model.py 
``` 
can be run when you dont want to run the web-app and simply see how bottles are being segmented
You must use pip-install to install all necessary dependencies and libraries. We recommend using virtualenv to store all pip-installs 
<br> <br>2nd_defect_detect.model as the names suggests is a secondary CNN model with a higher accuracy of around 94%, however the frame numpy array from the video needs to be resized in order to fit its custom mask layers. 
