# Bottle-Defect-Detection
A deep learning model which is able to identify bottles, predict their material and find out whether they are defective(crushed, cut etc.) or not
This work was done during the course of BITS Pilani PS-1 by team comprising of Ishaan, Jishnu, Javin & Shivank as a project for Plastic Water Labs Prvt. Ltd.

<img src="https://github.com/Jarvis-BITS/bottle-defect-detection/blob/main/Sample-pictures/sample_gif.gif" height=250px align=center>
<img src="https://github.com/Jarvis-BITS/bottle-defect-detection/blob/main/Sample-pictures/bottle_detection.PNG" height=250px><img src="https://github.com/Jarvis-BITS/bottle-defect-detection/blob/main/Sample-pictures/Color_detection.png" height=250px><img src="https://github.com/Jarvis-BITS/bottle-defect-detection/blob/main/Sample-pictures/Material_detection.PNG" height=250px><img src="https://github.com/Jarvis-BITS/bottle-defect-detection/blob/main/Sample-pictures/defect_detection.PNG" height=250px>

```
server_model.py 
``` 
can be run when you dont want to run the web-app and simply see how bottles are being segmented
You must use pip-install to install all necessary dependencies and libraries. We recommend using virtualenv to store all pip-installs 
<br> <br>2nd_defect_detect.model as the names suggests is a secondary CNN model with a higher accuracy of around 94%, however the frame numpy array from the video needs to be resized in order to fit its custom mask layers. 

Project description: Created a dataset of 1000 images of normal & defective water bottles using Kaggle & image augmentation techniques. We fine tuned a transfer learning Mask-RCNN model for object detection & pixellib for image segmentation. We then created a CNN to classify the segmented water bottle image as normal/defective (as well as type of defect i.e Scratch or dent) with a 87.7% accuracy. A second CNN was created to detect the material of bottle (plastic or glass) with a 72% accuracy. The different models weights were saved as .h5 file using pickle & integrated into a single deep-learning model. Finally a web-app was created for cloud deployment of the model was created using Flask.
