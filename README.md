# Bottle-Defect-Detection
A deep learning model which is able to identify bottles, predict their material and find out whether they are defective(crushed, cut etc.) or not
This work was done during the course of BITS Pilani PS-1 by team comprising of Ishaan, Jishnu, Javin & Shivank as a project for Plastic Water Labs Prvt. Ltd.
```
server_model.py 
``` 
can be run when you dont want to run the web-app and simply see how bottles are being segmented
You must use pip-install to install all necessary dependencies and libraries. We recommend using virtualenv to store all pip-installs 
2nd-defect.model as the names suggests is a secondary CNN model with a higher accuracy of around 94%, however the frame numpy array from the video needs to be resized in order to fit its custom mask layers. 
