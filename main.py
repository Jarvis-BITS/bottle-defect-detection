import warnings
warnings.filterwarnings("ignore")

import cv2
import pixellib
from pixellib.instance import instance_segmentation
import datetime, time
import matplotlib.pyplot
import numpy as np
import pandas as pd
from IPython.display import Image, display
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import os, sys


from flask import Flask, render_template, Response, request
from threading import Thread


global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

segment_image = instance_segmentation(infer_speed = "rapid")
segment_image.load_model("E:\pwl-model\Tester\mask_rcnn_coco.h5")
target_classes = segment_image.select_target_classes(bottle = True)

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')

vid = cv2.VideoCapture(0)

index=["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('E:\pwl-model\Tester\colors.csv', names=index, header=None)

font = cv2.FONT_HERSHEY_SIMPLEX

def recognize_color(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname

CATEGORIES = ["Glass", "Plastic"]

def prepare(filepath):
    IMG_SIZE = 100  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.

model = tf.keras.models.load_model("E:\pwl-model\Tester\\64x3-CNN_material.model")
# new_model = load_model('E:\pwl-model\Jarvis_defect.h5')
new_model = load_model('E:\pwl-model\Tester\model_tacc8275_val8493.h5')
def defect_predictor_writer(prediction,input_image):
	font = cv2.FONT_HERSHEY_SIMPLEX
	if prediction == 0:
		cv2.putText(input_image,'NORMAL',(frame_size-225,frame_size-50), font, 1.5,(255,255,255),3)
	elif prediction == 1:
		cv2.putText(input_image,'DEFECT',(frame_size-225,frame_size-50), font, 1.5,(0,0,255),3)
	else:
		cv2.putText(input_image,'No Bottle',(frame_size-240,frame_size-50), font, 1.5,(0,0,255),3)
	return input_image

def material_predictor_writer(prediction,input_image):
	font = cv2.FONT_HERSHEY_SIMPLEX
	if prediction == 0:
		cv2.putText(input_image,'Glass',(frame_size-225,frame_size-20), font, 0.7,(255,255,255),2)
	elif prediction == 1:
		cv2.putText(input_image,'Plastic',(frame_size-225,frame_size-20), font, 0.7,(255,255,255),2)
	else:
		cv2.putText(input_image,'UNKNOWN',(frame_size-225,frame_size-20), font, 0.7,(0,0,255),2)
	return input_image

def defect_predictor(file_name,image_input):

	img_path = os.path.join(os.getcwd(), file_name)
	#display(Image(filename=img_path))

	path = img_path
	img = image.load_img(path, target_size=(150, 150))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)

	images = np.vstack([x])
	classes = new_model.predict(images, batch_size=10)
	#print(classes[0])
	if classes[0]<0.5:
	    image_input = defect_predictor_writer(0,image_input)
	else:
	    image_input = defect_predictor_writer(1,image_input)

	return image_input

def gen_frames():  # generate frame by frame from camera
	global out, capture,rec_frame		

	while(True):
		
		# Capture the video frame
		# by frame
		ret, frame = vid.read()
		if ret:

			#to resize the image
			frame_size = 800
			frame = cv2.resize(frame, (frame_size,frame_size))

			#to save the frame to imput for object extraction
			cv2.imwrite('saved_frame.jpg',frame)
	
			#To semantically segment target object in the frame and apply bounding box around it for viewing
			result=segment_image.segmentFrame(frame, segment_target_classes = target_classes, show_bboxes = "true")
			disp = result[1]

			#Object extraction from saved image (depends on whether further models need them or not)
			segment_image.segmentImage("saved_frame.jpg", segment_target_classes = target_classes, extract_segmented_objects= True, save_extracted_objects=True)



			#Cropping image to feed to material check model (test)
			#cropped_image = img[(int(img.shape[0]/2)-50):(int(img.shape[0]/2)+50), (int(img.shape[1]/2)-50):(int(img.shape[1]/2)+50)]
			#cv2.imwrite("cropped.jpg", cropped_image)

			try:
				img = cv2.imread("segmented_object_1.jpg")
				h, w, c = img.shape

				ar = h/w

				h=int(h/2)
				w=int(w/2)

				b,g,r = img[h,w]
				b = int(b)
				g = int(g)
				r = int(r)

				prediction = model.predict([prepare('segmented_object_1.jpg')])

			except Exception:
				pass


			#greyscale
			#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			disp = cv2.rectangle(disp, (0,frame_size-100), (frame_size,frame_size), (0,0,0), -1)

			try:
				text = 'Colour: ' + recognize_color(r,g,b)
				text2 = 'Ht. to Wd. ratio: ' + str(round(ar,2))
				cv2.putText(disp,text,(20,frame_size-30), font, 1,(255,255,255),2)
				cv2.putText(disp,text2,(20,frame_size-70), font, 1,(255,255,255),2)
			except Exception:
				cv2.putText(disp,'No Colour',(20,frame_size-30), font, 1,(255,255,255),2)
				cv2.putText(disp,'No Aspect Ratio',(20,frame_size-70), font, 1,(255,255,255),2)
	
			try:
				disp = material_predictor_writer(prediction[0][0],disp)
			except Exception:
				disp = material_predictor_writer(2,disp)
		

			if(os.path.isfile('segmented_object_1.jpg')==True):
				disp = defect_predictor('segmented_object_1.jpg',disp)
			else:
				disp = defect_predictor_writer(2,disp)

		#if int(prediction[0][0]) == 1:
			#cv2.putText(image,"Plastic",(20,390), font, 1,(255,255,255),2)
		#else:
			#cv2.putText(image,"Metal",(20,420), font, 1,(255,255,255),2)

		# Display the resulting frame
		#cv2.imshow('image segmentation', disp)
	
		#time.sleep(10)

		# the 'q' button is set as the
		# quitting button you may use any
		# desired button of your choice
		# if cv2.waitKey(1) & 0xFF == ord('q'):
		# 	try:		
		# 		os.remove("saved_frame.jpg")
		# 		os.remove("segmented_object_1.jpg")
		# 	except Exception:
		# 		pass
		# 	break
			try:
				ret, buffer = cv2.imencode('.jpg', cv2.flip(disp,1))
				disp = buffer.tobytes()
				yield (b'--frame\r\n'
					b'Content-Type: image/jpeg\r\n\r\n' + disp + b'\r\n')

			except Exception as e:
				pass

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
	global switch,vid
	if request.method == 'POST':
		if request.form.get('stop') == 'Stop/Start':
			
			if(switch==1):
				switch=0
				vid.release()
				cv2.destroyAllWindows()
			else:
				vid = cv2.VideoCapture(0)
				switch=1


	elif request.method=='GET':
		return render_template('index.html')
	return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


