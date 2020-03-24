#!/usr/bin/env python3

import cv2
from keras.models import load_model
import joblib
import keras
import numpy as np
import base64
from model import ReturnModel
#from skimage.transform import resize


##################################################################################################################
########################### Vars for IMU prediction ##############################################################
##################################################################################################################

n_features=3
look_back=3
temp=[]
count=0

threshold = 0.1
scaler_filename = "models_imu/scaler_for_linear_acceleration"
model_filename="models_imu/LSTM_AutoEncoder_3_steps.hdf5"
scaler = joblib.load(scaler_filename)
model=load_model(model_filename)


##################################################################################################################
########################### Vars for Image prediction ############################################################
##################################################################################################################

image_temp = []
image_count = 0
image_name_count = 0
imu_name_count = 0
image_model_encoder="models/tanh_en_2_24_en.h5"
image_model_decoder = "models/tanh_de_2_24_en.h5"
image_medel_seq2seq = "models/tanh_seq2seq.h5"

#image_model_generator = "/home/vinu/GENERATOR.h5"

threshold_image = 0.026354463578475717

image_model = ReturnModel()
#image_model.summary()

print ("loading complete")

def ImuCallback(L,imu_hed):
	global temp
	global count
	global imu_name_count



	L = np.array([L])
	L = scaler.transform(L)
	
	if(count<=2):

		temp.append(L)

		if(count==2):
			
			temp = np.array(temp).reshape(1,look_back,n_features)

			pred = model.predict(temp)
			error = np.mean(np.mean(np.abs(pred-temp),axis=1),axis=0)
			
			if(error[0]>threshold):
				fo = open("output.txt","a")
				fo.write("ANOMALY," + str(imu_hed['stamp']['secs'] )+ "," + str(imu_hed['stamp']['secs'] ) + "," + str(imu_hed['seq']) + "\n")
				fo.close()
				print("IMU Data ->  secs: {} , nsecs: {} , seq: {} ,  Status: ANOMALY".format(str(imu_hed['stamp']['secs'] ),str(imu_hed['stamp']['secs'] ),str(imu_hed['seq'])))
				imu_name_count+=1
			else:
				fo = open("output.txt","a")
				fo.write("NORMAL," + str(imu_hed['stamp']['secs'] )+ "," + str(imu_hed['stamp']['secs'] ) + "," + str(imu_hed['seq']) + "\n")
				fo.close()
				print("IMU Data ->  secs: {} , nsecs: {} , seq: {} ,  Status: NORMAL".format(str(imu_hed['stamp']['secs'] ),str(imu_hed['stamp']['secs'] ),str(imu_hed['seq'])))
				imu_name_count+=1
		count+=1

	if(count==3):
		
		temp = np.append(temp[0][1:],L,axis = 0)
		temp = np.array([temp])
		temp=temp.reshape(1,look_back,n_features)
		
		pred=model.predict(temp)
		error = np.mean(np.mean(np.abs(pred-temp),axis=1),axis=0)
		    
		if(error[0]>threshold):
			fo = open("output.txt","a")
			fo.write("ANOMALY," + str(imu_hed['stamp']['secs'] )+ "," + str(imu_hed['stamp']['secs'] ) + "," + str(imu_hed['seq']) + "\n")
			fo.close()
			print("IMU Data ->  secs: {} , nsecs: {} , seq: {} ,  Status: ANOMALY".format(str(imu_hed['stamp']['secs'] ),str(imu_hed['stamp']['secs'] ),str(imu_hed['seq'])))
			imu_name_count+=1
		else:
			fo = open("output.txt","a")
			fo.write("NORMAL," + str(imu_hed['stamp']['secs'] )+ "," + str(imu_hed['stamp']['secs'] ) + "," + str(imu_hed['seq']) + "\n")
			fo.close()
			imu_name_count+=1
			print("IMU Data ->  secs: {} , nsecs: {} , seq: {} ,  Status: NORMAL".format(str(imu_hed['stamp']['secs'] ),str(imu_hed['stamp']['secs'] ),str(imu_hed['seq'])))
	

def ImageCallback(Imagedata):
	im_hed = Imagedata['header']
	img = base64.b64decode(Imagedata['source'])
	img = np.frombuffer(img, dtype=np.uint8)
	img = np.reshape(img, (128,128))
	bb=img
	img = np.true_divide(img,255)

	#img = base64.b64decode(Imagedata['data'])
	#img = np.frombuffer(img, dtype=np.uint8)
	#img = np.reshape(img[::-1], (Imagedata['height'],Imagedata['width']))
	#bb=img

	#img = resize(img,(128,128))
	#print(img.size)

	img = np.resize(img,(128,128,1))
	
	global image_temp
	global image_count
	global image_name_count
	
	if(image_count<=3):

		image_temp.append(np.array(img))
		
		if(image_count==3):
			image_temp = np.array(image_temp)
			
			evals = image_model.evaluate([np.array([image_temp[0:3]]),np.zeros((1,1,1024))],np.array([image_temp[-1]]))
			pred = image_model.predict([np.array([image_temp[0:3]]),np.zeros((1,1,1024))],verbose = False)
			
			pred = np.clip(pred*255.0, 0, 255).reshape(128,128)
			pred = pred.astype(np.uint8)	
			
			if(evals>threshold_image):
				fo = open("output_image.txt","a")
				fo.write("ANOMALY," + str(im_hed['stamp']['secs'] )+ "," + str(im_hed['stamp']['secs'] ) + "," + str(im_hed['seq']) + "\n")
				fo.close()
				print("IMAGE Data ->  secs: {} , nsecs: {} , seq: {} ,  Status: ANOMALY".format(str(im_hed['stamp']['secs'] ),str(im_hed['stamp']['secs'] ),str(im_hed['seq'])))
				cv2.imwrite('static/images/'+str(image_name_count)+ '.png',bb)
				image_name_count+=1
			else:
				fo = open("output_image.txt","a")
				fo.write("NORMAL," + str(im_hed['stamp']['secs'] )+ "," + str(im_hed['stamp']['secs'] ) + "," + str(im_hed['seq']) + "\n")
				fo.close()
				print("IMAGE Data ->  secs: {} , nsecs: {} , seq: {} ,  Status: NORMAL".format(str(im_hed['stamp']['secs'] ),str(im_hed['stamp']['secs'] ),str(im_hed['seq'])))
				cv2.imwrite('static/images/'+str(image_name_count)+ '.png',bb)
				image_name_count+=1
		image_count+=1
	
	if(image_count==4):
		
		image_temp = np.append(image_temp[1:],np.array([img]),axis = 0)
		
		evals = image_model.evaluate([np.array([image_temp[0:3]]),np.zeros((1,1,1024))],np.array([image_temp[-1]]))
		pred = image_model.predict([np.array([image_temp[0:3]]),np.zeros((1,1,1024))],verbose = False)

		pred = np.clip(pred*255.0, 0, 255).reshape(128,128)
		pred = pred.astype(np.uint8)			
			
		if(evals>threshold_image):
			fo = open("output_image.txt","a")
			fo.write("ANOMALY," + str(im_hed['stamp']['secs'] )+ "," + str(im_hed['stamp']['secs'] ) + "," + str(im_hed['seq']) + "\n")
			fo.close()
			print("IMAGE Data ->  secs: {} , nsecs: {} , seq: {} ,  Status: ANOMALY".format(str(im_hed['stamp']['secs'] ),str(im_hed['stamp']['secs'] ),str(im_hed['seq'])))
			cv2.imwrite('static/images/'+str(image_name_count)+ '.png',bb)
			image_name_count+=1
		else:
			fo = open("output_image.txt","a")
			fo.write("NORMAL," + str(im_hed['stamp']['secs'] )+ "," + str(im_hed['stamp']['secs'] ) + "," + str(im_hed['seq']) + "\n")
			fo.close()
			print("IMAGE Data ->  secs: {} , nsecs: {} , seq: {} ,  Status: NORMAL".format(str(im_hed['stamp']['secs'] ),str(im_hed['stamp']['secs'] ),str(im_hed['seq'])))
			cv2.imwrite('static/images/'+str(image_name_count)+ '.png',bb)
			image_name_count+=1
			#cv2.imwrite('n___'+str(Imagedata.header.seq)+'.png',bb)
		
