#!/usr/bin/env python3
from __future__ import print_function
import roslibpy
from callbackFunc import ImuCallback
from callbackFunc import ImageCallback
import time


from flask import Flask, jsonify, render_template, request, flash
import random, threading, webbrowser
import time
import os
from os import path

app = Flask(__name__)

IMAGE_FOLDER = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

count = 0
image_count = 0

@app.route('/get_word')
def next():

	global count
	fo = open("output.txt","r")
	states_list = fo.read().split("\n")[:-1]
	count += 1

	if 0 <= count < len(states_list) -1:
		L = states_list[count].split(',')

	return jsonify(L)

@app.route('/get_word_back')
def previous():

	global count
	fo = open("output.txt","r")
	states_list = fo.read().split("\n")[:-1]
	
	if 0 < count < len(states_list):		
		L = states_list[count].split(',')
		count -= 1

	return jsonify(L)


@app.route('/get_image')
def nextImage():

	global image_count

	image_folder = 'static/images/'
	fo = open("output_image.txt","r")
	states_list = fo.read().split("\n")[:-1]

	def split_elelm(x):
		return x.split(",")

	states_list = list(map(split_elelm ,states_list))

	image_names = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

	image_det  = {}

	for i in image_names:
		ext_pos = i.index(".")
		im_index  = int(i[:ext_pos])
		image_det[i] = states_list[im_index]

	chosen_img = str(image_count) + '.png'

	data = image_det[chosen_img]

	L = ['static/images/'+ chosen_img,data]

	image_count += 1

	print (L)

	return jsonify(L)


@app.route('/get_image_back')
def backImage():

	global image_count

	image_folder = 'static/images/'
	fo = open("output_image.txt","r")
	states_list = fo.read().split("\n")[:-1]

	def split_elelm(x):
		return x.split(",")

	states_list = list(map(split_elelm ,states_list))

	image_names = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

	image_det  = {}

	for i in image_names:
		ext_pos = i.index(".")
		im_index  = int(i[:ext_pos])
		image_det[i] = states_list[im_index]

	chosen_img = str(image_count) + '.png'

	data = image_det[chosen_img]

	L = ['static/images/'+ chosen_img,data]


	if (image_count>1):
		image_count -= 1

	print (L)

	return jsonify(L)

@app.route('/')
def index():

    fo = open("output.txt","r")
    states_list = fo.read().split("\n")[:-1]
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
	text = request.form['text']
	if path.exists('./data/'+ text):
		main_(text)
		return render_template('index.html')
	else:

		#flash('Bag file not found')
		return render_template('index.html')


def main_(bag_file):

	fo = open("output_image.txt","w")
	fo.write("")
	fo.close()

	fo = open("output.txt","w")
	fo.write("")
	fo.close()

	bag_file = '/data/'+str(bag_file)
	played = False

	def bagPlayed(Bagdata):
		global played
		if(Bagdata):
			played = False
			fo = open("output.txt","a")
			time.sleep(1)
			fo.write("bag finished playing")						
			print("bag finished playing")
			fo.close()
	
	client = roslibpy.Ros(host=os.environ['HOST'], port=8080)
	client.run()
	client.on_ready(lambda: print('Is ROS connected?', client.is_connected))

	imu_listener = roslibpy.Topic(client, '/mavros/imu/data', 'sensor_msgs/Imu')
	imu_listener.subscribe(lambda IMUdata: ImuCallback([float(IMUdata['linear_acceleration']['x']),float(IMUdata['linear_acceleration']['y']),float(IMUdata['linear_acceleration']['z'])],IMUdata['header']) )

	#image_listener = roslibpy.Topic(client, '/pylon_camera_node/image_raw', 'sensor_msgs/Image')
	image_listener = roslibpy.Topic(client, '/resizedImage', 'sensor_msgs/TimeReference')
	image_listener.subscribe( lambda Imagedata: ImageCallback(Imagedata) )

	bag_listener = roslibpy.Topic(client, '/bagEnd', 'std_msgs/Bool')
	bag_listener.subscribe( lambda Bagdata: bagPlayed(Bagdata) )

	bag_player = roslibpy.Topic(client, '/bagStart', 'std_msgs/String')

	#To play Bag file - change above bag path and call funtion below (now only runs once)
	##funtion start
	if(client.is_connected and (not played)):
		bag_player.publish(roslibpy.Message({'data': bag_file}))
		played = True
	##funtion end

	try:
		while True:
			pass
	except KeyboardInterrupt:
		client.terminate()


 
if __name__ == "__main__":
 
    #threading.Timer(15.00, lambda: webbrowser.open("http://127.0.0.1:5000/") ).start()
 
    app.run(host='0.0.0.0',port=5000, debug=True)