import os

image_folder = 'processed'
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

print (image_det)