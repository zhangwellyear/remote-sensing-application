import numpy as np
from rs_data_pro import read_as_dataset, read_as_array, create_rs_data
from classify import color_table, generate_classify_pic

def change_detect(img1_filename, img2_filename):
	read_dataset = read_as_dataset(img1_filename)

	img1 = read_as_array(read_as_dataset(img1_filename))
	img2 = read_as_array(read_as_dataset(img2_filename))

	change = img1 - img2

	change_dataset, x_size, y_size, band_count = create_rs_data('change.tif', 'GTiff', read_dataset)

	raster = np.zeros((x_size, y_size),  dtype=np.uint8)
	change_threshhold = 200

	for i in range(band_count):
		raster = change[:, :, i]
		raster = np.where(raster > change_threshhold, raster, 0)
		print(raster)
		change_dataset.GetRasterBand(i+1).WriteArray(raster)
		print("band " + str(i+1) + " has been processed")

	print("change detect done...")

def classify_detect(img1_filename, img2_filename):
	read_dataset = read_as_dataset(img1_filename)

	img1 = read_as_array(read_as_dataset(img1_filename))
	img2 = read_as_array(read_as_dataset(img2_filename))

	x_size = read_dataset.RasterXSize
	y_size = read_dataset.RasterYSize

	change = np.zeros((x_size, y_size))

	min_img1 = np.min(img1)
	max_img1 = np.max(img1)
	min_img2 = np.min(img2)
	max_img2 = np.max(img2)

	img1_class_num = max_img1 - min_img1 + 1
	img2_class_num = max_img2 - min_img2 + 1

	tag = 1
	change_tags = np.zeros((img1_class_num, img2_class_num))
	for i in range(img1_class_num):
		for j in range(img2_class_num):
			if i != j:
				change_tags[i, j] = tag
				tag += 1


	# print(x_size, y_size)
	for x in range(x_size):
		for y in range(y_size):
			change[x, y] = change_tags[img1[y, x]-1, img2[y, x]-1]

	generate_classify_pic(change, 'class_change.jpg', img1_class_num*img2_class_num-min(img1_class_num, img2_class_num))
	print('detect change done...')