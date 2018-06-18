from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QMessageBox
from PIL import Image
from osgeo import osr

import gdal
import math
import struct
import numpy as np

def read_as_dataset(filename):
	dataset = gdal.Open(filename, gdal.GA_ReadOnly)

	if not dataset:
		QMessageBox.information(self, "Remote sensing Process Error, ", "Cannot load %s." % filename)
		return

	return dataset

# read data as array for display
def read_as_tuple_floats(dataset):
	rs_data_array = np.zeros((dataset.RasterYSize, dataset.RasterXSize, 3), 'uint8')
	print('band count ' + str(dataset.RasterCount))
	if dataset.RasterCount == 1:
		rs_data_array_R = dataset.GetRasterBand(1).ReadAsArray()
		rs_data_array_G = dataset.GetRasterBand(1).ReadAsArray()
		rs_data_array_B = dataset.GetRasterBand(1).ReadAsArray()
	else:
		rs_data_array_R = dataset.GetRasterBand(3).ReadAsArray()
		rs_data_array_G = dataset.GetRasterBand(2).ReadAsArray()
		rs_data_array_B = dataset.GetRasterBand(1).ReadAsArray()

	rs_data_array[..., 0] = rs_data_array_R
	rs_data_array[..., 1] = rs_data_array_G
	rs_data_array[..., 2] = rs_data_array_B

	min_value = np.amin(rs_data_array)
	max_value = np.amax(rs_data_array)

	rs_data_array = rs_data_array*(int(256/(max_value-min_value)))

	# here, use linear strech
	return rs_data_array

# read data as array
def read_as_array(dataset):
	rs_data_array = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount), 'uint8')

	for i in range(dataset.RasterCount):
		rs_data_array[..., i] = dataset.GetRasterBand(i+1).ReadAsArray()

	# here, use linear strech
	return rs_data_array

# display data
def display_array_data(rs_data_array):
	im = Image.fromarray(rs_data_array)
	im.show()

def hist_equal_dis(dataset):
	# get the one band of the remote sensing data
	band = 1
	# rs_data_array = dataset.GetRasterBand(band).ReadRaster()
	rs_band = dataset.GetRasterBand(band)
	rs_band_array = rs_band.ReadRaster(xoff=0, yoff=0,
                           xsize=rs_band.XSize, ysize=rs_band.YSize,
                           buf_xsize=rs_band.XSize, buf_ysize=rs_band.YSize,
                           buf_type=gdal.GDT_UInt32)

	rs_band_array = np.array(struct.unpack('i' * rs_band.XSize * rs_band.YSize, rs_band_array))
	# print(rs_band_array)
	max_val = np.max(rs_band_array)
	min_val = np.min(rs_band_array)

	# calculate the frequency of an image
	static_frequency = [0] * (max_val-min_val+1)
	for i in range(rs_band.XSize):
		for j in range(rs_band.YSize):
			# print(rs_band_array[i*rs_band.XSize+j])
			static_frequency[rs_band_array[i*rs_band.XSize+j]-min_val] += 1
	# static_frequency = sum(static_frequency)
	pixels_num = rs_band.XSize * rs_band.YSize
	static_frequency = list(map(lambda x: x/pixels_num, static_frequency))

	equalization_pixel_list = []
	frequency = 0
	for i in range(len(static_frequency)):
		frequency += static_frequency[i]
		equalization_pixel = frequency * (255-0)
		equalization_pixel_list.append(equalization_pixel)

	for i in range(rs_band.XSize):
		for j in range(rs_band.YSize):
			# print(type(rs_band_array[i*rs_band.XSize+j]), equalization_pixel_list[rs_band_array[i*rs_band.XSize+j]-min_val])
			rs_band_array[i*rs_band.XSize+j] = equalization_pixel_list[rs_band_array[i*rs_band.XSize+j]-min_val]

	rs_data_array = np.zeros((dataset.RasterYSize, dataset.RasterXSize, 3), 'uint8')
	rs_data_array[..., 0] = rs_band_array.reshape((dataset.RasterXSize, dataset.RasterYSize))
	rs_data_array[..., 1] = rs_band_array.reshape((dataset.RasterXSize, dataset.RasterYSize))
	rs_data_array[..., 2] = rs_band_array.reshape((dataset.RasterXSize, dataset.RasterYSize))

	im = Image.fromarray(rs_data_array)
	im.show()

# create remote sensing data
def create_rs_data(dst_filename, fileformat, inputDS):
	x_size = inputDS.RasterXSize
	y_size = inputDS.RasterYSize
	band_count = inputDS.RasterCount

	geotransform = inputDS.GetGeoTransform()
	geoprojection = inputDS.GetProjection()

	fileformat = fileformat
	driver = gdal.GetDriverByName(fileformat)

	dst_ds = driver.Create(dst_filename, xsize=x_size, ysize=y_size,
                       bands=band_count, eType=gdal.GDT_Byte)

	dst_ds.SetGeoTransform(geotransform)
	dst_ds.SetProjection(geoprojection)

	return dst_ds, x_size, y_size, band_count	


# image smooth
def filter(dataset, filter_type='average', filter_size=3):
	# define filter size
	# filter_size = 5

	out_filename = 'filter.tif'
	dst_ds, x_size, y_size, band_count = create_rs_data(out_filename, "GTiff", dataset)
	raster = np.zeros((x_size-filter_size+1, y_size-filter_size+1), dtype=np.uint8)
	half_filter_size = int(filter_size/2)

	# 处理均值滤波
	if filter_type == 'average':
		average_filter = np.ones((filter_size, filter_size), dtype=object)
		
		for k in range(band_count):
			# read the data of one band
			rs_one_band = dataset.GetRasterBand(k+1).ReadAsArray()
			for i in range(half_filter_size, x_size-half_filter_size):
				for j in range(half_filter_size, y_size-half_filter_size):
					tmp = rs_one_band[i-half_filter_size:i-half_filter_size+filter_size, j-half_filter_size:j-half_filter_size+filter_size]
					raster[i-half_filter_size][j-half_filter_size] = np.sum(np.dot(tmp, average_filter)) / (filter_size**2)

			dst_ds.GetRasterBand(k+1).WriteArray(raster)
			print("band " + str(k+1) + " has been processed")
	# 处理中值滤波
	elif filter_type == 'median':
		for k in range(band_count):
			# read the data of one band
			rs_one_band = dataset.GetRasterBand(k+1).ReadAsArray()
			for i in range(half_filter_size, x_size-half_filter_size):
				for j in range(half_filter_size, y_size-half_filter_size):
					tmp = rs_one_band[i-half_filter_size:i-half_filter_size+filter_size, j-half_filter_size:j-half_filter_size+filter_size]
					tmp = tmp.flatten()
					tmp.sort()
					raster[i-half_filter_size][j-half_filter_size] = tmp[int((filter_size**2)/2)]

			dst_ds.GetRasterBand(k+1).WriteArray(raster)
			print("band " + str(k+1) + " has been processed")

	print('done')
	dst_ds = None

# remote sensing data sharpen
def sharpen(dataset, sharpen_type='sobel'):
	filter_size = 3
	out_filename = 'sharpen.tif'
	dst_ds, x_size, y_size, band_count = create_rs_data(out_filename, "GTiff", dataset)
	raster = np.zeros((x_size-filter_size+1, y_size-filter_size+1), dtype=np.uint8)
	half_filter_size = int(filter_size/2)

	if sharpen_type == 'sobel':
		sharpen_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
		sharpen_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

		for k in range(band_count):
			# read the data of one band
			rs_one_band = dataset.GetRasterBand(k+1).ReadAsArray()
			for i in range(half_filter_size, x_size-half_filter_size):
				for j in range(half_filter_size, y_size-half_filter_size):
					tmp = rs_one_band[i-half_filter_size:i-half_filter_size+filter_size, j-half_filter_size:j-half_filter_size+filter_size]
					x_dir_gradient = np.sum(np.dot(tmp, sharpen_filter_X))
					y_dir_gradient = np.sum(np.dot(tmp, sharpen_filter_Y))
					raster[i-half_filter_size][j-half_filter_size] = math.sqrt((x_dir_gradient**2+y_dir_gradient**2))

			dst_ds.GetRasterBand(k+1).WriteArray(raster)
			print("band " + str(k+1) + " has been processed")
	elif sharpen_type == 'laplace':
		# 拉普拉斯算子模板L3-对45°旋转各项同性
		sharpen_filter = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

		for k in range(band_count):
			# read the data of one band
			rs_one_band = dataset.GetRasterBand(k+1).ReadAsArray()
			for i in range(half_filter_size, x_size-half_filter_size):
				for j in range(half_filter_size, y_size-half_filter_size):
					tmp = rs_one_band[i-half_filter_size:i-half_filter_size+filter_size, j-half_filter_size:j-half_filter_size+filter_size]
					raster[i-half_filter_size][j-half_filter_size] = np.sum(np.dot(tmp, sharpen_filter_X))

			dst_ds.GetRasterBand(k+1).WriteArray(raster)
			print("band " + str(k+1) + " has been processed")

	print('done')