import random
import numpy as np
import math
import rs_data_pro
import cv2
"""
	method: 	isodata
	parameter:  c——预期的类数
				N(c)——初始聚类中心个数（可以不等于c）
				θ(n)——每一类中允许的最小模式数目（若少于此类就不能单独成为一类）
				θ(s)——类内各分量分布距离标准差上限（大于此数就分裂）
				θ(D)——两类中心间最小距离下限（若小于此数，这两类应该合并）
				L——在每次迭代中可以合并的类的最多对数
				I——允许的最多迭代次数
	author:		zwy

"""
#------------------------------------------------------ begin isodata ---------------------------------------------------------
# step 2: create seed
def create_seed(rs_data_array, x_size, y_size, N_c, band_count):
	# create initial seed for the classify
	seed_points = np.zeros((N_c, band_count))

	x_partition = int(x_size / N_c)
	y_partition = int(y_size / N_c)

	for i in range(N_c):
		index_X = random.randint(i*x_partition, (i+1)*x_partition)
		index_Y = random.randint(i*y_partition, (i+1)*y_partition)
		print(rs_data_array.shape)
		seed_points[i, ...] = rs_data_array[index_X, index_Y, ...]

	return seed_points

def belong(rs_data_array, seed_points, x_size, y_size, N_c):
	classes_tag = np.zeros((x_size, y_size))

	# classify data by the initial seed
	for i in range(x_size):
		for j in range(y_size):
			dis = [0] * N_c
			for c in range(N_c):
				dis[c] = sum(abs(seed_points[c, ...] - rs_data_array[..., i, j]))
			classes_tag[i][j] = dis.index(min(dis))

	return classes_tag

def cal_center(rs_data_array, classes_tag, N_c, band_count):
	class_center = np.zeros((N_c, band_count))

	for i in range(band_count):
		for j in range(N_c):
			array_one_band = rs_data_array[i, ...]
			tmp_sum_one_band = sum((array_one_band[classes_tag == j]))
			one_class_number = sum(sum(classes_tag == j))
			class_center[j, i] = (tmp_sum_one_band / one_class_number)

	return class_center

def dis_to_center(rs_data_array, classes_tag, center, x_size, y_size, N_c):
	dis_in_diff_classes = np.zeros(N_c)

	for c in range(N_c):
		count_num = 0
		for i in range(x_size):
			for j in range(y_size):
				if classes_tag[i, j] == c:
					count_num += 1
					dis_in_diff_classes[c] += sum(abs(rs_data_array[..., i, j] - center[c, ...]))

		dis_in_diff_classes[c] = dis_in_diff_classes[c] / count_num
	
	return dis_in_diff_classes

def cal_general_dis(classes_tag, N_c, dis_in_diff_classes, total_pixels):
	aver_d = 0

	for i in range(N_c):
		aver_d += (sum(sum(classes_tag == i)) * dis_in_diff_classes[i])

	aver_d = aver_d / total_pixels

	# print(aver_d)
	return aver_d

# step6:
# dis_in_diff_classes: z
# calculate the standard deviation
def cal_standard_deviation(rs_data_array, classes_tag, dis_in_diff_classes, N_c, x_size, y_size, band_count):
	sigma = np.zeros((N_c, band_count))

	for c in range(N_c):
		for b in range(band_count):
			count_num = 0
			for i in range(x_size):
				for j in range(y_size):
					if classes_tag[i, j] == c:
						count_num += 1
						sigma[c, b] += ((rs_data_array[b, i, j]-dis_in_diff_classes[c]) ** 2)

			sigma[c, b] = math.sqrt((sigma[c, b]/count_num))

	return sigma

def get_max_sigma(sigmas):
	return np.max(sigmas, axis=1), sigmas.argmax(axis=1)

def split(max_sigma, sigmas, max_sigma_index, classes_tag, dis_in_diff_classes, aver_d, theta_n, N_c, classes_center, band_count):
	# print(dis_in_diff_classes[0], aver_d)
	# print(max_sigma_index)
	for i in range(N_c):
		n_j = sum(sum(classes_tag == i))
		if dis_in_diff_classes[i] > aver_d or n_j > 2*(theta_n+1):
			# np.append(classes_center, np.zeros((1, band_count)))
			tmp = np.zeros((N_c+1, band_count))
			tmp[:N_c, ...] = classes_center
			classes_center = tmp

			# print(classes_center)
			k = 0.4
			classes_center[i, max_sigma_index[i]] = sigmas[i, max_sigma_index[i]] - k*sigmas[i, max_sigma_index[i]]
			classes_center[-1, ...] = sigmas[i, ...]
			classes_center[-1, max_sigma_index[i]] = classes_center[-1, max_sigma_index[i]] + k*sigmas[i, max_sigma_index[i]]
			N_c += 1

	# print('split', classes_center, N_c)
	return classes_center, N_c

def merge(D, N_c, theta_D, L, classes_center, classes_tag):
	D_minus_theta = []
	for i in range(N_c):
		for j in range(i+1, N_c):
			if D[i, j] < theta_D:
				D_minus_theta.append(D[i, j])

	D_minus_theta.sort()

	merge_count = 0
	for l in range(L):
		i, j = np.where(D == D_minus_theta[l])
		classes_center[i, ...] = (sum(sum(classes_tag==i))*classes_center[i, ...] + sum(sum(classes_tag==j))*classes_center[j, ...])/\
								 (sum(sum(classes_tag==i))+sum(sum(classes_tag==j)))
		classes_center = np.delete(classes_center, (j), axis=0)
		merge_count += 1

	N_c -= merge_count
	# print('merge')
	return classes_center, N_c

# step9
# calculate the number in single class
def cal_dis_to_center(classes_center, N_c):
	D = np.zeros((N_c, N_c-1))

	for i in range(N_c):
		for j in range(i+1, N_c):
			D[i, j] = sum(abs(classes_center[i]-classes_center[j]))


def isodata(dataset, c=5, N_c=3, theta_n=100000, theta_s=10, theta_D=200, L=4, I=50):
	# step 1
	x_size = dataset.RasterXSize
	y_size = dataset.RasterYSize
	band_count = dataset.RasterCount
	rs_data_array = dataset.ReadAsArray()

	"""
		step1: create seed
	"""
	seed_points = create_seed(rs_data_array, x_size, y_size, N_c, band_count)

	loop_iter = 1
	
	while loop_iter < I:

		"""
			step2: classify
		"""
		classes_tag = belong(rs_data_array, seed_points, x_size, y_size, N_c)

		"""
			step3: judge by theta_n: whether the class is to be merged
		"""
		# record whether the center is cancled
		is_center_cancle = True
		# at begin, it should be True, because you should run the following code at beginning
		while is_center_cancle:
			i = 0
			while i < N_c:
				size_class_i = sum(sum(classes_tag == i))
				# print(size_class_i, i)
				if size_class_i < theta_n:
					is_center_cancle = True
					N_c -= 1
					# goto step2
					seed_points = create_seed(rs_data_array, x_size, y_size, N_c, band_count)
					# run step3 again
					classes_tag = belong(rs_data_array, seed_points, x_size, y_size, N_c)

				i += 1

			if i == N_c and is_center_cancle:
				is_center_cancle = False

		"""
			step4: calculate the center of every class
		"""
		# 1) calculate the center of a class
		classes_center = cal_center(rs_data_array, classes_tag, N_c, band_count)
		# 2) calculate the distance from one point to the class center
		dis_in_diff_classes = dis_to_center(rs_data_array, classes_tag, classes_center, x_size, y_size, N_c)
		# 3) calculate the sum distance of all kinds of class
		aver_d = cal_general_dis(classes_tag, N_c, dis_in_diff_classes, x_size*y_size)
		
		"""
			step5: decide stop, split or merge
			step6: calculate the standard deviation
		"""
		if loop_iter == I:
			# theta_D = 0
			cal_dis_to_center(classes_center, N_c)
			# # print(9)
			# pass
		else:
			if N_c <= int(c/2):
				standard_deviation = cal_standard_deviation(rs_data_array, classes_tag, dis_in_diff_classes, N_c, x_size, y_size, band_count)
				# step7
				max_sigma, max_sigma_index = get_max_sigma(standard_deviation)
				# step8
				classes_center, N_c = split(max_sigma, standard_deviation, max_sigma_index, classes_tag, \
											dis_in_diff_classes, aver_d, theta_n, N_c, classes_center, band_count)
				# step9
				D = cal_dis_to_center(classes_center, N_c)
				# step10: merge
				classes_center, N_c = merge(D, N_c, theta_D, L, classes_center, classes_tag)
				# print('sigma', max_sigma, max_sigma_index)
			elif N_c >= 2*c:
				# step9
				D = cal_dis_to_center(classes_center, N_c)
				# step10: merge
				classes_center, N_c = merge(D, N_c, theta_D, L, classes_center, classes_tag)
			else:
				if loop_iter % 2 == 0:
					standard_deviation = cal_standard_deviation(rs_data_array, classes_tag, dis_in_diff_classes, N_c, x_size, y_size, band_count)
					# step7
					max_sigma, max_sigma_index = get_max_sigma(standard_deviation)
					# step8
					classes_center, N_c = split(max_sigma, standard_deviation, max_sigma_index, classes_tag, \
												dis_in_diff_classes, aver_d, theta_n, N_c, classes_center, band_count)
					# step9
					D = cal_dis_to_center(classes_center, N_c)
					# step10: merge
					classes_center, N_c = merge(D, N_c, theta_D, L, classes_center, classes_tag)
					# print('sigma', max_sigma, max_sigma_index)
				else:
					# step9
					D = cal_dis_to_center(classes_center, N_c)
					# step10: merge
					classes_center, N_c = merge(D, N_c, theta_D, L, classes_center, classes_tag)

		"""
			step7: get max sigma
		"""
		seed_points = classes_center
		print(loop_iter, 'iter: ')
		for c in range(N_c):
			print(sum(sum(classes_tag == c)))

		loop_iter += 1

#--------------------------------------------------------- isodata end ---------------------------------------------------------
def k_means(k_num, filename, max_iter=20):
	dataset = rs_data_pro.read_as_dataset(filename)
	rs_data_array = rs_data_pro.read_as_array(dataset)

	x_size = dataset.RasterXSize
	y_size = dataset.RasterYSize
	band_count = dataset.RasterCount

	kmeans_seed_points = create_seed(rs_data_array, x_size, y_size, k_num, band_count)

	classes_tag = np.zeros((x_size, y_size))

	for i_iter in range(max_iter):
		# calculate distances for each class
		distances = np.zeros((x_size, y_size, k_num))

		for i in range(k_num):
			for i_x in range(x_size):
				for i_y in range(y_size):
					distances[i_x, i_y, i] = sum(abs(rs_data_array[i_x, i_y, ...] - kmeans_seed_points[i, ...]))

		# find min distances and classify
		for i_x in range(x_size):
			for i_y in range(y_size):
				index = np.where(distances[i_x, i_y, ...] == min(distances[i_x, i_y, ...]))
				classes_tag[i_x, i_y] = index[0][0]

		# recalculate seed points
		kmeans_seed_points = np.zeros((k_num, band_count))
		count_num = [1] * k_num

		for i in range(k_num):
			for i_x in range(x_size):
				for i_y in range(y_size):
					if classes_tag[i_x, i_y] == i:
						count_num[i] += 1
						kmeans_seed_points[i, ...] += rs_data_array[i_x, i_y, ...]

			kmeans_seed_points[i, ...] /= count_num[i]

		# print(kmeans_seed_points)
		print(count_num)
		print('iter ' + str(i_iter) + " done...")

	generate_classify_pic(classes_tag, 'classify_result.jpg', k_num)
	# print(classes_tag)
	generate_txt('classify.txt', kmeans_seed_points)

#--------------------------------------------------- generate classify pictures -----------------------------------------------
def generate_classify_pic(classes_tag, des_filename, k_num):
	tags_shape = classes_tag.shape
	img_shape = (tags_shape[0], tags_shape[1], 3)
	colors = color_table()

	array = np.zeros(img_shape)
	for i in range(k_num):
		for x in range(tags_shape[0]):
			for y in range(tags_shape[1]):
				if classes_tag[x, y] == i:
					array[x, y, ...] = colors[i]

	# mat_array = cv2.fromarray(array)
	cv2.imwrite(des_filename, array)

#------------------------------------------------ generate kmeans result by txt -----------------------------------------------
def generate_txt(filename, seed_points):
	output = open(filename, 'w')
	output_str = ""
	for i in range(len(seed_points)):
		for j in range(len(seed_points[0])):
			output_str = output_str + str(seed_points[i][j]) + " "
		output_str += "\n"

	output.write(output_str)

#-------------------------------------------------------- supervised classification -------------------------------------------
def supervised_classify(classify_txt, classify_filename):
	file = open(classify_txt, 'r')
	txt = file.read()
	txt_lines = txt.split('\n')

	k_num = len(txt_lines)-1
	band = len(txt_lines[0].split(" ")) - 1

	seeds = np.zeros((k_num, band))

	for i in range(len(txt_lines)-1):
		numbers_txt = txt_lines[i].split(" ")
		for j in range(len(numbers_txt)-1):
			seeds[i, j] = float(numbers_txt[j])


	rs_dataset = rs_data_pro.read_as_dataset(classify_filename)
	rs_data_array = rs_data_pro.read_as_array(rs_dataset)

	x_size = rs_dataset.RasterXSize
	y_size = rs_dataset.RasterYSize
	band_count = rs_dataset.RasterCount

	distances = np.zeros((x_size, y_size, k_num))

	for x in range(x_size):
		for y in range(y_size):
			for i in range(k_num):
				distances[x, y, i] = sum(abs(rs_data_array[x, y, ...] - seeds[i, ...]))

	classes_tag = np.zeros((x_size, y_size))

	# find min distances and classify
	for i_x in range(x_size):
		for i_y in range(y_size):
			index = np.where(distances[i_x, i_y, ...] == min(distances[i_x, i_y, ...]))
			classes_tag[i_x, i_y] = index[0][0]

	generate_classify_pic(classes_tag, 'classify_result2.jpg', k_num)
	print("k near classify done...")

#------------------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------- general use -------------------------------------------------------
#---------------------------------------------------------- color table -------------------------------------------------------
def color_table():
	return np.array([
		[0, 0, 0],			# black
		[255, 255, 255],	# White
		[255, 0, 0],		# red1
		[0, 255, 0],		# Lime
		[0, 0, 255],		# Blue
		[255, 255, 0],		# Yellow
		[0, 255, 255],		# Cyan
		[255, 0, 255],		# Magenta/Fuchsia
		[192, 192, 192],	# Silver
		[128, 128, 128],	# Gray
		[128, 0, 0],		# Maroon
		[128, 128, 0],		# Olive
		[0, 128, 0],		# Green
		[128, 0, 128],		# Purple
		[0, 128, 128],		# Teal
		[0, 0, 128],		# Navy
		[139, 0, 0],		# Dark red
		[165, 42, 42],		# brown
		[178,34,34],		# Firebrick
		[220,20,60],		# Crimson
		[255,99,71],		# Tomato
		[255,127,80],		# Coral
		[205,92,92],		# Indian Red
		[240,128,128],		# Light coral
		[233,150,122],		# Dark salmon
		[250,128,114],		# Salmon
		[255,160,122],		# Light salmon
		[255,69,0],			# Orange Red
		[255,140,0],		# Dark Orange
		[255, 165, 0],		# Oragne
		[255,215,0],		# Gold
		[184,134,11],		# Dark golden rod
		[218,165,32],		# Golden Rod
		[238,232,170],		# Pale Golden Rod
	])
#------------------------------------------------------------------------------------------------------------------------------