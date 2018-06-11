import cv2
import numpy as np
from rs_data_pro import read_as_dataset, read_as_tuple_floats

def registration(img_src, img_des):
	src_filename = img_src[0]
	dst_filename = img_des[0]

	src_img = read_as_tuple_floats(read_as_dataset(src_filename))
	# src_img = src_img[:1000, :1000, :]

	dst_img = read_as_tuple_floats(read_as_dataset(dst_filename))
	# dst_img = src_img[:1000, :1000, :]

	# print(src_img.shape)

	# 找到关键点及其sift特征
	sift = cv2.xfeatures2d.SIFT_create()
	src_kp, src_des = sift.detectAndCompute(src_img, None)
	dst_kp, dst_des = sift.detectAndCompute(dst_img, None)

	# 定义匹配函数，对特征点进行匹配
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(src_des, dst_des, k=2)

	# 存储匹配好的点
	good = []

	for m,n in matches:
	    if m.distance < 0.7*n.distance:
	        good.append(m)
	
	# 定义最小的匹配点的个数
	MIN_MATCH_COUNT = 6
	if len(good) > MIN_MATCH_COUNT:
	    src_pts = np.float32([ src_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	    dst_pts = np.float32([ dst_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
	else:
	    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
	    matchesMask = None

	# 找到仿射变换系数
	h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

	# 应用仿射变换系数，对待校正图像进行校正
	height, width, _ = dst_img.shape
	# 待转换图像坐标角点
	corners = np.array([[0, 0, 1], [0, height, 1], [width, 0, 1], [width, height, 1]])
	# 转换后的图像坐标角点
	trans_corners = np.dot(h, np.transpose(corners))

	# 计算转换后的图片的大小
	regWidth = int(max(trans_corners[0]))
	regHeight = int(max(trans_corners[1]))

	imReg = cv2.warpPerspective(src_img, h, (regWidth, regHeight))

	# 写入图片
	outFilename = "aligned.jpg"
	print("Saving aligned image : ", outFilename); 
	cv2.imwrite(outFilename, imReg)

	# 输出仿射变换矩阵
	print("Estimated homography : \n",  h)