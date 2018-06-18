import numpy as np
import rs_data_pro
import cv2
from rs_data_pro import read_as_dataset, read_as_tuple_floats

def hist_match(source_filename, template_filename):
	"""
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """
	source_filename = source_filename[0]
	template_filename = template_filename[0]

	source_array = read_as_tuple_floats(read_as_dataset(source_filename))
	template_array = read_as_tuple_floats(read_as_dataset(template_filename))

	old_shape = source_array.shape
	source_array = source_array.ravel()
	template_array = template_array.ravel()

	# get the set of unique pixel values and their corresponding indices and
	# counts
	s_values, bin_idx, s_counts = np.unique(source_array, return_inverse=True,
	                                        return_counts=True)

	t_values, t_counts = np.unique(template_array, return_counts=True)

	# take the cumsum of the counts and normalize by the number of pixels to
	# get the empirical cumulative distribution functions for the source and
	# template images (maps pixel value --> quantile)
	s_quantiles = np.cumsum(s_counts).astype(np.float64)
	s_quantiles /= s_quantiles[-1]

	t_quantiles = np.cumsum(t_counts).astype(np.float64)
	t_quantiles /= t_quantiles[-1]

	# interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
	interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
	result = interp_t_values[bin_idx].reshape(old_shape)

	# 设置图像的格式为按照Image可以读取的格式
	result = result.astype('uint8')

	# 写入图片
	outFilename = "radiation.jpg"
	print("Saving aligned image : ", outFilename); 
	cv2.imwrite(outFilename, result)

	print('radiation correction done...')

	return result