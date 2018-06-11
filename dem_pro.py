from osgeo import gdal
from numpy import gradient, pi, arctan, arctan2, sin, cos, sqrt, zeros, uint8
from rs_data_pro import display_array_data

def dem_show(dem_filename, way=0):
	ds = gdal.Open(dem_filename)
	band = ds.GetRasterBand(1)
	arr = band.ReadAsArray()

	hs_array = hillshade(arr, 315, 45)

	display_array_data(hs_array)
	
	# plt.imshow(hs_array,cmap='Greys')
	# plt.show()
	

def hillshade(array, azimuth, angle_altitude):
	x, y = gradient(array)
	slope = pi/2. - arctan(sqrt(x*x, y*y))
	aspect = arctan2(-x, y)
	azimuthrad = azimuth*pi / 180.
	altituderad = angle_altitude*pi / 180.

	shaded = sin(altituderad) * sin(slope)\
     + cos(altituderad) * cos(slope)\
     * cos(azimuthrad - aspect)

	# print('hillshade display done...')
	return 255*(shaded + 1)/2