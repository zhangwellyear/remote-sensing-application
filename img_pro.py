from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PIL import Image

# from rs_data_pro import *
import rs_data_pro
import classify
import registration
import detect_change
import dem_pro

class RsPro(QMainWindow):
	def __init__(self):
		super(RsPro, self).__init__()

		self.printer = QPrinter()
		self.scaleFactor = 0.0

		self.imageLabel = QLabel()
		self.imageLabel.setBackgroundRole(QPalette.Base)
		self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
		self.imageLabel.setScaledContents(True)

		self.scrollArea = QScrollArea()
		self.scrollArea.setBackgroundRole(QPalette.Dark)
		self.scrollArea.setWidget(self.imageLabel)
		self.setCentralWidget(self.scrollArea)

		self.createActions()
		self.createMenus()

		self.setWindowTitle("Image View")
		self.resize(800, 640)

	def open_img(self):
		filename = QFileDialog.getOpenFileName(self, "Open Img", QDir.currentPath())

		# read color image
		self.img = Image.open(filename)

	# read remote sensing image
	def open_rs_data(self):
		filename = QFileDialog.getOpenFileName(self, "Open RS Data", QDir.currentPath())
		print(filename)
		# read rs data
		self.dataset = rs_data_pro.read_as_dataset(filename[0])
		# read image data as array
		self.rs_data_array = rs_data_pro.read_as_tuple_floats(self.dataset)

	def rsview(self):		
		# display the data have been read
		rs_data_pro.display_array_data(self.rs_data_array)

	# histogram equalization for a band
	def hist_equal(self):
		rs_data_pro.hist_equal_dis(self.dataset)

	def print_(self):
		dialog = QPrintDialog(self.printer, self)
		if dialog.exec_():
			painter = QPainter(self.printer)
			rect = painter.viewport()
			size = self.imageLabel.pixmap().size()
			size.scale(rect.size(), Qt.KeepAspectRatio)
			painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
			painter.setWindow(self.imageLabel.pixmap().rect())
			painter.drawPixmap(0, 0, self.imageLabel.pixmap())

	def fitToWindow(self):
		fitToWindow = self.fitToWindowAct.isChecked()
		self.scrollArea.setWidgetResizable(fitToWindow)

		if not fitToWindow:
			self.normalSize()

		self.updateActions()

	def createMenus(self):
		#------------------------add menu-------------------------------------
		self.fileMenu = QMenu("&File", self)
		self.rsView = QMenu("&ImageView", self)
		self.imgPro = QMenu("&ImageProcess", self)

		self.smooth = QMenu("&Smooth", self)
		self.sharpen = QMenu("&Sharpen", self)

		# create menu for dem data process
		self.dem = QMenu("&Dem", self)

		# create classify menu
		self.classify = QMenu("&Classify", self)

		# create registration menu
		self.registration = QMenu("&Registration", self)

		# create change detection menu
		self.change_detect = QMenu("&ChangeDetect", self)

		#------------------------add actions----------------------------------
		# add actions for file operate
		self.fileMenu.addAction(self.openAct)
		self.fileMenu.addAction(self.openColorImg)

		# add actions for image view
		self.rsView.addAction(self.rsViewAct)
		self.rsView.addAction(self.histEqualAct)

		# add actions for image filter
		self.smooth.addAction(self.averageFilterAct) 
		self.smooth.addAction(self.medianFilterAct)

		# add actions for image sharpen
		self.sharpen.addAction(self.sharpenAct)

		# add actions for dem image process
		self.dem.addAction(self.OpenDEMAct)
		self.dem.addAction(self.demDisplayAct)

		# add actions for classify
		self.classify.addAction(self.openClDataAct)
		self.classify.addAction(self.isodataAct)
		self.classify.addAction(self.kMeansAct)
		self.classify.addAction(self.kNearAct)

		# add action for registration
		self.registration.addAction(self.regist)

		#--------------------------- detect change --------------------------
		# add action for change detection
		self.change_detect.addAction(self.detectAct)
		self.change_detect.addAction(self.cla_detectAct)
		#--------------------------------------------------------------------

		#----------------------------- 辐射校正 ------------------------------
		self.
		#--------------------------------------------------------------------

		#----------------------------add action end--------------------------

		#----------------------------add menu to toolbar---------------------
		self.menuBar().addMenu(self.fileMenu)
		self.menuBar().addMenu(self.rsView)
		self.menuBar().addMenu(self.imgPro)
		self.menuBar().addMenu(self.smooth)
		self.menuBar().addMenu(self.sharpen)
		self.menuBar().addMenu(self.dem)
		self.menuBar().addMenu(self.classify)
		self.menuBar().addMenu(self.registration)
		self.menuBar().addMenu(self.change_detect)
		#--------------------------------end---------------------------------

	def createActions(self):
		#----------------------------------------------------basic operation---------------------------------------------------------------
		self.openAct = QAction("&Open...", self, shortcut="Ctrl+O", triggered=self.open_rs_data)
		self.openColorImg = QAction("Image Open...", self, shortcut="Ctrl+I", triggered=self.open_img)
		self.printAct = QAction("&Print...", self, shortcut="Ctrl+P", triggered=self.print_)
		self.fitToWindowAct = QAction("&Fit to Window", self, shortcut="Ctrl+F", enabled=False, checkable=True, triggered=self.fitToWindow)
		self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)
		self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
		self.normalSizeAct = QAction("&normal Size", self, shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)

		#---------------------------------------------------algorithm operation------------------------------------------
		# view rs data
		self.rsViewAct = QAction("View Data", self, triggered=self.rsview)
		self.histEqualAct = QAction("Hist Equal", self, triggered=self.hist_equal)

		# smooth operation
		self.averageFilterAct = QAction("&Average Smooth", self, triggered=self.average_filter)
		self.medianFilterAct = QAction("&Median Filter", self, triggered=self.median_filter)

		# sharpen operation
		self.sharpenAct = QAction("Sobel", self, triggered=self.sobel_sharpen)

		#------------------------------------------ dem data process ----------------------------------------------------
		self.demDisplayAct = QAction("Dem Dispaly", self, triggered=self.dem_display)
		self.OpenDEMAct = QAction("OpenDEM", self, triggered=self.dem_open)
		#------------------------------------------ dem process end -----------------------------------------------------

		#------------------------------------------- classify pro -------------------------------------------------------
		# classify data
		self.openClDataAct = QAction("Open", self, triggered=self.open_classify_data)
		self.isodataAct = QAction("isodata", self, triggered=self.isodata)
		self.kMeansAct = QAction("k均值", self, triggered=self.kmeans)
		self.kNearAct = QAction("k近邻", self, triggered=self.knear)
		#----------------------------------------------------------------------------------------------------------------

		# registration
		self.regist = QAction("registration", self, triggered=self.registing)

		#---------------------------------------- change detection ------------------------------------------------------
		# change detection
		self.detectAct = QAction("直接变化检测", self, triggered=self.detectChange)
		self.cla_detectAct = QAction("分类后变化检测", self, triggered=self.classChangeDetect)
		#-------------------------------------- change detection end ----------------------------------------------------

	def updateActions(self):
		self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
		self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
		self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

	def zoomIn(self):
		self.scaleImage(1.25)

	def zoomOut(self):
		self.scaleImage(0.8)

	def normalSize(self):
		self.imageLabel.adjustSize()
		self.scaleFactor = 1.0

	# process filter operation
	def average_filter(self):
		rs_data_pro.filter(self.dataset)

	def median_filter(self):
		rs_data_pro.filter(self.dataset, filter_type='median')

	# process sharpen operation
	def sobel_sharpen(self):
		rs_data_pro.sharpen(self.dataset)

	#------------------------dem process-----------------------------
	# process dem data
	def dem_open(self):
		dem_filepath= QFileDialog.getOpenFileName(self, "Open DEM Image", QDir.currentPath() + '')
		self.dem_img_filename = dem_filepath[0]

	def dem_display(self):
		dem_pro.dem_show(self.dem_img_filename)
	#----------------------------------------------------------------

	#-------------------------------------------------- classify -----------------------------------------------------
	# classify the data
	def open_classify_data(self):
		classify_filepath = QFileDialog.getOpenFileName(self, "Open DEM Image", QDir.currentPath() + '')
		self.classify_filename = classify_filepath[0]

	def isodata(self):
		dataset = rs_data_pro.read_as_dataset(self.classify_filename)
		classify.isodata(dataset)

	def kmeans(self):
		classify.k_means(5, self.classify_filename, max_iter=10)

	def knear(self):
		classify.supervised_classify("classify.txt", self.classify_filename)
	#------------------------------------------------ classify end --------------------------------------------------------

	# registration for the data
	def registing(self):
		filename_res = QFileDialog.getOpenFileName(self, "Open Img1", QDir.currentPath())
		filename_des = QFileDialog.getOpenFileName(self, "Open Img2", QDir.currentPath())
		# registration function
		registration.registration(filename_res, filename_des)

	#------------------------------------------------------ change detect ---------------------------------------------------------------
	def detectChange(self):
		img1 = QFileDialog.getOpenFileName(self, "Open Img1", QDir.currentPath() + '\data\change_detection\direct_compare')
		img2 = QFileDialog.getOpenFileName(self, "Open Img2", QDir.currentPath() + '\data\change_detection\direct_compare')
		# detect change function
		# img1[0] represent the file name of the img1
		# img2[0] represent the file name of the img2
		detect_change.change_detect(img1[0], img2[0])

	def classChangeDetect(self):
		img1 = QFileDialog.getOpenFileName(self, "Open Classify Img1", QDir.currentPath() + '\data\change_detection\classify_postprocess')
		img2 = QFileDialog.getOpenFileName(self, "Open Classify Img2", QDir.currentPath() + '\data\change_detection\classify_postprocess')
		detect_change.classify_detect(img1[0], img2[0])
	#----------------------------------------------- change detect end -------------------------------------------------------------------

if __name__ == '__main__':
	import sys

	app = QApplication(sys.argv)
	remotePro = RsPro()
	remotePro.show()
	sys.exit(app.exec_())