from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QImage, QPainter, QPalette, QPixmap, QColor
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QLabel,
        QMainWindow, QMenu, QMessageBox, QScrollArea, QSizePolicy)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PIL import Image
from qtuse import array_to_qimage

# from rs_data_pro import *
import rs_data_pro
import classify
import registration
import detect_change
import dem_pro
import radiation_correction

class RsPro(QMainWindow):
	def __init__(self):
		super(RsPro, self).__init__()

		# 鼠标的位置
		self.pos1 = [0,0]
		self.pos2 = [0,0]

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

		self.setWindowTitle("遥感图像处理App")
		self.resize(640, 800)
		self.show()

	# 绘制矩形
	def paintEvent(self, event):
		width = self.pos2[0]-self.pos1[0]
		height = self.pos2[1] - self.pos1[1]     

		qp = QPainter()
		qp.begin(self) 
		# qp.setPen(QColor(168, 34, 3))
		qp.drawRect(self.pos1[0], self.pos1[1], width, height)
		qp.fillRect(self.pos1[0], self.pos1[1], width, height, QColor(168, 34, 3))       
		qp.end()

	# 处理鼠标点击事件
	# 鼠标点击
	def mousePressEvent(self, event):
		self.pos1[0], self.pos1[1] = event.pos().x(), event.pos().y()
		print(self.pos1[0], ' ', self.pos1[1])

	# 鼠标释放
	def mouseReleaseEvent(self, event):
		self.pos2[0], self.pos2[1] = event.pos().x(), event.pos().y()
		print(self.pos2[0], ' ', self.pos2[1])
		self.update()

	def open_img(self):
		filename = QFileDialog.getOpenFileName(self, "打开图片", QDir.currentPath())

		# read color image
		self.img = Image.open(filename)

	def display_data(self):
		image = array_to_qimage(self.rs_data_array)
		if image.isNull():
			QMessageBox.information(self, "Image Viewer",
			        "Cannot load %s." % fileName)
			return

		self.imageLabel.setPixmap(QPixmap.fromImage(image))
		self.scaleFactor = 1.0

		self.printAct.setEnabled(True)
		self.fitToWindowAct.setEnabled(True)
		self.updateActions()

		if not self.fitToWindowAct.isChecked():
		    self.imageLabel.adjustSize()

	# read remote sensing image
	def open_rs_data(self):
		filename = QFileDialog.getOpenFileName(self, "打开遥感图像", QDir.currentPath())
		print(filename)
		# read rs data
		self.dataset = rs_data_pro.read_as_dataset(filename[0])
		# read image data as array
		self.rs_data_array = rs_data_pro.read_as_tuple_floats(self.dataset)
		self.display_data()

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
		self.fileMenu = QMenu("&文件", self)
		self.rsView = QMenu("&图像显示", self)
		self.imgPro = QMenu("&图像处理", self)

		self.smooth = QMenu("&平滑", self)
		self.sharpen = QMenu("&锐化", self)

		# create menu for dem data process
		self.dem = QMenu("&地形", self)

		# create classify menu
		self.classify = QMenu("&分类", self)

		# create registration menu
		self.registration = QMenu("&空间配准", self)

		# create radiate correction menu
		self.rd_crr = QMenu("&辐射校正", self)

		# create change detection menu
		self.change_detect = QMenu("&变化检测", self)
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

		# 为辐射校正添加响应函数
		self.rd_crr.addAction(self.radi_corr)

		#--------------------------- detect change --------------------------
		# add action for change detection
		self.change_detect.addAction(self.detectAct)
		self.change_detect.addAction(self.cla_detectAct)
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
		self.menuBar().addMenu(self.rd_crr)
		self.menuBar().addMenu(self.change_detect)
		#--------------------------------end---------------------------------

	def createActions(self):
		#----------------------------------------------------basic operation---------------------------------------------------------------
		self.openAct = QAction("&打开遥感图像", self, shortcut="Ctrl+O", triggered=self.open_rs_data)
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
		self.openClDataAct = QAction("打开待分类文件", self, triggered=self.open_classify_data)
		self.isodataAct = QAction("isodata分类", self, triggered=self.isodata)
		self.kMeansAct = QAction("k均值", self, triggered=self.kmeans)
		self.kNearAct = QAction("k近邻", self, triggered=self.knear)
		#----------------------------------------------------------------------------------------------------------------

		# registration
		self.regist = QAction("配准", self, triggered=self.registing)

		# 辐射校正
		self.radi_corr = QAction("直方图匹配法", self, triggered=self.hist_match)

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
		filename_res = QFileDialog.getOpenFileName(self, "打开基准影像", QDir.currentPath())
		filename_des = QFileDialog.getOpenFileName(self, "打开匹配影像", QDir.currentPath())
		# registration function
		registration.registration(filename_res, filename_des)

	# 直方图匹配法的辐射校正
	def hist_match(self):
		filename_source = QFileDialog.getOpenFileName(self, "打开待匹配的文件", QDir.currentPath())
		filename_template = QFileDialog.getOpenFileName(self, "打开模板文件", QDir.currentPath())
		radiation_correction.hist_match(filename_source, filename_template)

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