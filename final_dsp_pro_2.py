from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from pyqtgraph import PlotWidget, PlotItem
import pyqtgraph as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq
import numpy.fft as fft
import scipy
from scipy import signal
import matplotlib.pyplot as plt

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(791, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.Tab_2 = QtWidgets.QTabWidget(self.centralwidget)
        self.Tab_2.setStyleSheet("font: 12pt \"MS Shell Dlg 2\";")
        self.Tab_2.setObjectName("Tab_2")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.tab_3)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.splitter = QtWidgets.QSplitter(self.tab_3)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.Recovary_graph = QtWidgets.QComboBox(self.layoutWidget)
        self.Recovary_graph.setObjectName("Recovary_graph")
        self.Recovary_graph.addItem("")
        self.Recovary_graph.addItem("")
        self.Recovary_graph.addItem("")
        self.horizontalLayout_3.addWidget(self.Recovary_graph)
        self.loadsignals= QtWidgets.QComboBox(self.layoutWidget)
        self.loadsignals.setObjectName("loadsignals\n""")
        self.loadsignals.addItem("")
        self.horizontalLayout_3.addWidget(self.loadsignals)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.main_graph= PlotWidget(self.layoutWidget)
        self.main_graph.setObjectName("main_graph\n""")
        self.horizontalLayout.addWidget(self.main_graph)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.widget = QtWidgets.QWidget(self.layoutWidget)
        self.widget.setObjectName("widget")
        self.verticalLayout.addWidget(self.widget)
        self.Slider_for_controlling_freq= QtWidgets.QSlider(self.layoutWidget)
        self.Slider_for_controlling_freq.setOrientation(QtCore.Qt.Vertical)
        self.Slider_for_controlling_freq.setObjectName("Slider\n""")
        self.verticalLayout.addWidget(self.Slider_for_controlling_freq)
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.Hide= QtWidgets.QPushButton(self.layoutWidget)
        self.Hide.setObjectName("Hide\n""")
        self.horizontalLayout_2.addWidget(self.Hide)
        self.Show = QtWidgets.QPushButton(self.layoutWidget)
        self.Show.setObjectName("Show")
        self.horizontalLayout_2.addWidget(self.Show)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.secondary_graph= PlotWidget(self.splitter)
        self.secondary_graph.setObjectName("secondary_graph\n""")
        self.horizontalLayout_4.addWidget(self.splitter)
        self.Tab_2.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout(self.tab_4)
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.splitter_2 = QtWidgets.QSplitter(self.tab_4)
        self.splitter_2.setOrientation(QtCore.Qt.Vertical)
        self.splitter_2.setObjectName("splitter_2")
        self.splitter_3 = QtWidgets.QSplitter(self.splitter_2)
        self.splitter_3.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_3.setObjectName("splitter_3")
        self.primitive_graph = PlotWidget(self.splitter_3)
        self.primitive_graph.setObjectName("primitive_graph")
        self.layoutWidget_5 = QtWidgets.QWidget(self.splitter_3)
        self.layoutWidget_5.setObjectName("layoutWidget_5")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.layoutWidget_5)
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.Magnitude = QtWidgets.QLabel(self.layoutWidget_5)
        self.Magnitude.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";")
        self.Magnitude.setObjectName("Magnitude")
        self.verticalLayout_10.addWidget(self.Magnitude)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.Mag_text = QtWidgets.QLineEdit(self.layoutWidget_5)
        self.Mag_text.setObjectName("Mag_text")
        self.horizontalLayout_9.addWidget(self.Mag_text)
        self.verticalLayout_10.addLayout(self.horizontalLayout_9)
        self.Phase = QtWidgets.QLabel(self.layoutWidget_5)
        self.Phase.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";")
        self.Phase.setObjectName("Phase")
        self.verticalLayout_10.addWidget(self.Phase)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.Phase_text = QtWidgets.QLineEdit(self.layoutWidget_5)
        self.Phase_text.setObjectName("Phase_text")
        self.horizontalLayout_15.addWidget(self.Phase_text)
        self.verticalLayout_10.addLayout(self.horizontalLayout_15)
        self.Frequency = QtWidgets.QLabel(self.layoutWidget_5)
        self.Frequency.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";")
        self.Frequency.setObjectName("Frequency")
        self.verticalLayout_10.addWidget(self.Frequency)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.Freq_text = QtWidgets.QLineEdit(self.layoutWidget_5)
        self.Freq_text.setObjectName("Freq_text")
        self.horizontalLayout_16.addWidget(self.Freq_text)
        self.verticalLayout_10.addLayout(self.horizontalLayout_16)
        self.verticalLayout_9.addLayout(self.verticalLayout_10)
        self.Plot = QtWidgets.QPushButton(self.layoutWidget_5)
        self.Plot.setStyleSheet("font: 11pt \"MS Shell Dlg 2\";\n" "font: 10pt \"MS Shell Dlg 2\";")
        self.Plot.setObjectName("Plot")
        self.verticalLayout_9.addWidget(self.Plot)
        self.layoutWidget_6 = QtWidgets.QWidget(self.splitter_2)
        self.layoutWidget_6.setObjectName("layoutWidget_6")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.layoutWidget_6)
        self.verticalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.Delete = QtWidgets.QComboBox(self.layoutWidget_6)
        self.Delete.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";")
        self.Delete.setObjectName("Delete")
        self.Delete.addItem("")
        self.horizontalLayout_17.addWidget(self.Delete)
        self.Submit = QtWidgets.QPushButton(self.layoutWidget_6)
        self.Submit.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";")
        self.Submit.setObjectName("Submit")
        self.horizontalLayout_17.addWidget(self.Submit)
        self.Add = QtWidgets.QPushButton(self.layoutWidget_6)
        self.Add.setStyleSheet("font: 10pt \"MS Shell Dlg 2\";")
        self.Add.setObjectName("Add")
        self.horizontalLayout_17.addWidget(self.Add)
        self.verticalLayout_11.addLayout(self.horizontalLayout_17)
        self.composer_graph = PlotWidget(self.layoutWidget_6)
        self.composer_graph.setObjectName("composer_graph")
        self.verticalLayout_11.addWidget(self.composer_graph)
        self.horizontalLayout_18.addWidget(self.splitter_2)
        self.Tab_2.addTab(self.tab_4, "")
        self.horizontalLayout_5.addWidget(self.Tab_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 791, 18))
        self.menubar.setObjectName("menubar")
        self.menuOpen = QtWidgets.QMenu(self.menubar)
        self.menuOpen.setObjectName("menuOpen")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.menuOpen.addAction(self.actionOpen)
        self.menubar.addAction(self.menuOpen.menuAction())
        self.retranslateUi(MainWindow)
        self.Tab_2.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        # Buttons Activation 
        self.Hide.clicked.connect(self.hide)
        self.Show.clicked.connect(self.show)
        self.Plot.clicked.connect(lambda: self.plot())
        self.Delete.activated.connect(self.onSelected)
        self.Submit.clicked.connect(lambda: self.submit())
        self.Add.clicked.connect(lambda: self.add())
        self.actionOpen.triggered.connect(lambda: self.open_file())
        self.submit_is_clicked = 0   #Flag for the composer Signal
        self.composerlist = []
        self.amplitudelist = []
        self.frquencylist = [] 
        self.magnitudelist=[]
        self.phaselist=[]
        self.loadsignals.activated.connect(self.pick_signal)
        self.Slider_for_controlling_freq.sliderReleased.connect(self.slider_freq)
        self.Slider_for_controlling_freq.setMaximum(3)  # Max_Value
        self.Slider_for_controlling_freq.setMinimum(0)  # Min_Value
        self.Slider_for_controlling_freq.setValue(2)    # Default_Value

    def plot(self):
        self.primitive_graph.clear()
        magnitude = float(self.Mag_text.text())
        phase = float(self.Phase_text.text())
        Frequency = float(self.Freq_text.text())
        self.frquencylist.append(Frequency)
        self.magnitudelist.append(magnitude)
        self.phaselist.append(phase)
        samplingFrequency = 1000                                           # 1000 sample
        samplingInterval = 1 / samplingFrequency                           # Calculate the time step
        self.time = np.arange(0, 1, samplingInterval)                      # 1000 values for time between 0 and 1
        pi = np.pi
        phi = phase * pi / 180
        amplitude = magnitude * (np.sin(2 * pi * Frequency * self.time + phi))    
        self.amplitudelist.append(amplitude)                        # The y values for each signal is stored in a list
        self.primitive_graph.plot(self.time, amplitude)
        self.Delete.addItem("Mag: " + self.Mag_text.text() + "/" + "Feq: " + self.Freq_text.text() + "/ph: " + self.Phase_text.text()) #Each added signal can be deleted and specified by its magnitude, phase and frequency

    def add(self):

        self.composer_graph.clear()
        self.composer = list(map(sum, zip(*self.amplitudelist)))     # The y values after the summation for the added signals
        self.composer_graph.plot(self.time, self.composer)

    def submit(self):
        self.submit_is_clicked = 1                       # That is an indication that is the currrent signal is from composer
        self.main_graph.clear()
        self.main_graph.plot(self.time, self.composer)
        self.composerlist.append(self.composer)           # y values for all the signals summated by composer
        self.composer_graph.clear()
        self.amplitudelist = []
        self.Delete.clear()
        self.Delete.addItem("Delete")
        self.maxfrequency = max(self.frquencylist)  
        self.loadsignals.addItem(str(self.maxfrequency) + "Hz signal") #Each signal is submitted is added in the "loadsignals" combobox
        self.slider_freq()

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None, "QFileDialog.getOpenFileName()", "", "All Files (*);;csv Files (*.csv)", options=options)
        if fileName:
            self.read_file(fileName)

    def read_file(self, file_path):
        self.main_graph.clear()
        path = file_path
        data = pd.read_csv(path)
        self.csv_values_on_y_axis = data.values[:, 0] 
        self.csv_values_on_x_axis = np.arange(0, 9.99, 0.01) 
        self.data_line = self.main_graph.plot(self.csv_values_on_x_axis, self.csv_values_on_y_axis) # Plotting time and amplitude on the main graph axes
        self.F_max = self.get_max_frequency(self.csv_values_on_y_axis)  # Calling the fun for getting the max freq for our csv file
        print(self.F_max)
        self.slider_freq()  

    def get_max_frequency(self, amplitude):
        spectrum = fft.fft(amplitude)  # computing the Fourier transform
        freq = fft.fftfreq(len(spectrum))  # the magnitudes of the FFT # Discrete Fourier Transform sample frequencies
        print("freq")
        print(freq)                           
        threshold = 0.5 * max(abs(spectrum))  # cutting the list of the spectrum at some threshold
        mask = abs(spectrum) > threshold      # identifing the dominant frequencies in the spectrum
        print("mask") 
        print(mask)
        peaks = freq[mask]               # The contents of peaks is the frequencies in units of the sampling rate
        print("peaks")
        print(peaks)
        peaks = abs(peaks)      # Results in two values for peaks
        fmax = max(peaks * 100)
        print("fmax")
        print(fmax)
        return fmax 

    def onSelected(self, x):          # Delete combobox
        if x == 1:
            self.amplitudelist[0] = [0] * 1000   # Replacement for y values in amplitudelist[0] with zeros to delete it
            self.add()                           # adding this replacement in composer list
        if x == 2:
            self.amplitudelist[1] = [0] * 1000
            self.add()
        if x == 3:
            self.amplitudelist[2] = [0] * 1000
            self.add()
        if x == 4:
            self.amplitudelist[3] = [0] * 1000
            self.add()
        if x == 5:
            self.amplitudelist[4] = [0] * 1000
            self.add()
        if x == 6:
            self.amplitudelist[5] = [0] * 1000
            self.add()

    def pick_signal(self, y):  # Picking an signal from loadsignals combobox
        self.main_graph.clear()
        if y == 1:
            self.main_graph.plot(self.time, self.composerlist[0])

        elif y == 2:
            self.main_graph.plot(self.time, self.composerlist[1])

        elif y == 3:
            self.main_graph.plot(self.time, self.composerlist[2])

        elif y == 4:
            self.main_graph.plot(self.time, self.composerlist[3])

    def hide(self):
        self.secondary_graph.hide()   # hidding secondary_graph when hide putton is pushed or the main_graph is choosen

    def show(self):
        self.secondary_graph.show()

    def slider_freq(self):
        
        factor=int(self.Slider_for_controlling_freq.value())  # factor value range is [0:3]
        if self.submit_is_clicked == 1:                            # The composer signal is controlled
           fs = int(factor * self.maxfrequency)                   # fs = [0:3]*fmax
           if fs == 0:                                           # the division by zero is not allowed
               if self.Recovary_graph.currentIndex()== 1:
                  self.main_graph.clear()
                  self.main_graph.plot(self.time, self.composer)  # Just plotting the main signal without sampling
                  self.hide()                                      # Automatic pushed hide button when main_graph is choosen
               elif  self.Recovary_graph.currentIndex()==2:
                   self.secondary_graph.clear()                    # plotting the main signal is only on the main_graph      
                   self.show()
           else:

               if self.Recovary_graph.currentIndex() ==1:
                    ts = 1/fs
                    ynew = scipy.signal.resample(self.composer, fs)  # amplitude samples
                    tnew = np.arange(0, 1, 1/fs)                     # time samples
                    self.main_graph.clear()
                    self.main_graph.plot(self.time, self.composer)
                    self.main_graph.plot(tnew, ynew, pen=None, symbol='o') 
                    y_reconstructed = 0                                     # initial value
                    for index in range(0,  len(tnew)):
                         y_reconstructed += ynew[index] * np.sinc((np.array(self.time) - ts * index )/ ts) # The reconstruction Formula
                    self.main_graph.plot(self.time,y_reconstructed, pen=pg.mkPen('r')) # Red pen for reconstructed signal
                    self.hide() 

               elif  self.Recovary_graph.currentIndex()==2:
                  ts =1/fs
                  ynew2 = scipy.signal.resample(self.composer, fs)
                  print(max(ynew2))
                  tnew2 = np.arange(0, 1, 1/fs)
                  print(tnew2)
                  self.secondary_graph.clear()
                  self.secondary_graph.plot(tnew2, ynew2, pen=None, symbol='o')
                  y_reconstructed = 0 
                  for index in range(0,  len(tnew2)):
                          y_reconstructed += ynew2[index] * np.sinc((np.array(self.time) - ts * index )/ ts)
                  self.secondary_graph.plot(self.time,y_reconstructed,pen=pg.mkPen('r'))
                  self.show()
        else :
           fs = int(factor *self.F_max )
           if fs==0:
               if self.Recovary_graph.currentIndex()==1:
                 self.main_graph.clear()
                 self.main_graph.plot(self.csv_values_on_x_axis, self.csv_values_on_y_axis)
                 self.hide()
               elif  self.Recovary_graph.currentIndex()==2:
                  self.secondary_graph.clear()
                  
           else:
               if self.Recovary_graph.currentIndex()==1:
                  ts=1/fs
                  ynew = scipy.signal.resample(self.csv_values_on_y_axis, 10*fs)
                  print(max(ynew))
                  tnew = np.arange(0, 10, step=1/fs, dtype=float)
                  print(tnew)
                  self.main_graph.clear()
                  self.main_graph.plot(self.csv_values_on_x_axis, self.csv_values_on_y_axis)
                  self.main_graph.plot(tnew, ynew,pen=None,symbol="o")
                  y_reconstructed = 0 
                  for index in range(0,  len(tnew)):
                          y_reconstructed += ynew[index] * np.sinc((np.array(self.csv_values_on_x_axis) - ts * index )/ ts)
                  self.main_graph.plot(self.csv_values_on_x_axis,  y_reconstructed ,pen=pg.mkPen('r'))
                  self.hide()
               elif  self.Recovary_graph.currentIndex()==2:
                    ts=1/fs
                    ynew = scipy.signal.resample(self.csv_values_on_y_axis, 10*fs) 
                    tnew = np.arange(0, 10, step=1/fs, dtype=float)
                    self.secondary_graph.clear()
                    self.secondary_graph.plot(tnew, ynew,pen=None,symbol="o")
                    y_reconstructed = 0 
                    for index in range(0,  len(tnew)):
                          y_reconstructed += ynew[index] * np.sinc((np.array(self.csv_values_on_x_axis) - ts * index )/ ts)
                    self.secondary_graph.plot(self.csv_values_on_x_axis, y_reconstructed,pen=pg.mkPen('r'))
                    self.show()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Recovary_graph.setItemText(0, _translate("MainWindow", "Recovery Graph"))
        self.Recovary_graph.setItemText(1, _translate("MainWindow", "Main Graph"))
        self.Recovary_graph.setItemText(2, _translate("MainWindow", "Seondary Graph"))
        self.loadsignals.setItemText(0, _translate("MainWindow", "Load Signal"))
        self.label_2.setText(_translate("MainWindow", "3 F"))
        self.label.setText(_translate("MainWindow", "0 F"))
        self.Hide.setText(_translate("MainWindow", "Hide"))
        self.Show.setText(_translate("MainWindow", "Show"))
        self.Tab_2.setTabText(self.Tab_2.indexOf(self.tab_3), _translate("MainWindow", "Tab 1"))
        self.Magnitude.setText(_translate("MainWindow", "Magnitude"))
        self.Phase.setText(_translate("MainWindow", "Phase"))
        self.Frequency.setText(_translate("MainWindow", "Frequency"))
        self.Plot.setText(_translate("MainWindow", "Plot"))
        self.Delete.setItemText(0, _translate("MainWindow", "Delete"))
        self.Submit.setText(_translate("MainWindow", "Submit"))
        self.Add.setText(_translate("MainWindow", "Add"))
        self.Tab_2.setTabText(self.Tab_2.indexOf(self.tab_4), _translate("MainWindow", "Tab 2"))
        self.menuOpen.setTitle(_translate("MainWindow", "File"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

    # References
    # 1] https://github.com/jimmyg1997/Python-Digital-Signal-Processing-Basics/blob/main/analogic%20signal%20-%20sampling%20-%20reconstruction%20-%20FT.ipynb
    # 2] https://stackoverflow.com/questions/46934084/interpolate-two-sine-waves-with-different-time-points-based-on-cycle-percent-mat
    # 3] https://stackoverflow.com/questions/8582559/determining-frequency-of-an-array-in-python
    # 4] https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html
    # 5] https://i.imgur.com/sa29LPL.jpg