# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 6.9.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QMainWindow, QMenuBar, QPushButton,
    QSizePolicy, QStatusBar, QTreeView, QVBoxLayout,
    QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1008, 688)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setMaximumSize(QSize(16777215, 35))
        font = QFont()
        font.setPointSize(25)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.label)

        self.frame_3 = QFrame(self.centralwidget)
        self.frame_3.setObjectName(u"frame_3")
        self.frame_3.setMaximumSize(QSize(16777215, 70))
        self.frame_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Shadow.Plain)
        self.horizontalLayout_2 = QHBoxLayout(self.frame_3)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.btnBrowse = QPushButton(self.frame_3)
        self.btnBrowse.setObjectName(u"btnBrowse")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnBrowse.sizePolicy().hasHeightForWidth())
        self.btnBrowse.setSizePolicy(sizePolicy)
        self.btnBrowse.setMaximumSize(QSize(75, 16777215))
        font1 = QFont()
        font1.setPointSize(20)
        font1.setBold(False)
        font1.setItalic(False)
        font1.setUnderline(False)
        font1.setStrikeOut(False)
        self.btnBrowse.setFont(font1)
        self.btnBrowse.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.btnBrowse.setAutoExclusive(False)

        self.horizontalLayout_2.addWidget(self.btnBrowse)

        self.lblPath = QLabel(self.frame_3)
        self.lblPath.setObjectName(u"lblPath")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lblPath.sizePolicy().hasHeightForWidth())
        self.lblPath.setSizePolicy(sizePolicy1)
        self.lblPath.setMinimumSize(QSize(350, 50))

        self.horizontalLayout_2.addWidget(self.lblPath)

        self.btnStartAlysis = QPushButton(self.frame_3)
        self.btnStartAlysis.setObjectName(u"btnStartAlysis")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.btnStartAlysis.sizePolicy().hasHeightForWidth())
        self.btnStartAlysis.setSizePolicy(sizePolicy2)
        self.btnStartAlysis.setMaximumSize(QSize(16777215, 75))
        self.btnStartAlysis.setAutoFillBackground(False)
        self.btnStartAlysis.setAutoDefault(False)
        self.btnStartAlysis.setFlat(False)

        self.horizontalLayout_2.addWidget(self.btnStartAlysis)

        self.btnStop = QPushButton(self.frame_3)
        self.btnStop.setObjectName(u"btnStop")
        self.btnStop.setMaximumSize(QSize(16777215, 75))

        self.horizontalLayout_2.addWidget(self.btnStop)


        self.verticalLayout.addWidget(self.frame_3)

        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy3)
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.frame.setLineWidth(14)
        self.horizontalLayout = QHBoxLayout(self.frame)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.tvFiles = QTreeView(self.frame)
        self.tvFiles.setObjectName(u"tvFiles")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.tvFiles.sizePolicy().hasHeightForWidth())
        self.tvFiles.setSizePolicy(sizePolicy4)
        self.tvFiles.setMaximumSize(QSize(200, 16777215))

        self.horizontalLayout.addWidget(self.tvFiles)

        self.lblVido = QWidget(self.frame)
        self.lblVido.setObjectName(u"lblVido")
        self.lblVido.setToolTipDuration(-6)

        self.horizontalLayout.addWidget(self.lblVido)


        self.verticalLayout.addWidget(self.frame)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1008, 33))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.btnStartAlysis.setDefault(False)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u904b\u52d5\u5206\u6790\u5668", None))
        self.btnBrowse.setText(QCoreApplication.translate("MainWindow", u"\u76ee\u9304", None))
        self.lblPath.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.btnStartAlysis.setText(QCoreApplication.translate("MainWindow", u"\u958b\u59cb\u5206\u6790", None))
        self.btnStop.setText(QCoreApplication.translate("MainWindow", u"\u66ab\u505c\u64a5\u653e", None))
    # retranslateUi

