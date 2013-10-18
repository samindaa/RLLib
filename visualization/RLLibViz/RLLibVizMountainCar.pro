#-------------------------------------------------
#
# Project created by QtCreator 2013-10-08T00:39:26
#
#-------------------------------------------------

QT       += core gui

TARGET = RLLibVizMountainCar
TEMPLATE = app

INCLUDEPATH += ../../src \
               ../../simulation

SOURCES += MainMountainCar.cpp\
        ModelBase.cpp \
	ModelThread.cpp \
	MountainCarModel.cpp \
	ViewBase.cpp \
	Framebuffer.cpp \
	MountainCarView.cpp \
    	Window.cpp \
	plot/qcustomplot.cpp \
	PlotView.cpp \
	ValueFunctionView.cpp

HEADERS  += Mat.h \
    	Vec.h \
    	ViewBase.h \
	Window.h \
	ModelBase.h \
	ModelThread.h \
	Framebuffer.h \
	MountainCarView.h \
	MountainCarModel.h \
	plot/qcustomplot.h \
	PlotView.h \
	NULLView.h \
	ValueFunctionView.h

	
