#-------------------------------------------------
#
# Project created by QtCreator 2013-10-08T00:39:26
#
#-------------------------------------------------

QT       += core gui

TARGET = RLLibVizSwingPendulum
TEMPLATE = app

INCLUDEPATH += ../../src \
               ../../simulation

SOURCES += MainSwingPendulum.cpp\
        ModelBase.cpp \
	ModelThread.cpp \
	SwingPendulumModel.cpp \
	ViewBase.cpp \
	Framebuffer.cpp \
	SwingPendulumView.cpp \
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
	SwingPendulumView.h \
	SwingPendulumModel.h \
	plot/qcustomplot.h \
	PlotView.h \
	NULLView.h \
	ValueFunctionView.h

	
