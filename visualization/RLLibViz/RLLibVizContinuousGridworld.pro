#-------------------------------------------------
#
# Project created by QtCreator 2013-10-08T00:39:26
#
#-------------------------------------------------

QT       += core gui widgets printsupport

TARGET = RLLibVizContinuousGridworld
TEMPLATE = app

INCLUDEPATH += ../../include \
               ../../simulation

SOURCES += MainContinuousGridworld.cpp\
        ModelBase.cpp \
	ModelThread.cpp \
	ContinuousGridworldModel.cpp \
	ViewBase.cpp \
	Framebuffer.cpp \
	ContinuousGridworldView.cpp \
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
	ContinuousGridworldView.h \
	ContinuousGridworldModel.h \
	plot/qcustomplot.h \
	PlotView.h \
	NULLView.h \
	ValueFunctionView.h

	
