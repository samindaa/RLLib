#-------------------------------------------------
#
# Project created by QtCreator 2014-12-03T01:25:57
#
#-------------------------------------------------

QT       += core gui widgets printsupport

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = RLLibViz
TEMPLATE = app

INCLUDEPATH += ../../include \
               ../../simulation \
               ../../util

SOURCES += main.cpp\
    plot/qcustomplot.cpp \
    ThreadBase.cpp \
    LearningThread.cpp \
    EvaluationThread.cpp \
    MountainCarModel.cpp \
    MountainCarView.cpp \
    PlotView.cpp \
    ValueFunctionView.cpp \
    ModelBase.cpp \
    ViewBase.cpp \
    Window.cpp \
    RLLibVizMediator.cpp \
    ContinuousGridworldModel.cpp \
    ContinuousGridworldView.cpp \
    Framebuffer.cpp \
    SwingPendulumModel.cpp \
    SwingPendulumModel2.cpp \
    SwingPendulumView.cpp \
    MountainCarModel2.cpp \
    SwingPendulumModel3.cpp \
    MountainCarModel3.cpp \
    SwingPendulumModel4.cpp \
    AcrobotModel.cpp \
    AcrobotView.cpp \
    CartPoleModel.cpp \
    CartPoleView.cpp \
    WindowLayout.cpp

HEADERS  += \
    plot/qcustomplot.h \
    ThreadBase.h \
    LearningThread.h \
    EvaluationThread.h \
    MountainCarModel.h \
    MountainCarView.h \
    NULLView.h \
    PlotView.h \
    ValueFunctionView.h \
    ModelBase.h \
    ViewBase.h \
    Window.h \
    Vec.h \
    Mat.h \
    RLLibVizMediator.h \
    ContinuousGridworldModel.h \
    ContinuousGridworldView.h \
    Framebuffer.h \
    SwingPendulumModel.h \
    SwingPendulumModel2.h \
    SwingPendulumView.h \
    MountainCarModel2.h \
    SwingPendulumModel3.h \
    MountainCarModel3.h \
    SwingPendulumModel4.h \
    AcrobotModel.h \
    AcrobotView.h \
    CartPoleModel.h \
    CartPoleView.h \
    WindowLayout.h


FORMS    += \
    RLLibVizForm.ui
