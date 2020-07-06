# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 16:12:30 2018

@author: yansinan
"""
__all__ = ['CVBlurUI','CVBlur','drAddGaussian']
#from dr.cv.CVTools import *
from dr import cv as cv
import numpy as np
#Gaussian外发光
def drAddGaussian(img,tupKernel,iters):return cv.add(img,cv.GaussianBlur(img,tupKernel,iters))

class CVUI():
    def __init__(self,inOp,inSz=2,inIters=1,inStrName='morphology'):
        self.valMid=40
        self._name=inStrName
        self._sizeKernel = inSz
        self._iters=inIters
        self._cur_mode=inOp
        cv.namedWindow(self.name)
        cv.createTrackbar('ksize', self.name, self.sizeKernel, self.valMid*2, self.eTrackbar)
        cv.createTrackbar('iters', self.name, self.iters, self.valMid, self.eTrackbar)
        cv.setMouseCallback(self.name,self.eMouse)        
        
    def eMouse(self,eButton,x,y,State,dummy):
        if eButton==1 : self.nextOperation()#左键
        if eButton ==2 : self.nextKernel()#右键
    def eTrackbar(self,dummy=None):
        self.sizeKernel = cv.getTrackbarPos('ksize', self.name)
        self.iters = cv.getTrackbarPos('iters', self.name)
        
    def updateUI(self,inImg):
        resShow=inImg.copy()
        cv.drPutText(resShow, 'mode: ' + self.cur_mode, (10, 40))
        cv.drPutText(resShow, 'operation: ' + self._strOperation, (10, 60))
        cv.drPutText(resShow, 'ksize: %d  iters: %d' % (self._sizeKernel, self._iters), (10, 100))
        cv.drImshowInfo(self.name, resShow,resShow.shape)
        return inImg
    @property 
    def sizeKernel(self): return self._sizeKernel
    @sizeKernel.setter
    def sizeKernel(self,inSz):
        self._sizeKernel=inSz
        cv.setTrackbarPos('ksize',self.name,self._sizeKernel)

    @property 
    def iters(self):
        return self._iters
    @iters.setter
    def iters(self,inbarPos):
        self._iters=inbarPos
        cv.setTrackbarPos('iters',self.name,self._iters)
    @property 
    def name(self):
        return self._name+''
    @property 
    def cur_mode(self) : return self._cur_mode

class CVBlur():
    strGaussian='gaussian'
    strMedian='median'
    strBilateral='bilateral'
    strBlur='blur'
    arrBlurUI=[]
    #cv.GaussianBlur(fgmask,(25,25),0)
    #cv.medianBlur(fCore,45)
    #cv.bilateralFilter(fsmall,9,75,75)
    #cv.blur
    def __init__(self,inOp,inSz=15,inIters=1,inIters1=75):
        self._sizeKernel = inSz
        self._iters=inIters
        self._iters1=inIters1
        self._strOperation=inOp
    def update(self,inImg,inOp='',inSz=None,inIters=None,inIters1=None):
        if inSz!=None : self.sizeKernel = inSz
        if inIters!=None : self.iters=inIters
        #优先用传入的数据
        if inOp!='' : self.strOperation=inOp
        res=inImg
        try :
            res=CVBlur.blur(inImg,self._strOperation,self._sizeKernel,self._iters,self._iters1)
        except Exception as e:
            return cv.drPutText(inImg,'Err:'+str(e),(10,inImg.shape[0]-10))
        finally: return res
        
    @property
    def strOperation(self):return self._strOperation
    @strOperation.setter
    def strOperation(self,inV):self._strOperation=inV
    @property
    def sizeKernel(self):return self._sizeKernel
    @sizeKernel.setter
    def sizeKernel(self,inV):self._sizeKernel=inV
    @property
    def iters(self):return self._iters
    @iters.setter
    def iters(self,inV):self._iters=inV
    
    def Gaussian(inImg,inSz,inIters) : return cv.GaussianBlur(inImg,(inSz,inSz),inIters)
    def Bilateral(inImg,inIters,inSegColor,inSegSpace) : return cv.bilateralFilter(inImg,inIters,inSegColor,inSegSpace)
    def Median(inImg,inIters) : return cv.medianBlur(inImg,inIters)
    def Blur(inImg,inIters) : return cv.blur(inImg,(inIters,inIters))
    def blur(inImg,inOp,inSz,inIters,inIters1=75):#inIters1 给bilateraFilter的sigmaSpace用
        if inOp==CVBlur.strBlur :
            res=CVBlur.Blur(inImg,inIters)
        elif inOp==CVBlur.strGaussian : 
            res=CVBlur.Gaussian(inImg,inSz,inIters)
        elif inOp==CVBlur.strMedian : 
            res=cv.medianBlur(inImg,inIters)
        elif inOp==CVBlur.strBilateral : 
            res=CVBlur.Bilateral(inImg,inIters,inSz,inIters1)
        else : res=cv.drPutText(inImg,str('Blur mode Not avialable '), (10, 10))
        return res
class CVBlurUI(CVUI,CVBlur):
    def __init__(self,inOp,inSz=15,inIters=1,inIters1=75,inStrName='blur'):
        CVBlur.__init__(self,inOp,inSz,inIters,inIters1)
        CVUI.__init__(self,self.strOperation,self.sizeKernel,self.iters,inStrName=inStrName)
        
    def update(self,inImg,inOp='',inSz=None,inIters=None,inIters1=None):
        res=super().update(inImg,inOp,inSz,inIters,inIters1)
        super().updateUI(res)
        return res
    @property 
    def name(self):
        return self._name+''
    @property 
    def cur_mode(self) : return self.strOperation
