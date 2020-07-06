#!/usr/bin/env python

'''
Morphology operations.

Usage:
  morphology.py [<image>]

Keys:
  1 or 左键  - change operation
  2 or 右键  - change structure element shape
  ESC - exit
'''


#import numpy as np
import cv2 as cv
from itertools import cycle
#from common import draw_str
def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

#TODO:初始化时侯指定某种模式，并且和指针同步
class CVMorphology:
    modes = cycle(['erode/dilate', 'open/close', 'blackhat/tophat', 'gradient'])
    str_modes = cycle(['MORPH_ELLIPSE', 'MORPH_RECT', 'MORPH_CROSS'])
    def __init__(self,inOp,inKernel=cv.MORPH_ELLIPSE,inSz=2,inIters=1,inStrName='morphology'):
        self.name=inStrName
        self.cur_mode = next(self.modes)
        self._strKernel = next(self.str_modes)
        self.operation=inOp
        self.kernel=inKernel
        #if (self.operation==cv.MORPH_ERODE) or (self.operation==cv.MORPH_OPEN) or (self.operation==cv.MORPH_BLACKHAT) :self.sz=inSz+self.valMid
        #else : self.sz = inSz
        self._sizeKernel = inSz
        self.iters = inIters
        self._img=[]
    #如果传入参数，按参数操作，如果没传，从trackbar读取
    def update(self,inImg,inSz=None,inIters=None):
        if len(inImg)==0 :return []
        if inSz!=None : self.sizeKernel = inSz
        if inIters!=None : self.iters =inIters
        
        st = cv.getStructuringElement(self.kernel, (self.sizeKernel, self._sizeKernel))
        self.res = cv.morphologyEx(inImg, self.operation , st, iterations=self._iters)
        return self.res
    '''
    @property
    def img(self): return self._img
    @img.setter
    def img(self,inImg):
        self._img=inImg
        self.update()
    '''
    @property
    def operation(self):return getattr(cv, self._strOperation)
    @operation.setter
    def operation(self,inCV_Morph_Mode):
        #print(cv.MORPH_OPEN,cv.MORPH_CLOSE,cv.MORPH_GRADIENT,cv.MORPH_TOPHAT,cv.MORPH_BLACKHAT)#2,3,4,5,6      
        if inCV_Morph_Mode== cv.MORPH_DILATE : 
            self._strOperation='MORPH_DILATE'
            self.cur_mode='erode/dilate'
        elif inCV_Morph_Mode== cv.MORPH_ERODE : 
            self._strOperation='MORPH_ERODE'
            self.cur_mode='erode/dilate'
        elif inCV_Morph_Mode== cv.MORPH_OPEN :
            self._strOperation='MORPH_OPEN'
            self.cur_mode='open/close'
        elif inCV_Morph_Mode== cv.MORPH_CLOSE : 
            self._strOperation='MORPH_CLOSE'
            self.cur_mode='open/close'
        elif inCV_Morph_Mode== cv.MORPH_GRADIENT : 
            self._strOperation='MORPH_GRADIENT'
            self.cur_mode='gradient'
        elif inCV_Morph_Mode== cv.MORPH_TOPHAT : 
            self._strOperation='MORPH_TOPHAT'
            self.cur_mode='blackhat/tophat'
        elif inCV_Morph_Mode== cv.MORPH_BLACKHAT : 
            self._strOperation='MORPH_BLACKHAT'
            self.cur_mode='blackhat/tophat'
        else : return
        while self.cur_mode != next(self.modes) : continue
    @property
    def kernel(self):return getattr(cv, self.strKernel)
    @kernel.setter
    def kernel(self,inCV_Morph_Mode): 
        #print(cv.MORPH_ELLIPSE,cv.MORPH_RECT,cv.MORPH_CROSS)#2，0，1
        if inCV_Morph_Mode== cv.MORPH_ELLIPSE : self._strKernel='MORPH_ELLIPSE'
        elif inCV_Morph_Mode== cv.MORPH_RECT : self._strKernel='MORPH_RECT'
        elif inCV_Morph_Mode== cv.MORPH_CROSS : self._strKernel='MORPH_CROSS'
        else : return
        while self._strKernel != next(self.str_modes) : continue
    @property
    def sizeKernel(self):return self._sizeKernel
    @sizeKernel.setter
    def sizeKernel(self,inV):self._sizeKernel=inV
    @property 
    def iters(self):
        return self._iters
    @iters.setter
    def iters(self,inV):
        self._iters=inV
        
    @property
    def strKernel(self): return self._strKernel#'MORPH_' + self.cur_str_mode.upper()
    @property
    def strOperation(self): return self._strOperation#'MORPH_' + self.cur_str_mode.upper()
        
    #定义操作
    def nextOperation(self): self.cur_mode=next(self.modes)
    def nextKernel(self): self._strKernel=next(self.str_modes)
    
class CVMorphologyUI(CVMorphology):
    valMid=40
    def __init__(self,inOp=None,inKernel=cv.MORPH_ELLIPSE,inSz=2,inIters=1,inStrName='morphology'):
        self._barPos=0

        CVMorphology.__init__(self,inOp,inKernel,inSz,inIters,inStrName)
        
        if (self.operation==cv.MORPH_ERODE) or (self.operation==cv.MORPH_OPEN) or (self.operation==cv.MORPH_BLACKHAT) : self.barPos=abs(self.sizeKernel-self.valMid)
        elif self.operation==cv.MORPH_GRADIENT : self.barPos=inSz
        else : self.barPos=inSz+self.valMid
        
        cv.namedWindow(self.name)
        cv.createTrackbar('op/size', self.name, self.barPos, self.valMid*2, self.eTrackbar)
        cv.createTrackbar('iters', self.name, self.iters, self.valMid, self.eTrackbar)
        cv.setMouseCallback(self.name,self.eMouse)        
        
    def eMouse(self,eButton,x,y,State,dummy):
        if eButton==1 : self.nextOperation()#左键
        if eButton ==2 : self.nextKernel()#右键
    def eTrackbar(self,dummy=None):
        self.barPos = cv.getTrackbarPos('op/size', self.name)
        self.iters = cv.getTrackbarPos('iters', self.name)
        #self.update(self.barPos,self.iters)
        
    def update(self,inImg):
        opers = self.cur_mode.split('/')
        if len(opers) > 1:
            op = opers[(self.barPos - self.valMid) > 0]
        else:
            op = opers[0]
        self._strOperation = 'MORPH_' + op.upper()
        resShow=super().update(inImg).copy()
        #if inSz!=None : self.barPos = inSz
        #if inIters!=None : self.iters =inIters

        draw_str(resShow, (10, 20), 'mode: ' + self.cur_mode)
        draw_str(resShow, (10, 40), 'operation: ' + self._strOperation)
        draw_str(resShow, (10, 60), 'structure: ' + self._strKernel)
        draw_str(resShow, (10, 80), 'ksize: %d  iters: %d' % (self._sizeKernel, self._iters))
        cv.imshow(self.name, resShow)
        return self.res
    @property 
    def barPos(self): return self._barPos
    @barPos.setter
    def barPos(self,inSz):
        self._barPos=inSz
        ''' '''
        opers = self.cur_mode.split('/')
        if len(opers) > 1:
            sizeKernel = self.barPos - self.valMid
            sizeKernel = abs(sizeKernel)
            self._sizeKernel=max(1,sizeKernel)
        else : self._sizeKernel = self.barPos  
        cv.setTrackbarPos('op/size',self.name,self._barPos)

    @property 
    def iters(self):
        return self._iters
    @iters.setter
    def iters(self,inbarPos):
        self._iters=inbarPos
        cv.setTrackbarPos('iters',self.name,self._iters)


if __name__ == '__main__':
    print(__doc__)

    video_capture = cv.VideoCapture('output.avi')
    ret, img = video_capture.read()

    morph=CVMorphologyUI(cv.MORPH_CLOSE,inSz=6,inIters=8,inStrName='step1MORPH_CLOSE01')
    #morph.kernel=cv.MORPH_CROSS
    #morph.operation=cv.MORPH_TOPHAT
    #update()
    fgbgCNT = cv.bgsegm.createBackgroundSubtractorCNT()
    while True:
        ch = cv.waitKey(142)
        ret, f = video_capture.read()
        img = fgbgCNT.apply(f)
        if ch == 27:
            break
        if ch == ord('1'):
            morph.nextOperation()#cur_mode = next(modes)
        if ch == ord('2'):
            morph.nextKernel()#_strKernel = next(str_modes)
        morph.update(img)
    video_capture.release()

    cv.destroyAllWindows()
