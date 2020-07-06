# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 19:02:29 2018

@author: yansinan
"""

import sys
sys.path.append('D:\\python3DrLib')
import numpy as np
from itertools import cycle

##Dr
from dr import cv as cv
class DRDo:
    def __init__(self,strWinName,inFun,para):
        self.name=strWinName
        cv.namedWindow(self.name)
        self.dictPara=para
        print(self.dictPara)
        for k in para :
            cv.createTrackbar(k,self.name,para[k],100,self.nothing)
        self.op=inFun
        print('init...'+strWinName)
    def run(self,img):
        cv.drImshowInfo(self.name,img,self.strInfo)

    def getPara(self,inK):
        k=cv.getTrackbarPos(inK,self.name)
        if inK=='k' and k%2==0:
            k=k+1
            cv.setTrackbarPos(inK,self.name,k)
            self.dictPara[inK]=k
        return cv.getTrackbarPos(inK,self.name)
    @property
    def strInfo(self):
        s=str(self.dictPara)#'k:'+str(self.k)+'    iters:'+str(self.iters)
        return s
    def nothing(self,inV):pass
    dPipeline={}
    @classmethod
    def do(cls,inImg,strWinName,inFun,**para):
        #print(type(super()))
        if DRDo.dPipeline.get(strWinName)==None:
            w=DRDo.dPipeline.setdefault(strWinName,cls(strWinName,inFun,para))
        else : 
            w=DRDo.dPipeline[strWinName]
        return w.run(inImg)
class DRBlur(DRDo):
    def run(self,img):
        if self.op==cv.GaussianBlur:
            out=self.op(img,(self.getPara('k'),)*2,self.getPara('iters'))
        elif self.op==cv.medianBlur:
            out=self.op(img,self.getPara('k'))
        elif self.op==cv.blur:
            out=self.op(img,(self.getPara('k'),)*2)
        elif self.op==cv.bilateralFilter:#(inImg,inIters,inSegColor,inSegSpace):
            out=self.op(img,self.getPara('iters'),self.getPara('segColor'),self.getPara('segSpace'))
            
        cv.drImshowInfo(self.name,out,self.strInfo)
        return out
class DRMorph(DRDo):
    #print(cv.MORPH_ELLIPSE,cv.MORPH_RECT,cv.MORPH_CROSS)#2，0，1
    
    def __init__(self,strWinName,inOp,para):
        self.kernel=para['kernel']#cv.getStructuringElement(inKernel, (k,)*2)
        del(para['kernel'])
        DRDo.__init__(self,strWinName,cv.morphologyEx,para)
        self.opMorph=inOp
        print('initMorph.'+self.strOp+'..'+strWinName)
    def run(self,img):
        if self.op==cv.morphologyEx:#cv.morphologyEx(inImg, self.operation , st, iterations=self._iters)#(inImg,inIters,inSegColor,inSegSpace):
            out=self.op(img,self.opMorph,self.structKernel,iterations=self.getPara('iters'))
        fShow=out.copy()
        fShow=cv.drPutText(fShow,self.strOp+'/'+DRMorph.arrStrKernel[self.kernel],(5,30))
        cv.drImshowInfo(self.name,fShow,self.strInfo)
        return out
    @property
    def structKernel(self):return cv.getStructuringElement(self.kernel, (self.getPara('k'),)*2)
    @property
    def strOp(self):return DRMorph.arrStrOpMorph[self.opMorph]
        #print(cv.MORPH_OPEN,cv.MORPH_CLOSE,cv.MORPH_GRADIENT,cv.MORPH_TOPHAT,cv.MORPH_BLACKHAT)#2,3,4,5,6 
    @property
    def strInfo(self):
        s=str(self.dictPara)#'k:'+str(self.k)+'    iters:'+str(self.iters)
        return s
    arrStrOpMorph=['MORPH_ERODE','MORPH_DILATE','MORPH_OPEN','MORPH_CLOSE','MORPH_GRADIENT','MORPH_TOPHAT','MORPH_BLACKHAT']    
    arrStrKernel=[ 'MORPH_RECT', 'MORPH_CROSS','MORPH_ELLIPSE']
    modes = cycle(['erode/dilate', 'open/close', 'blackhat/tophat', 'gradient'])
    str_modes = cycle([ 'MORPH_RECT', 'MORPH_CROSS','MORPH_ELLIPSE'])
    '''
    @classmethod
    def do(cls,inImg,strWinName,inOp,inKernel=cv.MORPH_ELLIPSE,k=3,iters=1):
        if DRDo.dPipeline.get(strWinName)==None:
            w=DRDo.dPipeline.setdefault(strWinName,cls(strWinName,inOp,kernel=inKernel,k=k,iters=iters))
        else : 
            w=DRDo.dPipeline[strWinName]
        return w.run(inImg)
    '''
if __name__ == '__main__':
    c=cv.CVCameraThread('camera',(640,320),1)
    while c.waitKey(1):
        f=c.lastFrame
        res=DRBlur.do(f,'tBilateral',cv.bilateralFilter,iters=7,segColor=7,segSpace=7)
        res=DRBlur.do(f,'tGaussian',cv.GaussianBlur,k=7,iters=10)
        res=DRBlur.do(f,'tMedian',cv.medianBlur,k=7)
        res=DRBlur.do(f,'tBlur',cv.blur,k=7)
        res=DRMorph.do(f,'tErode',cv.MORPH_ERODE,kernel=cv.MORPH_CROSS,k=7,iters=1)
        cv.drImshowInfo(c.name,f,f.shape)