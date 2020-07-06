# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 13:40:33 2018

@author: yansinan
"""
import sys
sys.path.append('../../')

#import cv2 as cv
from dr import cv as cv
import numpy as np
import time
from dr.Event import ThreadLoop
from dr import isWin
'''
class squeezeNetProcess(EventDispatcher):
    def __init__(self): #,dataPath="" ,inImageShape=[1,227,227,3],inArrClassName=None#,inImageShape=[1,227,227,3]
        EventDispatcher.__init__(self)
        #self.sqz=squeezeNet('sqz_full.mat',inImageShape =[1,227,227,3],inArrClassName=classes)
        self.res=[]
        self.start()
    #def init(self):
        
    #复写基类的函数,这个函数会在while isActive==True循环里循环，每次阻塞等待
    def funLoop(self):
        if self.queFeed.qsize()>0 :
            args=self.queFeed.get()
            self.res=args['sqz'].predict(inImage=args['img'],inLayerName=args['layerName'],inRefineDataKeep=args['keep'])
            self.queResult.put(self.res)
            
    def predictProcess(self,inArgs):
        if self.queFeed.qsize()>0 : return []
        self.queFeed.put(inArgs)
        self.eProcess.set()
        if self.queResult.qsize()>0 : return self.queResult.get()
        else : return []
    def destroy(self):
        self.isActive=False

    def predict(self,inImage,inLayerName,inRefineDataKeep):
        sqz.predict(inImage,inLayerName,inRefineDataKeep)
        return
    @property
    def shapeResult(self):return (1000,)#self.sqz.shapeResult
    @property
    def shapeImg(self):return (227,227)#self.sqz.shapeImg
'''
    


class DRCamera():
    shapeCapture=(640,480)
    def __init__(self,inShapeImg=(),inCamDevice=-1):
        self._timeLast=time.time()
        self.video_capture=None
        self.devId=inCamDevice
        self.shapeImg=DRCamera.shapeCapture#(640,480)
        if len(inShapeImg)>=2 : 
            self.shapeImg=(inShapeImg[1],inShapeImg[0])
            if self.shapeImg[0]==0 : self.shapeImg[0] = DRCamera.shapeCapture[0]
            if self.shapeImg[1]==0 : self.shapeImg[1] = DRCamera.shapeCapture[1]
            self._frame=self.frameDefault
            self._lastFrame=self.frameDefault
            self.orgFrame=self.frameDefault
            self.camInit(self.shapeImg)
        #最后获取的图像
        self._frame=None
    
    def camInit(self,inShapeImg):
        self.video_capture=cv.VideoCapture(self.devId)
        if type(self.devId)==int: 
            #self.video_capture.set(cv.CAP_PROP_FRAME_WIDTH, inShapeImg[0]); 
            #self.video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, inShapeImg[1]);
            self.video_capture.set(cv.CAP_PROP_FRAME_WIDTH, DRCamera.shapeCapture[0]); 
            self.video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, DRCamera.shapeCapture[1]);
            #self.video_capture.set(cv.CAP_PROP_SATURATION,0.2);
            print('CVCamera camInit',self.video_capture)
        outFrame=self.getFrame(inShape=inShapeImg,idx=2)
        return outFrame#self.name,self.video_capture
        
    def getFrame(self,inShape=None,idx=0,e=None):
        self._timeLast=time.time()
        if inShape==None : inShape= self.shapeImg
        if self.video_capture != None:
            ret, frame = self.video_capture.read()
            if not ret : frame = self.frameDefault
        else :
            #frame = frame = cv.imread('testHuman'+ str(idx) +'.jpg')
            frame = self.frameDefault
            ret = len(frame)
        self.orgFrame=frame.copy()
        if ret :
            if frame.shape != inShape+(3,) : frame=self.getResizePad(frame,self.shapeImg)#frame = cv.resize(frame,self.shapeImg,interpolation=cv.INTER_CUBIC)
        else : 
            frame=self.getResizePad(self.frameDefault,self.shapeImg)
        self._lastFrame=frame.copy()
        return frame
    #居中等比缩放
    def getResizePad(self,inImg,inDstSize=(227,227)):
        return cv.drResizeCenter(inImg,inDstSize)
        '''
        desired_size = inDstSize[1]
        old_size = inImg.shape[:2] # old_size is in (height, width) format
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        
        # new_size should be in (width, height) format
        im = cv2.resize(inImg, (new_size[1], new_size[0]))
        
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        
        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        return new_im
        '''
    @property
    #没有新数据就返回[],返回数据的副本
    def frame(self): return self.getFrame()
    @property
    #总是至少返回一个数据,返回数据的副本
    def lastFrame(self):return self._lastFrame
    @property
    def fps(self):return round(1.0/(time.time()-self._timeLast),1)
    @property
    def frameDefault(self):
        out=np.zeros(shape=(np.abs(self.shapeImg+(3,))),dtype=np.uint8)#np.zeros(self.shapeImg)
        #out=cv.drColor(self.shapeImg,0)
        cv.putText(out, "NO Image...", (int(self.shapeImg[0]/2)-int(8*22/2),int(self.shapeImg[1]/2)), cv.FONT_HERSHEY_PLAIN,2, (0,128,255),thickness = 2)
        return out.copy()
    
    def destroy(self):
        if self.video_capture!=None : self.video_capture.release()   

class DRCameraThread(DRCamera,ThreadLoop):
    def __init__(self,inShapeImg,inCamDevice=-1,inControl=None):
        #传入()使得camInit先不执行，在thread里执行
        DRCamera.__init__(self,(),inCamDevice)
        self.shapeImg=inShapeImg
        self._frame=self.frameDefault
        self._lastFrame=self.frameDefault
        #20180418给while主循环用
        self.outterControl=self.nothing
        if inControl!=None : self.outterControl=inControl

        ThreadLoop.__init__(self,name='Camera')
        self.eFrame=self.eProcess
        self.start()
    def init(self):self.camInit(self.shapeImg)
    #复写基类的函数,这个函数会在while isActive==True循环里循环，每次阻塞等待
    def funLoop(self):
        self._frame=self.getFrame().copy()
        #self.queResult.put(self._frame)
        return self._lastFrame
    ######20180418给while主循环用
    def nothing(self,k):return k
        
    @property
    def step(self):return self.waitKey(1)
    
    def waitKey(self,inV):
        if isWin() : k=cv.waitKey(inV) & 0xFF
        else :
            k=0
            time.sleep(inV/1000)
        #k=cv.waitKey(1) & 0xFF
        res=self.outterControl(k)
        if k == ord('q'): res=-1
        if res==-1 : 
            self.isActive=False
            if isWin() : cv.destroyAllWindows()
        return self.isActive
    ######20180418给while主循环用---end        
    @property
    #没有新数据就返回[],返回数据的副本
    def frame(self):
        if (not self.eProcess.isSet()) : 
            self.do()
        else : pass #正在计算中，或者传入图像==0
        #优先传出之前的结果
        if len(self._frame)>0 :    
            out = self._frame.copy()
            self._frame=[]#.clear()
            ##如果计算完正在等待，并且有图象传入
            return out
        else : return []
        
        
        '''
        if self.eFrame.isSet() : #如果正在计算中
            return [] #并且还没有计算完，返回上一帧
        #out=self.getFrame()
        self.eFrame.set()
        #if self.queResult.qsize()>0 : return self.queResult.get()
        #else : return []
        if len(self._frame)==0 : out = []
        else : out = self._frame.copy()
        self._frame=[]
        return out
        '''
    @property
    #总是至少返回一个数据,返回数据的副本
    def lastFrame(self):
        if self.eFrame.isSet() : #如果正在计算中
            return self._lastFrame.copy() #并且还没有计算完，返回上一帧
        #out=self.getFrame()
        self.eFrame.set()
        #if self.queResult.qsize()>0 : return self.queResult.get()
        #else : return []
        return self._lastFrame.copy()
        
    def destroy(self):
        self.isActive=False
        if self.video_capture!=None : self.video_capture.release()   

class CVCameraThread(DRCameraThread):
    def __init__(self,inStrWinName,inShapeImg,inCamDevice=-1,inControl=None):
        #传入()使得camInit先不执行，在thread里执行
        DRCamera.__init__(self,(),inCamDevice)
        self.shapeImg=inShapeImg
        self._frame=self.frameDefault
        self._lastFrame=self.frameDefault
        #20180418给while主循环用
        self.outterControl=self.nothing
        if inControl!=None : self.outterControl=inControl

        ThreadLoop.__init__(self,name=inStrWinName)
        self.eFrame=self.eProcess
        self.start()
'''     '''   
class CVCamera(DRCamera):
    def __init__(self,inStrWinName,inShapeImg=(),inCamDevice=-1):
        DRCamera.__init__(self,inShapeImg,inCamDevice)
        
    
    def camInit(self,inShapeImg):
        if self.name == 'video' or self.name == 'camera': 
            self.video_capture=cv.VideoCapture(self.devId)
            self.video_capture.set(cv.CAP_PROP_FRAME_WIDTH, inShapeImg[0]); 
            self.video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, inShapeImg[1]);
            #self.video_capture.set(cv.CAP_PROP_SATURATION,0.2);
            print('CVCamera camInit',self.video_capture)
        else : self.video_capture = None
        outFrame=self.getFrame(inShape=inShapeImg,idx=2)
        #准备监控窗口
        #cv.namedWindow(inStrWinName,cv.WINDOW_NORMAL)
        #cv.imshow(self.name,outFrame)
        #设置调整Trackbar
        #cv.createTrackbar('minDist',self.name,int(dataset.minDist*10000),10000,eTrackBar)
        return outFrame#self.name,self.video_capture
        
    def getFrame(self,inShape=None,idx=0,e=None):
        self._timeLast=time.time()
        if inShape==None : inShape= self.shapeImg
        if self.video_capture != None:
            ret, frame = self.video_capture.read()
        else :
            #frame = frame = cv.imread('testHuman'+ str(idx) +'.jpg')
            frame = cv.imread('dog.jpg')
            ret = len(frame)
        if ret :
            if frame.shape != inShape+(3,) : frame=self.getResizePad(frame,self.shapeImg)#frame = cv.resize(frame,self.shapeImg,interpolation=cv.INTER_CUBIC)
        else : 
            frame=self.getResizePad(self.frameDefault,self.shapeImg)
        self._lastFrame=frame.copy()
        return frame
        
        
def cvCameraTest():
    c=CVCameraThread('video',(1024,768),inCamDevice=-1)
    #p=multiprocessing.Pool()
    try:
        while True:
            f=c.frame#只能读取一次
            if len(f)!=0 :cv.imshow(c.name,f)
            if cv.waitKey(1) & 0xFF == ord('q'):break

    except KeyboardInterrupt:
        c.destroy()
        cv.destroyAllWindows()
    finally :
        c.destroy()
        cv.destroyAllWindows()

def funControl(c):
    print('outter funControl')
def loop(c):
    #cv.drImshowInfo('frameLast',c.lastFrame,c.lastFrame.shape,shapeScreen=(800,600))
    cv.drImshowInfo('frameLast',c.lastFrame,c.lastFrame.shape)
    #cv.drImshowInfo('frameOrg',c.orgFrame,c.orgFrame.shape)
    
if __name__ == '__main__':
    print('CVCamera:MainFun is starting..')
    DRCamera.shapeCapture=(1920,1080)
    #用法A
    #cvCameraTest()
    #主循环用法B
    #c=CVCameraThread('video',(227,227),inCamDevice=1,inControl=funControl)
    c=DRCameraThread((1920,1080),inCamDevice=3)
    while c.waitKey(1):loop(c)
    #while c.step:loop(c)
    '''
    a=cv.drColor((180,640),255)
    cv.drImshowInfo('before',a,a.shape)
    a=cv.drResizeCenter(a,(-1320,-320))
    cv.drImshowInfo('after',a,a.shape)
    cv.waitKey(5000)
    cv.destroyAllWindows()
    '''
    pass
