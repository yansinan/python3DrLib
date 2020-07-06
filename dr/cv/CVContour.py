# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 09:56:45 2018

@author: yansinan
"""
import numpy as np
from dr import cv as cv
from itertools import cycle

#from dr.cv.CVTools import *
#根据boundbox计算中心缩放box,输入一个(x,y,w,h)数组,返回一个（右下x,右下y,左上x,左上y）的数组
def drBoundingRectToRect(inBox,inScale=1):
    box=[0,0,0,0]
    if inScale==0 :return box
    b=inBox
    center=[b[0]+(b[2]/2),b[1]+(b[3]/2)]
    #center=[b[0],b[1]]
    box[0]=int(center[0]+(b[2]*inScale/2))#右上X
    box[1]=int(center[1]+(b[3]*inScale/2))#右上Y
    box[2]=int(center[0]-(b[2]*inScale/2))#左上X
    box[3]=int(center[1]-(b[3]*inScale/2))#左上Y
    return box
    '''
    box[0]=int(b[0]+b[2]*(1+(inScale/2)))#右上X
    box[1]=int(b[1]+b[3]*(1+(inScale/2)))#右上Y
    box[2]=int(box[0]-b[2]*(1+inScale))#左上X
    box[3]=int(box[1]-b[3]*(1+inScale))#左上Y
    '''
#计算Contour中心缩放box,输入一个(x,y,w,h)数组
def drBoundingRect(inContour,inScale=1):
    return drBoundingRectToRect(cv.boundingRect(inContour),inScale)
def drDenoise(img,minArea,maxArea,avgField):
    _, contours0, hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    arrArea=[]
    c_min = []
    imgOrg=img.copy()
    imgShow=cv.drGray(img.shape,0)
    imgNoiseSmall=cv.drGray(img.shape,0)
    imgNoiseMedi=cv.drGray(img.shape,0)
    colorRes=(0,0,0)
    for i in range(len(contours0)):
        #epsilon=0.000001*cv.arcLength(contours0[i],True)#周长
        arrChildContours=contours0[i]#cv.approxPolyDP(contours0[i],epsilon,True)
        #arrChildContours = cv.convexHull(contours0[i])
        area = cv.contourArea(arrChildContours)
        br=cv.boundingRect(arrChildContours)
        arrArea.append(area)
        # 处理掉小的轮廓区域，这个区域的大小自己定义。 
        ''''''
        if(area < maxArea):
            c_min.append(arrChildContours)
            #b=cv.boundingRect(arrChildContours)##returns the minimal up-right bounding rectangle for the specified point set
            box=drBoundingRectToRect(br,0)
            if area > minArea :#是不确定噪点
                colorRes=(int(255*area/maxArea),255,0)#areaRate=area/maxArea
                #box=drBoundingRect(arrChildContours,2)
                boxInner=drBoundingRectToRect(br)
                box=drBoundingRectToRect(br,3)
                roi=imgOrg[box[3]:box[1],box[2]:box[0]]
                roiInner=imgOrg[boxInner[3]:boxInner[1],boxInner[2]:boxInner[0]]#=roiOutter[boxInner[3]-box[3]:boxInner[1]-box[1],boxInner[2]-box[2]:boxInner[0]:box[0]]
                #TODO:扩大的boundingRect有时候坐标会超出！！print(box[3],box[1],box[2],box[0],roi.size)
                if roi.size!=0 :
                    rateOutter=(np.count_nonzero(roi)-np.count_nonzero(roiInner))/roi.size
                    rateOutter=round(rateOutter,3)
                    #rateAll=(np.count_nonzero(roi))/roi.size
                    #rateAvg=np.average(roi)/255
                    #img[c_minBox[i][3]:c_minBox[i][1],c_minBox[i][2]:c_minBox[i][0]]=roi+100
                    cv.rectangle(imgShow, (box[0],box[1]), (box[2],box[3]), (128, 255, 255), thickness = 1)
                    cv.rectangle(imgShow, (boxInner[0],boxInner[1]), (boxInner[2],boxInner[3]), (64, 255, 255), thickness = 1)
                    cv.drPutText(imgShow,str(rateOutter),(box[2],box[3]))
                    #cv.drPutText(imgShow,str(rateAll),(box[2],box[3]+10))
                    #cv.drPutText(imgShow,str(rateAvg),(box[2],box[3]+20))
                    #if np.average(roiOutter)<avgField : colorRes=(0,0,0)
                    if rateOutter<0.2 : 
                        colorRes=(255*int(1-((rateOutter/0.2)**2)),0,0)
                        if rateOutter<0.15 : 
                            colorRes=(255,0,0)
                            #img=imgNoiseMedi#如果确实是独立点，归在small里
                    else : 
                        colorRes=(0,0,0)
                    
                else : 
                    print('roi.size==0',box[3],box[1],box[2],box[0])
                    continue
            # thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。  
                imgNoiseMedi=cv.drawContours(imgNoiseMedi, [arrChildContours], -1, colorRes, thickness=-1)
            else :#超小区域
                colorRes=(255,255,255)
                imgNoiseSmall=cv.drawContours(imgNoiseSmall, [arrChildContours], -1, colorRes, thickness=-1)
                
        else :#最大的体块
            colorRes=(255,255,255)
            imgOrg=cv.drawContours(imgOrg, [arrChildContours], -1, colorRes, thickness=-1)
    '''
    for i in range(len(c_min)) :
        colorRes=(int(255*c_minArea[i]),255,0)
        if sum(c_minBox[i])!=0 :
            roi=img[c_minBox[i][3]:c_minBox[i][1],c_minBox[i][2]:c_minBox[i][0]]
            #img[c_minBox[i][3]:c_minBox[i][1],c_minBox[i][2]:c_minBox[i][0]]=roi+100
            cv.rectangle(imgShow, (c_minBox[i][0],c_minBox[i][1]), (c_minBox[i][2],c_minBox[i][3]), (128, 255, 255), thickness = 1)
            cv.drPutText(imgShow,str(np.average(roi)),(c_minBox[i][2],c_minBox[i][3]))
            if np.average(roi)<avgField : colorRes=(0,0,0)
            else : colorRes=(255,0,0)
        img=cv.drawContours(img, [c_min[i]], -1, colorRes, thickness=-1)
    '''
    imgFill=cv.subtract(imgOrg,imgNoiseSmall)
    imgFill=cv.subtract(imgFill,imgNoiseMedi)
    return imgShow,imgNoiseMedi,imgNoiseSmall,imgFill

#thickness 0 只取数据不画图，>1画线，<0填充并且abs(thickness)画线
def cvContour(inImg,win='',acc=0.005,colorFill=(255,255,255),colorBorder=(255,255,255),thickness=-1,mode=cv.RETR_EXTERNAL,method=cv.CHAIN_APPROX_SIMPLE):
    res=cv.drCvtTo(inImg.copy(),'gray')
    _, contours0, hierarchy = cv.findContours(res,mode,method)
    if thickness!=0:
        # 需要搞一个list给cv2.drawContours()才行！！！！！  
        c_max = []
        arrArea=[]
        h, w = res.shape
        res=cv.drCvtTo(inImg.copy(),'rgb')
        for i in range(len(contours0)):
            epsilon=acc*cv.arcLength(contours0[i],True)#周长
            arrChildContours=cv.approxPolyDP(contours0[i],epsilon,True)
            #arrChildContours = cv.convexHull(contours0[i])
            area = cv.contourArea(arrChildContours)
            arrArea.append(area)
            # 处理掉小的轮廓区域，这个区域的大小自己定义。 
            ''''''
            if(area < (h*w/250)):
                c_min = []  
                c_min.append(arrChildContours)  
                # thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。  
                cv.drawContours(res, c_min, -1, (0,0,0), thickness=-1)  
                continue
            c_max.append(arrChildContours)
        if thickness!=-1 :
            cv.drawContours(res, c_max, -1, colorBorder,thickness =abs(thickness), lineType =cv.LINE_8)
        if thickness<=-1 :
            cv.drawContours(res, c_max, -1, colorFill,thickness =-1, lineType =cv.LINE_8)
    if win!='' : cv.drImshowInfo(win, res,str(res.shape)+'\nthickness'+str(thickness))#cv.imshow(win,res)
    return res,contours0, hierarchy

#thickness 0 只取数据不画图，>1画线，<0填充并且abs(thickness)画线
def cvContourRemove(inImg,win='',acc=0.005,colorFill=(255,255,255),colorBorder=(255,255,255),thickness=-1,mode=cv.RETR_EXTERNAL,method=cv.CHAIN_APPROX_SIMPLE):
    res=cv.drCvtTo(inImg.copy(),'gray')
    _, contours0, hierarchy = cv.findContours(res,mode,method)
    if thickness!=0:
        # 需要搞一个list给cv2.drawContours()才行！！！！！  
        c_max = []
        arrArea=[]
        h, w = res.shape
        res=cv.drCvtTo(inImg.copy(),'rgb')
        for i in range(len(contours0)):
            epsilon=acc*cv.arcLength(contours0[i],True)#周长
            arrChildContours=cv.approxPolyDP(contours0[i],epsilon,True)
            #arrChildContours = cv.convexHull(contours0[i])
            area = cv.contourArea(arrChildContours)
            arrArea.append(area)
            # 处理掉小的轮廓区域，这个区域的大小自己定义。 
            ''''''
            if(area < (h*w/250)):
                c_min = []  
                c_min.append(arrChildContours)  
                # thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。  
                cv.drawContours(res, c_min, -1, (0,0,0), thickness=-1)  
                continue
            c_max.append(arrChildContours)
        if thickness!=-1 :
            cv.drawContours(res, c_max, -1, colorBorder,thickness =abs(thickness), lineType =cv.LINE_8)
        if thickness<=-1 :
            cv.drawContours(res, c_max, -1, colorFill,thickness =-1, lineType =cv.LINE_8)
    if win!='' : cv.drImshowInfo(win, res,str(res.shape)+'\nthickness'+str(thickness))#cv.imshow(win,res)
    return res,contours0, hierarchy