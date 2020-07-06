#__all__ = ['CVTools', 'CVMorphology','CVContour','CVCamera']

__all__ = ['drImshowInfoBF','drResizePad','drResizeCenter','drNormalWeights','drArrWeighted','drPutText','drTypeImg','drAddAlpha','drCvtTo','drBlend','drImshowInfo','drMultiImg','drGray','drColor']#,'draw_str', 'typeImg','addAlpha','cvtTo','blend','multiImg','cvImshowInfo','cvGray'
import cv2 as cv
import numpy as np
from dr import *

'''
def drResizeRatio(inImg,inDstSize):
    等比缩放,适应高宽,不加边.
    Args:
        inImg:source
        inDstSize:shape Tuple 
    Return:
        target_size:shape Tuple
        img：np.array
    resize_ratio = 1.0 * min(inDstSize[1],inDstSize[0]) / max(inImg.shape[1],inImg.shape[0])
    target_size = (int(resize_ratio * inImg.shape[1]), int(resize_ratio * inImg.shape[0]))
    return target_size,cv.resize(inImg,target_size)
'''
#resize后需要填充或者裁切
def drResizeCutFill(isCut,inImg,absDstSize,new_size,color=[0, 0, 0]):
    # new_size should be in (width, height) format
    im = cv.resize(inImg, (new_size[0], new_size[1]))

    delta_w = absDstSize[0] - new_size[0]
    delta_h = absDstSize[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    #print(new_size,top, bottom,left, right)
    #color = [0, 0, 0]
    #print(old_size,new_size,absDstSize,ratio,delta_w,delta_h,top, bottom,left,right)
    if isCut : new_im = im[-top:new_size[1]+bottom,-left:new_size[0]+right]#cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT,value=color)
    else : new_im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT,value=color)

    return new_im
#居中等比缩放
def drResizeCenter(inImg,inDstSize,color=[0, 0, 0]):
    absDstSize=(abs(inDstSize[0]),abs(inDstSize[1]))
    old_size = (inImg.shape[1],inImg.shape[0]) # old_size is in (height, width) format
    if absDstSize==old_size : return inImg
    ratioDest=absDstSize[0]/absDstSize[1]#w/h
    ratioImg=old_size[0]/old_size[1]
    if inDstSize[1]>0 and  inDstSize[0]>0: #都为正，按高等比缩放
        isCut=False
        if ratioDest>ratioImg : #目标更扁，则高填满
            desired_size = absDstSize[1]
            ratio = float(desired_size)/old_size[1]
        else :
            desired_size = absDstSize[0]
            ratio = float(desired_size)/old_size[0]
    elif inDstSize[1]<0 or inDstSize[0]<0 : #一个为负，按最小边，最大化裁切
        if inDstSize[1]<0 and inDstSize[0]<0 : #都为负，按最小边，最大化裁切
            isCut=True
            if ratioDest>ratioImg : #目标更扁，则宽填满，高溢出
                desired_size = absDstSize[0]
                ratio = float(desired_size)/old_size[0]
            else :
                desired_size = absDstSize[1]
                ratio = float(desired_size)/old_size[1]
        elif inDstSize[0]<0: #w为负，h为标准，裁切w
            desired_size = absDstSize[1]
            ratio = float(desired_size)/old_size[1]
            if ratioDest>ratioImg : 
                #目标更扁，则h填满，宽不足
                isCut=False
            else : 
                #以h为准，目标更窄，则img需要裁切
                isCut=True
        elif inDstSize[1]<0: #h为负，w为标准，裁切h
            desired_size = absDstSize[0]
            ratio = float(desired_size)/old_size[0]
            if ratioDest>ratioImg : 
                #目标更扁，则w填满，高需要裁切
                isCut=True
            else : 
                #以w为准，目标更窄，则img  高不足
                isCut=False
    new_size = tuple([int(x*ratio) for x in old_size])
    new_im=drResizeCutFill(isCut,inImg,absDstSize,new_size,color)
    return new_im
#居正方形中等比缩放
def drResizePad(inImg,inDstSize):
    absDstSize=np.abs(inDstSize)
    if inDstSize[1]>0 and  inDstSize[0]>0: #都为正，按高等比缩放
        desired_size = absDstSize[1]
        old_size = inImg.shape[:2] # old_size is in (height, width) format
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        
        # new_size should be in (width, height) format
        im = cv.resize(inImg, (new_size[1], new_size[0]))
        
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        
        color = [0, 0, 0]
        new_im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT,
            value=color)
    elif inDstSize[1]<0 or inDstSize[0]<0 : #一个为负，按最小边，最大化裁切
        if inDstSize[1]<0 and inDstSize[0]<0 : #都为负，按最小边，最大化裁切
            desired_size = min(absDstSize[1],absDstSize[0])
        elif inDstSize[1]<0: #w为负，h为标准，裁切w
            desired_size = absDstSize[0]
        elif inDstSize[0]<0: #h为负，w为标准，裁切h
            desired_size = absDstSize[1]
        old_size = inImg.shape[:2] # old_size is in (height, width) format
        ratio = float(desired_size)/min(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        # new_size should be in (width, height) format
        im = cv.resize(inImg, (new_size[1], new_size[0]))
        
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        #print(new_size,top, bottom,left, right)
        color = [0, 0, 0]
        new_im = im[-top:new_size[0]+bottom,-left:new_size[1]+right]#cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT,value=color)
        
    return new_im

def drNormalWeights(inTupWeights):
    mins=min(inTupWeights)
    maxs=max(inTupWeights)
    r=maxs-mins
    res=[mins]*len(inTupWeights)
    res=list(map(lambda x: x[0]-x[1], zip(inTupWeights, res))) 
    res=list(map(lambda x: x[0]/r, zip(res)))
    return res

def drArrWeighted(inTupImg,inTupWeights=[],isNormal=False):
    num=len(inTupImg)
    if len(inTupWeights)==0 : inTupWeights=[1/num]*num
    if isNormal : 
        inTupWeights=drNormalWeights(inTupWeights)#加总占比
        #print(inTupWeights,len(inTupWeights),len(inTupImg))
    if num<1 : return []
    if num<2 : return inTupImg[0]
    res=drGray(inTupImg[0].shape,0)
    resF=res*255.0/255.0
    idx=0
    for i in inTupImg:
        if res.shape !=i.shape : 
            tmpI = cv.resize(i,(res.shape[1],res.shape[0]))
            tmpI= drCvtTo(tmpI,drTypeImg(res.shape))
        else : 
            tmpI=i
        resF=cv.addWeighted(resF,1,(tmpI*255.0/255.0),inTupWeights[idx],0)
        idx+=1
    res = (resF*1).astype(np.uint8)
    return res
#from common import draw_str
def draw_str(dst, target, s):drPutText(dst, s,target)
def drPutText(dst,inStr, pos):
    x, y = pos
    cv.putText(dst, inStr, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, inStr, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
    return dst
def typeImg(inImg):return drTypeImg(inImg)
def drTypeImg(inImg):
    #TOTO：这里要补充判断一下是否有shape属性
    #print(type(inImg)==type(np.ndarray))
    #print(type(inImg)==np.ndarray,type(inImg),type(np.ndarray))
    if type(inImg)==np.ndarray : imgShape=inImg.shape
    else : imgShape=inImg
    if len(imgShape)==3 :
        #rgb,或rgba
        if imgShape[2]==3 :return 'rgb'
        elif imgShape[2]==4 :return 'rgba'
        else : return''
    elif len(imgShape)==2 :return 'gray'
    else :return ''

#inAlpha=[]则添加255alpha通道；或者将inALpha去色后作为alpha通道
def addAlpha(inImg,inAlpha=[]):return drAddAlpha(inImg,inAlpha)
def drAddAlpha(inImg,inAlpha=[]):
    #单纯添加一个1的alpha
    if len(inAlpha)==0 :
        return cv.cvtColor(inImg,cv.COLOR_RGB2RGBA)
    else :
        if typeImg(inImg)=='gray':inImg=cv.cvtColor(inImg,cv.COLOR_GRAY2RGB)
        #改为直接按照inImg的尺寸，对alpha缩放;#TODO::这里应该判断一下尺寸是否一致,
        inAlpha=cv.resize(inAlpha,(inImg.shape[1],inImg.shape[0]))

        #把inAlpha去色作为透明通道添加到img
        if typeImg(inAlpha)=='rgb' : 
            #TODO:这里只取了Blue通道，是错的
            res=cv.cvtColor(inAlpha, cv.COLOR_RGB2GRAY)#inFgImg[:,:,3].astype(np.float)/255
            return np.dstack((inImg, res)).astype(np.uint8)
        #把inAlpha作为透明通道添加到img
        elif typeImg(inAlpha)=='gray' :
            #print(inImg.shape,inAlpha.shape)(240, 320, 3) (480, 640)
            
            return np.dstack((inImg, inAlpha)).astype(np.uint8)
        #把inAlpha的alpha作为透明通道添加到img
        elif typeImg(inAlpha)=='rgba' : 
            return np.dstack((inImg, inAlpha[:,:,:3].astype(np.uint8)).astype(np.uint8))
        else : return []
    
def cvtTo(inImg,strDestType,win=''):return drCvtTo(inImg,strDestType,win)
def drCvtTo(inImg,strDestType,win=''):
    t=typeImg(inImg)
    img=inImg
    if t=='rgb' : 
        if strDestType=='gray' : img = cv.cvtColor(inImg,cv.COLOR_RGB2GRAY)
        if strDestType=='rgb' : img = inImg
        if strDestType=='rgba' : img = cv.cvtColor(inImg,cv.COLOR_RGB2RGBA)
    if t=='gray' : 
        if strDestType=='rgb' : img = cv.cvtColor(inImg,cv.COLOR_GRAY2RGB)
        if strDestType=='gray' : img = inImg
        if strDestType=='rgba' : img = cv.cvtColor(inImg,cv.COLOR_GRAY2RGBA)
    if t=='rgba' : 
        if strDestType=='gra' : img = cv.cvtColor(inImg,cv.COLOR_RGBA2GRAY)
        if strDestType=='rgbay' : img = inImg
        if strDestType=='rgb' : img = cv.cvtColor(inImg,cv.COLOR_RGBA2RGB)
    if win!='' : drImshowInfo(win, img,img.shape)#cv.imshow(win,res)
    return img
#透明通道合并
def blend(inFgImg,inBgImg,win=''):return drBlend(inFgImg,inBgImg,win)
def drBlend(inFgImg,inBgImg,inAlpha=None,win=''):
    #如果是RGB图像，
    if typeImg(inFgImg)=='rgb' : 
        #TODO:这里只取了Blue通道，是错的
        alpha=cv.cvtColor(inFgImg, cv.COLOR_RGB2GRAY)#inFgImg[:,:,3].astype(np.float)/255
        alpha_rgb=cv.cvtColor(alpha, cv.COLOR_GRAY2RGB)
        #alpha = np.dstack((alpha,alpha,alpha)).astype(np.float)
        foreground_rgb=inFgImg
    elif typeImg(inFgImg)=='gray' : 
        alpha_rgb=cv.cvtColor(inFgImg, cv.COLOR_GRAY2RGB)
        #alpha = np.dstack((alpha,alpha,alpha)).astype(np.float)
        foreground_rgb=alpha_rgb#addAlpha(inFgImg,inFgImg)
    elif typeImg(inFgImg)=='rgba' :
        # 通道分离，注意顺序BGR不是RGB
        (B,G,R,A) = cv.split(inFgImg)
        foreground_rgb=cv.merge((B,G,R))
        alpha_rgb=cv.cvtColor(A, cv.COLOR_GRAY2RGB)
    else : return []
    background_rgb=cvtTo(inBgImg,'rgb')#inBgImg.astype(np.uint8)
    alpha_rgb=alpha_rgb/255
    blended = alpha_rgb * foreground_rgb + (1 - alpha_rgb) * background_rgb
    res=blended.astype(np.uint8)
    if win!='' : drImshowInfo(win, res,res.shape)#cv.imshow(win,res)
    return res


def multiImg(inArrImg,win=''):
    row1=np.hstack((inArrImg[0],inArrImg[1]))
    row2=np.hstack((inArrImg[2],inArrImg[3]))
    res=np.vstack((row1,row2))
    if win!='' : drImshowInfo(win, res,res.shape)#cv.imshow(win,res)
    return res    
    #return drMultiImg(inArrImg,win)
def drMultiImg(inMatrixImg,shape,win=''):
    arrCol=[]
    maxNumCol=0
    #先找最宽的是几张图
    for c in inMatrixImg : maxNumCol=max(len(c),maxNumCol)
    
    for c in inMatrixImg :
        cResize=[]
        if len(c)>1 : 
            for element in c :
                if element.shape!=shape : element=cv.resize(element,(shape[1],shape[0]))
                if drTypeImg(element)!=drTypeImg(shape) : element=drCvtTo(element,drTypeImg(shape))
                cResize.append(element)
            #col=np.hstack(cResize)
        else : 
            #col = c
            if drTypeImg(c)!=drTypeImg(shape) : drCvtTo(c,drTypeImg(shape))
            cResize.append(c)
        if len(c)<maxNumCol : 
            for i in range(0,maxNumCol-len(c)) : cResize.append(drGray(shape,0))
        col=np.hstack(cResize)
        arrCol.append(col)
        
    if len(arrCol)>1 : res=np.vstack(arrCol)
    else : res = arrCol[0]
    if win!='' : drImshowInfo(win, res,res.shape)#cv.imshow(win,res)
    return res
DR_CV_FRAMEBUFFER_BG=np.ones((1,1,4),np.uint8)*128
def drGetImgFB(shapeScreen): 
    global DR_CV_FRAMEBUFFER_BG
    if type(DR_CV_FRAMEBUFFER_BG)==np.ndarray : 
        wBF,hBF=DR_CV_FRAMEBUFFER_BG.shape[1],DR_CV_FRAMEBUFFER_BG.shape[0]
    elif type(DR_CV_FRAMEBUFFER_BG)==cv.UMat:
        wBF,hBF=DR_CV_FRAMEBUFFER_BG.get().shape[1],DR_CV_FRAMEBUFFER_BG.get().shape[0]
        
    if (wBF,hBF)!=shapeScreen : 
         DR_CV_FRAMEBUFFER_BG=np.ones((shapeScreen[1],shapeScreen[0],4),np.uint8)*128#cv.drColor(shapeScreen+(4,),(255,128,128,128))
         return DR_CV_FRAMEBUFFER_BG
    else : return False

#用法：
#drImshowInfoBF(cv.cvtColor(c.lastFrame,cv.COLOR_RGB2RGBA),round(time.time()-t,3),(70,70),inPosScreen=(640,360))
def drImshowInfoBF(img,inObj,inPos=(5, 15),inPosScreen=(0,0),shapeScreen=(1280,720),framebuffer='/dev/fb0'):
    #t=time.time()
    fBG=drGetImgFB(shapeScreen)
    with open(framebuffer,'wb+') as f:
        r=f.read(1280*720*4)
        #f.seek(0)
        #f.truncate(1280*720*4)
        #fBG=np.uint8(np.frombuffer(r,dtype=np.uint8))#,count=1280*720*4,offset=1280*720*4))
        #fBG=np.reshape(fBG,(shapeScreen[1],shapeScreen[0],4))
        #f.flush()
        fShow=img
        wImg,hImg=fShow.shape[1],fShow.shape[0]
        wScreen,hScreen=shapeScreen
        wPos,hPos=inPosScreen
        if type(fBG)==np.ndarray :
            #出于速度考虑，局部备份背景
            fBackup=fBG[hPos:hPos+img.shape[0],wPos:wPos+img.shape[1]]
            fBG[hPos:hPos+img.shape[0],wPos:wPos+img.shape[1],0:3]=img
            fShow=drPutText(fShow,str(inObj),inPos)#drPutText(fBG,str(round(time.time()-t,3)),inPos)
            f.seek(wScreen*4*hPos+wPos*4)
            f.write(fShow.tobytes())
            #出于速度考虑，局部恢复背景
            fBG[hPos:hPos+img.shape[0],wPos:wPos+img.shape[1]]=fBackup
        else : 
            #fBG[0:img.shape[0],0:img.shape[1]]=img
            fShow=drPutText(fShow,str(inObj),inPos)
            for y in range(0,hImg):
                yScreen=y+hPos
                #print(wScreen*4*yScreen+wPos*4,r)
                f.seek(wScreen*4*yScreen+wPos*4)
                
                f.write(fShow[y,0:wImg,:].tobytes())
    #drImshowBF(img,shapeScreen,framebuffer)

#不影响原图的情况下在窗口里提示信息
def drImshowInfo(strWin,inImg,inObj,inPos=(5, 15),inPosScreen=(0,0),shapeScreen=(1280,720),framebuffer='/dev/fb0'):
    resShow=inImg.copy()
    drPutText(resShow, str(inObj), inPos)
    if isWin():
        cv.imshow(strWin,resShow)
    else :
        drImshowInfoBF(inImg,strWin+str(inObj),inPos,inPosScreen,shapeScreen,framebuffer)
    return inImg

def cvImshowInfo(strWin,inImg,inObj):return drImshowInfo(strWin,inImg,inObj,(10, 20))

def cvGray(inShape,inV):return drGray(inShape,inV)
def drGray(inShape,inV):
    #inShape是图片，用Like
    if type(inShape)==np.ndarray:
        res=np.ones_like(inShape)*inV
    elif type(inShape)==tuple:#输入是shape
        res=np.ones(inShape,dtype=np.uint8)*inV
    res=np.uint8(res)
    return res
def drColor(inShape,inColor):
    '''可以输入图片或者shape,
        Args:
            inShape可以是图片或者shape
            inColor可以是(r,g,b)或者int
    '''
    #res=drGray(inShape,1)
    if type(inColor)==int : return drGray(inShape,inColor)
    if type(inShape)==np.ndarray:
        if len(inShape.shape)==2:res=inShape#单通道直接用，起到遮照效果
        elif len(inShape.shape)>2 : res=np.ones((inShape.shape[1],inShape.shape[0]),dtype=np.uint8)#多通道只取形状大小shape
        else : return []
    elif type(inShape)==tuple:#输入是shape
        res=np.ones((inShape[1],inShape[0]),dtype=np.uint8)
    else : return []
    if type(inColor)==tuple and len(inColor)>=3 :
        tup=[]
        for c in inColor:
            tup.append(res*c)
    else : return []
        
    res=np.dstack(tup)
    res=np.uint8(res)
    return res
