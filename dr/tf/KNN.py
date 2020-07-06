# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 21:53:35 2018

@author: yansinan
"""
import tensorflow as tf
import numpy as np
    
class KNN:
    def __init__(self,inSampleShape):
        #对原始数据做阈值处理，保留个数百分比
        #self.maxSqzPara=inMaxSqzPara
        self.g = tf.Graph()
        self.sess=tf.Session(graph = self.g)

        self.reset(inSampleShape)
        with self.g.as_default():
            self.arrDistance = tf.sqrt(tf.reduce_sum(tf.pow(tf.add(self.xtr,tf.negative(self.xte)),2),reduction_indices=1))
            self.pred = tf.argmin(self.arrDistance,0)  

            diff=tf.add(self.xtr,tf.negative(self.xte))
            percent = tf.div(0.1,tf.add(0.1,tf.abs(diff)))
            self.arrPercent = tf.div(tf.reduce_sum(percent,reduction_indices=1),self.shapeSample[0])
            self.predPercent = tf.argmax(self.arrPercent,0)  
            
            #self.arrSoftmax=tf.nn.softmax(tf.multiply(self.arrDistance,-100.0))
            self.arrSoftmax = tf.nn.softmax(tf.multiply(tf.div(0.1,tf.add(0.1,self.arrDistance)),100))
            
            
    def addSample(self,inData,inClassId):
        self.arrData=np.vstack((self.arrData,inData[np.newaxis,:]))
        self.arrDataClassId = np.append(self.arrDataClassId,inClassId)
    def appendSamples(self,inArrData,inClassId):
        
        self.arrData=np.vstack((self.arrData,inArrData))
        
        self.arrDataClassId = np.append(self.arrDataClassId,inClassId*np.ones(inArrData.shape[0],dtype=np.int))

    def reset(self,inSampleShape):
        self.shapeSample=inSampleShape
        #self.arrData=np.zeros(((0,)+self.shapeSample))
        self.arrData=np.zeros(((0,)+self.shapeSample))
        self.arrDataClassId=np.zeros((0,),dtype = np.int)
        with self.g.as_default():
            self.xtr = tf.placeholder(tf.float32,shape=((None,)+self.shapeSample))  
            self.xte = tf.placeholder(tf.float32,shape=(self.shapeSample))
            self.sess.run(tf.global_variables_initializer())

                
    def run(self,inData,inMode='dist'):
        #如果没有数据就不用计算了
        if self.arrData.size<=0 : return -1,-1,-1
        #idx,arrDistance = self.sess.run([self.pred,self.distance],{self.xtr:self.dSamples['data'] , self.xte:reduceData})
        if inMode=='dist' : tmpRun=[self.pred,self.arrDistance,self.arrSoftmax]
        elif inMode == 'percent' : tmpRun=[self.predPercent,self.arrPercent,self.arrSoftmax]
        elif inMode == 'softmax' : tmpRun=[self.predSoftmax,self.arrSoftmax,self.arrSoftmax]
        idx,arr,arrAcc = self.sess.run(tmpRun,{self.xtr:self.arrData, self.xte:inData})
        return idx,arr,arrAcc
    
    #inArr是一维数组，保存了需要垂直合并的数据，每个数组必须维度一致.默认按第一个元素排序
    def knn(self,arrDistance,tupOthers,inRange=0):
        #流程：排序->截取->计算
        if len(arrDistance) == len(self.arrDataClassId) and  len(arrDistance)>0:
            #距离排序，
            newOrder=np.argsort(arrDistance)
            numArrToCombine=len(tupOthers)
            if numArrToCombine>0 :
                tupCombine=(arrDistance.copy()[newOrder],self.arrDataClassId.copy()[newOrder])
                for arr in tupOthers:
                    tupCombine=tupCombine+(arr.copy()[newOrder],)
                #tupleCombine=()
        else :
            print('error:KNN:knn:arrDistance is not the same size of self.arrDataClassId,or empty!!Details:\n',arrDistance,self.arrDataClassId)
            return -1,-1
        #把距离和类别(第一行距离，第二行类别，第三行softmax后的acc)对应成三行数组
        
        arrClassResAll= np.vstack(tupCombine)
        #截取最小数量,[[距离],[classId],[softmax百分比]
        if inRange==0 : inRange=self.getKNNRange(self.arrDataClassId)
        arrClassResRange=arrClassResAll[:,0:inRange]#3xn
        #前KNNRange范围内，统计各类别出现次数int
        arrBincountKNNRangeClass=np.bincount(arrClassResRange[1].astype(np.int))#1*KNNRange
        #print(self.arrData.shape,arrBincountKNNRangeClass)

        return arrClassResRange,arrBincountKNNRangeClass
    #因为数组数量不同，用切片的方式合并
    def getAllClassState(self,inBincountPartClass):
        #求所有类别的百分比,统计百分比
        arrBincountAllClass=np.bincount(self.arrDataClassId,weights=np.zeros((self.arrDataClassId.shape)))#shape(n,)
        
        #print('\n',arrClassResRange[1].astype(np.int32))
        arrBincountRangeClass=inBincountPartClass
        #因为数组数量不同，用切片的方式合并
        arrProb=(arrBincountAllClass[:arrBincountRangeClass.size]+arrBincountRangeClass)
        arrProb = np.append(arrProb,arrBincountAllClass[arrBincountRangeClass.size:])
        arrProb = np.around(arrProb,4)
        return arrProb
    #获取最近邻范围:所有类别里采样数量最小的作为范围
    def getKNNRange(self,inArrDataClassId):
        arrBincount=np.bincount(inArrDataClassId)
        #print('bincount::',arrBincount)
        arrBinCountNonzero=arrBincount[np.nonzero(arrBincount)]
        #print('非零的::',arrBinCountNonzero)
        arrIdNonzero = np.nonzero(arrBincount)[0]#因为nonzero输出是tuple  取第一个元素，类似output::非零的Class:: (array([1, 2, 3], dtype=int64),)
        #print('非零的id::',arrIdNonzero)
        #newOrder=np.argsort(-arrBincount)
        #print('出现次数多到少排序::',newOrder)
        idLeast=arrIdNonzero[np.argmin(arrBinCountNonzero)]
        #print('出现次数非零且最少的是几号::',idLeast)#np.argmin(out)
        cntLeast=arrBincount[idLeast]
        #print('回到最初的bincount数组，出现次数非零且最少的是几次::',cntList)#np.argmin(out)
        return cntLeast
    
    def softmax(self,inData,scale=1):
        tmpG=tf.Graph()
        with tmpG.as_default():
            xte = tf.placeholder(tf.float32,shape=inData.shape)
            #hldScale 
            #arrOut = tf.nn.softmax(tf.multiply(xte,10))
            arrOut = tf.nn.softmax(tf.multiply(tf.div(0.1,tf.add(0.1,xte)),100))
            #arrOut = tf.nn.softmax(xte)
            sess=tf.Session(graph = tmpG)
            sess.run(tf.global_variables_initializer())
            arrOut=sess.run([arrOut],{xte:inData})[0]
            sess.close()
            return arrOut

    def destroy(self):
        if self.sess!=None : self.sess.close()

if __name__ == '__main__':
    tmp=0