# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 17:12:03 2018

@author: yansinan
"""
__all__=['SqzRes','squeezeNetThread','squeezeNet',]
import tensorflow as tf
import numpy as np
import scipy.io as io
#import mat4py as io
import os
import time

#dr
from dr.Event import ThreadLoop

# SqueezeNet v1.1 (signature pool 1/3/5)
########################################
class SqzRes(dict):
    @staticmethod
    def fromDict(inDict):return SqzRes(inDict['img'],inDict['arrData'],inDict['strClassName'])
    def __init__(self,inImg,inArrData,inStrName):
        self['img']=inImg.copy()
        self['arrData']=inArrData
        self['strClassName']=inStrName
    def copy(self):return SqzRes.fromDict(self)
    @property
    def img(self):return self['img']
    @property
    def arrData(self):return self['arrData']
    @property
    def strClassName(self):return self['strClassName']
    
class squeezeNetBase(object):
    #每个Data保留多少数据
    maxSqzPara=1000
    def __init__(self,dataPath="",inImageShape=[1,227,227,3] ,inArrClassName=None):
        if dataPath!="":
            #模型参数集合
            self.modelData = {}
            #平均值？mean_pixel
            self.mean_pixel=0
            self.modelData,self.mean_pixel=self.load_net(dataPath)
        # Building network
        if inImageShape !=None:
            self._shapeImg = (inImageShape[1],inImageShape[2])
        if inArrClassName!=None : self.arrClassName = inArrClassName
        #每个Data保留多少数据
        self.maxSqzPara=1000

        #输入图像占位符
        self.imageHolder={}
        self.keepProbHolder={}
        #tensorflow的graph    #创建一个临时Graph数据流图
        self.g = tf.Graph()
        self.config = tf.ConfigProto(log_device_placement = False)
        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.allocator_type = 'BFC'
        with self.g.as_default():
            #已经创建好的网络
            self.net=self.build(inImageShape)
        #self.sess = tf.Session(graph = self.g)
        
    def predict(self,inImage,inRefineDataKeep=0.25,inLayerName='classifier_actv'):return {}#SqzRes.fromDict({'img':[],'arrData':[],'strClassName':''})    

    #构建网络
    def build(self,inImageShape,inKeepProb=1.):
        self.imageHolder = tf.placeholder(dtype=self.get_dtype_tf(), shape=inImageShape, name="image_placeholder")
        self.keepProbHolder = tf.placeholder(self.get_dtype_tf())
        net= self.net_preloaded('max', True)
        return net
    
    def load_net(self,data_path):
        if not os.path.isfile(data_path):
            #parser.error("Network %s does not exist. (Did you forget to download it?)" % data_path)
            print("Network %s does not exist. (Did you forget to download it?)" % data_path)
            return
    
        weights_raw = io.loadmat(data_path)
        #print(weights_raw)
        # Converting to needed type
        #conv_time = time.time()
        for name in weights_raw:
            self.modelData[name] = []
            # skipping '__version__', '__header__', '__globals__'
            if name[0:2] != '__':
                kernels, bias = weights_raw[name][0]
                self.modelData[name].append( kernels.astype(self.get_dtype_np()) )
                self.modelData[name].append( bias.astype(self.get_dtype_np()) )
        #print("Converted network data(%s): %fs" % (self.get_dtype_np(), time.time() - conv_time))
        
        self.mean_pixel= np.array([104.006, 116.669, 122.679], dtype=self.get_dtype_np())
        return self.modelData, self.mean_pixel

    def net_preloaded(self, pooling, needs_classifier=False):
        net = {}
        #cr_time = time.time()
    
        x = tf.cast(self.imageHolder, self.get_dtype_tf())
    
        # Feature extractor
        #####################
        
        # conv1 cluster
        layer_name = 'conv1'
        weights, biases = self.get_weights_biases(layer_name)
        x = self._conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID', stride=(2, 2))
        x = self._act_layer(net, layer_name + '_actv', x)
        x = self._pool_layer(net, 'pool1_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')
    
        # fire2 + fire3 clusters
        x = self.fire_cluster(net, x, cluster_name='fire2')
        #fire2_bypass = x
        x = self.fire_cluster(net, x, cluster_name='fire3')
        x = self._pool_layer(net, 'pool3_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')
    
        # fire4 + fire5 clusters
        x = self.fire_cluster(net, x, cluster_name='fire4')
        #fire4_bypass = x
        x = self.fire_cluster(net, x, cluster_name='fire5')
        x = self._pool_layer(net, 'pool5_pool', x, pooling, size=(3, 3), stride=(2, 2), padding='VALID')
    
        # remainder (no pooling)
        x = self.fire_cluster(net, x, cluster_name='fire6')
        #fire6_bypass = x
        x = self.fire_cluster(net, x, cluster_name='fire7')
        x = self.fire_cluster(net, x, cluster_name='fire8')
        x = self.fire_cluster(net, x, cluster_name='fire9')
        
        # Classifier
        #####################
        if needs_classifier == True:
            # Dropout [use value of 50% when training]
            x = tf.nn.dropout(x, self.keepProbHolder)
        
            # Fixed global avg pool/softmax classifier:
            # [227, 227, 3] -> 1000 classes
            layer_name = 'conv10'
            weights, biases = self.get_weights_biases(layer_name)
            x = self._conv_layer(net, layer_name + '_conv', x, weights, biases)
            x = self._act_layer(net, layer_name + '_actv', x)
            
            # Global Average Pooling
            x = tf.nn.avg_pool(x, ksize=(1, 13, 13, 1), strides=(1, 1, 1, 1), padding='VALID')
            net['classifier_pool'] = x
            
            #softmax值归一化，输入输出结构没变
            x = tf.nn.softmax(x)
            net['classifier_actv'] = x
        
        #print("Network instance created: %fs" % (time.time() - cr_time))
        self.net=net
        return self.net
    
    def fire_cluster(self,net, x, cluster_name):
        # central - squeeze
        layer_name = cluster_name + '/squeeze1x1'
        weights, biases = self.get_weights_biases(layer_name)
        x = self._conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID')
        x = self._act_layer(net, layer_name + '_actv', x)
        
        # left - expand 1x1
        layer_name = cluster_name + '/expand1x1'
        weights, biases = self.get_weights_biases(layer_name)
        x_l = self._conv_layer(net, layer_name + '_conv', x, weights, biases, padding='VALID')
        x_l = self._act_layer(net, layer_name + '_actv', x_l)
    
        # right - expand 3x3
        layer_name = cluster_name + '/expand3x3'
        weights, biases = self.get_weights_biases(layer_name)
        x_r = self._conv_layer(net, layer_name + '_conv', x, weights, biases, padding='SAME')
        x_r = self._act_layer(net, layer_name + '_actv', x_r)
        
        # concatenate expand 1x1 (left) and expand 3x3 (right)
        x = tf.concat([x_l, x_r], 3)
        net[cluster_name + '/concat_conc'] = x
        
        return x
    
        
    def _conv_layer(self,net, name, input, weights, bias, padding='SAME', stride=(1, 1)):
        conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, stride[0], stride[1], 1),
                padding=padding)
        x = tf.nn.bias_add(conv, bias)
        net[name] = x
        return x
    
    def _act_layer(self,net, name, input):
        x = tf.nn.relu(input)
        net[name] = x
        return x
        
    def _pool_layer(self,net, name, input, pooling, size=(2, 2), stride=(3, 3), padding='SAME'):
        if pooling == 'avg':
            x = tf.nn.avg_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                    padding=padding)
        else:
            x = tf.nn.max_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                    padding=padding)
        net[name] = x
        return x
    def get_weights_biases(self, layer_name):
        weights, biases = self.modelData[layer_name]
        biases = biases.reshape(-1)
        return (weights, biases)
    
    #####TOOLS
    def get_dtype_tf(self):
        return tf.float32
    def get_dtype_np(self):
        return np.float32
    @property
    def shapeResult(self):return (self.maxSqzPara,)
    @property
    def shapeImg(self):return self._shapeImg
    
    
    def destroy(self):
        self.sess.close()
    
    
class squeezeNet(squeezeNetBase):
    #分类名称
    #arrClassName=np.arange(0,1000,dtype=np.int)
    def __init__(self,dataPath="",inImageShape=[1,227,227,3] ,inArrClassName=None):
        #对原始数据做阈值处理，保留个数百分比
        #self.refineDataKeep=0.25
        squeezeNetBase.__init__(self,dataPath,inImageShape ,inArrClassName)
        self.sess = tf.Session(graph = self.g,config=self.config)

    #predictImage预测某张图片.Reture SqzRes
    def predict(self,inImage,inRefineDataKeep=0.25,inLayerName='classifier_actv'):
        if len(inImage)==0 : return SqzRes([],[],'')
        with self.sess.as_default() :
            sqznet_results=self.net['classifier_actv'].eval(feed_dict={self.imageHolder: [self.preprocess(inImage)], self.keepProbHolder: 1.})[0][0][0]
            #sqznet_results=self.net['fire9/concat_conc'].eval(feed_dict={self.imageHolder: [self.preprocess(inImage)], self.keepProbHolder: 1.})
            #把1000个类别减少到KNN.maxSqzPara，并且调优
            reduceData,orgData=squeezeNet.reduceSqzPara(sqznet_results,inRefineDataKeep)
            #print(reduceData)
            strClassName=self.getClassName(sqznet_results)
        return SqzRes.fromDict({'img':inImage.copy(),'arrData':reduceData,'strClassName':strClassName})
    #输入227,227的图片，这里不负责缩放
    #def setImage(self,self._shapeImg):
        # Loading image
        #img_content, orig_shape = imread_resize('dog.jpg')
        
        #np.empty([1,227,227,3]).shape#inImage.shape#(1,) + img_content.shape
        
    def getClassName(self,inRes):return (self.arrClassName[np.argmax(inRes)])
    #转换颜色通道
    def preprocess(self,image):
        swap_img = np.array(image)
        img_out = np.array(swap_img)
        img_out[:, :, 0] = swap_img[:, :, 2]
        img_out[:, :, 2] = swap_img[:, :, 0]
        return img_out - self.mean_pixel
    #转换颜色通道
    def unprocess(self,image):
        swap_img = np.array(image + self.mean_pixel)
        img_out = np.array(swap_img)
        img_out[:, :, 0] = swap_img[:, :, 2]
        img_out[:, :, 2] = swap_img[:, :, 0]
        return img_out
    ##############静态类
    #调整squeeznet网络参数参考数量,out:处理后数据,原始数据
    @staticmethod
    def reduceSqzPara(inData,inKeepPercent=0.25):
        #把1000个类别减少到KNN.maxSqzPara
        inData=inData[0:squeezeNetBase.maxSqzPara]
        orgData=inData.copy()
        #data调优，只把几率最大的留下,让较小的75%=0，留下25%最大的。留下的越少越准确，但适用宽度越窄
        #inData=KNN.getRefineArrData(inData,0.25)
        inData[np.argsort(inData)[:np.int(np.floor(squeezeNet.maxSqzPara*(1-inKeepPercent)))]]=0

        return inData,orgData
        
    #获取优化数据的功能函数,二维函数，给外部用.结果：保留最大的inKeepProb%值，剩余变为0，并保持索引不变
    @staticmethod
    def getRefineArrData(inArrData,inKeepPercent=1):
        arrIdxLow=np.argsort(inArrData,axis=1)[:,:np.int(np.floor(inArrData.shape[1]*(1-inKeepPercent)))]
        #print('idxSorted::\n',np.argsort(self.arrData,axis=1))
        #arrData[i,j]的原理:i,j的每一个对应位置两个值作为arrData的二维索引，取值
        maskIdxLow=np.ones(arrIdxLow.shape,dtype=np.int)*np.arange(0,arrIdxLow.shape[0],1)[:,np.newaxis]
        
        #print('sliceIdxArray::\n',arrIdxLow,maskIdxLow)
        #print('self.arrData 操作',self.arrData[maskIdxLow,arrIdxLow])
        out=inArrData.copy()
        out[maskIdxLow,arrIdxLow]=0
        return out
    
    
class squeezeNetThread(squeezeNet,ThreadLoop):
    def __init__(self,dataPath="",inImageShape=[1,227,227,3] ,inArrClassName=None):
        squeezeNet.__init__(self,dataPath,inImageShape ,inArrClassName)
        ThreadLoop.__init__(self)

        self.tmpframe=[]
        self.tmpRefineDataKeep=1
        self.tmpLayerName='classifier_actv'
        self._tmpRes=[]#结果形同{'img':图片,'arrData':reduceData,'strClassName':strClassName}
        self._tmpResLast=SqzRes([],[],'')
        self.start()
        #Test
        #self.tLast=time.time()
        self.tCost=0
    def funLoop(self):
        tLast=time.time()
        self._tmpRes=super().predict(self.tmpframe,self.tmpRefineDataKeep,inLayerName=self.tmpLayerName)
        #print('sqz timeCost::',time.time()-self.tLast,time.time())
        self.tCost=time.time()-tLast
    #,返回数据的副本
    def predict(self,inF,inRefineDataKeep=1,inLayerName='classifier_actv'):
        #print('sqz isSet()',time.time())
        #优先传出之前的结果
        if (not self.eProcess.isSet()) and len(inF)>0: 
            self.tmpframe=inF
            self.tmpRefineDataKeep=inRefineDataKeep
            self.tmpLayerName=inLayerName
            self.do()
        else : pass #正在计算中，或者传入图像==0
        if len(self._tmpRes)>0 :    
            out = self._tmpRes.copy()
            self._tmpResLast=out
            self._tmpRes.clear()
            ##如果计算完正在等待，并且有图象传入
            return out,self._tmpResLast
        else : return SqzRes([],[],''),self._tmpResLast#SqzRes([],[],'')
if __name__ == '__main__':
    pass
