from dr.tf.KNN import KNN
from dr.tf.squeezeNet import *
from tensorflow import *

from tensorflow.core.framework import graph_pb2
import os
__all__=['isGPU','load_frozenmodel','checkNode']


def isGPU(inV):
    if inV==False or inV==-1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        DEVICE = '_CPU'
    else:
        if str(inV)=='True':inV=0
        os.environ['CUDA_VISIBLE_DEVICES'] = str(inV)
        DEVICE = '_GPU'

import tensorflow as tf

# Load frozen Model
def load_frozenmodel(inPath):
    print('> Loading frozen model into memory')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        seg_graph_def = tf.GraphDef()
        #seg_graph_def = graph_pb2.GraphDef()
        with tf.gfile.GFile(inPath, 'rb') as fid:
        #with open(inPath, 'rb') as fid:
            seg_graph_def.ParseFromString(fid.read())
            #print('graph content:',seg_graph_def)
            tf.import_graph_def(seg_graph_def, name='')
    return detection_graph


def checkNode(g,inRange=(0,4000),chkType=('ResizeBilinear'),chkName=('sub'),chkInput=('MobilenetV2/expanded_conv_16/output:0')):
    nodeType={}
    nodeName={}
    strNodeRelation='\n'
    nodeInput={}

    for idx, node in enumerate(g.get_operations()):
        if idx<inRange[0] : continue
        print("       ", idx, node.type, node.name)
        #if node.type.startswith('Cast') : nodeCast.append(str(idx)+':'+node.name)
        if node.type in chkType :
            nodeType.setdefault(node.type,[]).append({'idx':idx,'node':node})
        if node.name in chkName :
            nodeName.setdefault(node.name,[]).append({'idx':idx,'node':node})
        for a in node.inputs:
            if a.name in chkInput :
                strNodeRelation=strNodeRelation+str('->'+str(idx)+':'+node.name)
                nodeInput.setdefault(a.name,[]).append({'idx':idx,'node':node,'relation':strNodeRelation})
            print("           IN:", a.name,a.get_shape())
        for a in node.outputs:
            print("           OUT:", a.name,a.get_shape())

        if idx>inRange[1] : break
    #print('nodeBatchSize:',nodeBatchSize,'  \nCast :',nodeCast,'  \nnodeCheck',nodeCheck,strNodeRelation)
    #for t in enumerate(nodeType):
    for chk in [nodeType,nodeName,nodeInput]:
        for strKey in chk:
            for d in chk[strKey]:
                idx=d['idx']
                node=d['node']
                print('--------------all  :',strKey,'---------------------')
                print("       ", idx, node.type, node.name)
                #if node.type.startswith('Cast') : nodeCast.append(str(idx)+':'+node.name)
                for a in node.inputs:
                    print("           IN:", a.name,a.get_shape())
                for a in node.outputs:
                    print("           OUT:", a.name,a.get_shape())

