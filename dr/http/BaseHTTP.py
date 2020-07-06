# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 11:54:47 2018

@author: yansinan
"""
__all__ = ['CVBaseRequestHandler','ThreadLoopHTTPServer']

import numpy as np

from http.server import BaseHTTPRequestHandler,HTTPServer
from socketserver import ThreadingMixIn
from cgi import parse_header, parse_multipart #post方法用到
from urllib.parse import parse_qs
from os import sep, curdir

import threading

from dr.Event import EventDispatcher
from dr.Event import ThreadLoop
import cv2 as cv

class CVBaseRequestHandler(BaseHTTPRequestHandler,EventDispatcher):
    isActive=True
    dictEvent={}
    strControl=''
    #mimeType
    #mime = {"html":"text/html", "css":"text/css", "png":"image/png"}
    #释放所有事件
    @staticmethod
    def destroy():
        for eKey in CVBaseRequestHandler.dictEvent.keys():
            arrE=CVBaseRequestHandler.dictEvent[eKey]
            for e in arrE:
                e.set()
    
    @staticmethod#判断某个事件里是有已有event
    def hasEvent(inEName,inE=None):
        if inE==None:
            if inEName in CVBaseRequestHandler.dictEvent.keys():#以注册过事件
                return CVBaseRequestHandler.dictEvent[inEName]
            else: return False
        else :
            if inEName in CVBaseRequestHandler.dictEvent.keys():#以注册过事件
                if inE in CVBaseRequestHandler.dictEvent[inEName]:#并且有这个Event
                    return inE
                else:#有事件，没Event，返回
                    return False
            else: return False
    
    @staticmethod#没有不添加，返回最后一个
    def setDefaultEvent(inEName,inE=None):
        if inEName in CVBaseRequestHandler.dictEvent.keys():#以注册过事件
            e=CVBaseRequestHandler.dictEvent[inEName][-1]
        else:#未注册过
            if inE==None:e=threading.Event()
            else : e=inE
            CVBaseRequestHandler.dictEvent.setdefault(inEName,[e])
        #print('CVBaseRequestHandler.setDefaultEvent',CVBaseRequestHandler.dictEvent)
        return e
    @staticmethod#已有也添加，返回新添加的
    def addEvent(inEName,inE=None):
        if inE==None:e=threading.Event()
        else : e=inE
        if inEName in CVBaseRequestHandler.dictEvent.keys():#以注册过事件
            CVBaseRequestHandler.dictEvent[inEName].append(e)
        else:#未注册过
            CVBaseRequestHandler.dictEvent.setdefault(inEName,[e])
        #print('CVBaseRequestHandler.addEvent',CVBaseRequestHandler.dictEvent)
        return e
    @staticmethod
    def dispatchEvent(inEName):
        if inEName in CVBaseRequestHandler.dictEvent.keys():#以注册过事件
            for e in CVBaseRequestHandler.dictEvent[inEName]:
                e.set()
        else:#未注册过
            #print('CVBaseRequestHandler.dispatchEvent no ',inEName,'Registed!')
            return
    def sendText(self,inS):
        self.send_header('Content-type','text/html')
        #self.send_header('Content-length',str(len(inS)))
        self.end_headers()
        self.wfile.write(bytes(inS,encoding='utf8'))
        #self.wfile.write(bytes("\n--jpgboundary\n",encoding='utf8'))
    def sendEventSource(self,inStr):
        self.wfile.write(bytes('data:'+inStr+'\n\n',encoding='utf8'))#+str(idRequestClass)
    def parse_POST(self):
        ctype, pdict = parse_header(self.headers['content-type'])
        if ctype == 'multipart/form-data':
            postvars = parse_multipart(self.rfile, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers['content-length'])
            postvars = parse_qs(self.rfile.read(length).decode('utf-8'), keep_blank_values=True)
        else:
            postvars = {}
        return postvars
    def sendFile(self,inPath):
        self.path=inPath
        try:
            strPath=curdir + sep +'https'+ self.path
            print('DR:load File URL:',strPath,'\n')
            reply=False
            if self.path.endswith('.html'):
                reply = True
                mimeType = 'text/html'
            if self.path.endswith('.css'):
                reply = True
                mimeType = 'text/css'
            if self.path.endswith('.js'):
                reply = True
                mimeType = 'application/javascript'
            if(reply == True):
            #if self.path.endswith('.html') or self.path == '/':
                #self.path = 'https\cam.html'
                self.send_response(200)
                #self.send_header('Content-type','text/html')
                self.send_header('Content-type',mimeType)
                self.end_headers()
                #self.wfile.write(bytes('<html><head><meta http-equiv="content-type" content="text/html; charset=utf-8"></head><body>',encoding='utf8'))
                #self.wfile.write(bytes('<img src="cam.mjpg"/>',encoding='utf8'))
                #self.wfile.write(bytes('</body></html>',encoding='utf8'))
                with open(strPath,'r',encoding='utf-8') as f:
                    self.wfile.write(bytes(f.read(),encoding='utf8'))
                return
            #图片类型的数据不用转二进制
            if self.path.endswith('.jpg'):
                reply = True
                mimeType = 'image.jpg'
            if self.path.endswith('.png'):
                reply = True
                mimeType = 'image/png'
            if self.path.endswith('.ico'):
                reply = True
                mimeType = 'image/ico'
            if self.path.endswith('.gif'):
                reply = True
                mimeType = 'image/gif'
            if(reply == True):
                self.send_response(200)
                self.send_header('Content-type',mimeType)
                self.end_headers()
                with open(strPath,'rb') as f:
                    self.wfile.write(f.read())
                return
        except IOError:
            #print('#############################curdir + sep + self.path::',curdir,sep,self.path)
            self.send_error(404, 'Not Found File %s' %self.path);
    def sendJPG(self,img):   
        bJpg=cv.imencode('.jpg',img)[1].tobytes()
        self.wfile.write(bJpg)
    
    def sendMJPG_header(self,inName='jpgboundary',imgSize=227*227*8*3):
        self.send_response(200)
        #self.send_header('Cache-Control','no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0')
        #self.send_header('Connection','close')
        self.send_header('Content-type','multipart/x-mixed-replace;boundary='+inName)
        #self.send_header('Pragma','no-cache')
        self.end_headers()
        #self.wfile.write(bytes('--'+inName,encoding='utf8'))#--boundary
        #self.send_header('Content-Length',imgSize)#如果没有长度，会导致POST以后不更新了，有冲突？？
        #self.send_header('Content-Type','image/jpeg')
        #self.end_headers()
        
    def sendMJPG(self,img,inName='jpgboundary',isSingleFrame=False):

        #bJpg=cv.imencode('.jpg',img)[1].tobytes()
        #img=np.ones((227,227,3),dtype=np.uint8)*14#dAction[self.path]().copy()#
        #bJpg=cv.imencode('.jpg',img)[1].tobytes()
        self.wfile.write(bytes('--'+inName,encoding='utf8'))#--boundary
        self.send_header('Content-Length',str(img.size*3*8))#如果没有长度，会导致POST以后不更新了，有冲突？？
        self.send_header('Content-Type','image/jpeg')
        self.end_headers()
        self.sendJPG(img)#发送上一次的体
        if isSingleFrame :
            bImg=np.zeros((1,1,3),dtype=np.uint8)
            self.wfile.write(bytes('--'+inName,encoding='utf8'))#--boundary
            self.send_header('Content-Length',str(bImg.size*3*8))#如果没有长度，会导致POST以后不更新了，有冲突？？
            self.send_header('Content-Type','image/jpeg')
            self.end_headers()
            self.sendJPG(bImg)#发送上一次的体
        #self._headers_buffer = []
        #self._headers_buffer.append(("\r\n").encode('latin-1', 'strict'))#必须的
        #self._headers_buffer.append(("--"+strBoundary).encode('latin-1', 'strict'))#发现下一个报文边界时，就认为当前数据块（文档）已经结束
        #self.send_header('Content-Length',str(len(img*8)))#如果没有长度，会导致POST以后不更新了，有冲突？？
        #self.send_header('Content-Type','image/jpeg')
        #self.end_headers()
        '''
        #self._headers_buffer.append(bJpg)
        #self._headers_buffer.append(("--"+inName+"\r\n").encode('latin-1', 'strict'))#发现下一个报文边界时，就认为当前数据块（文档）已经结束
        #self._headers_buffer.append(("%s: %s\r\n" % ('Content-Length', str(len(bJpg)))).encode('latin-1', 'strict'))
        #self._headers_buffer.append(("%s: %s\r\n" % ('Content-Type','image/jpeg')).encode('latin-1', 'strict'))#发现“Content-type”头标或到达头标结束处时，浏览器窗口中的前一个文档被清除，并开始显示下一个文档
        #self._headers_buffer.append(("\r\n").encode('latin-1', 'strict'))#必须的
        #self.flush_headers()
        '''
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
	"""Handle requests in a separate thread."""

class ThreadLoopHTTPServer(ThreadLoop):
    
    def __init__(self,inAddress=('',8080),inRequestHandler=None):#('192.168.1.58', 8080),CamHandler
        if inRequestHandler==None : self.requestHandler = CVBaseRequestHandler
        self.server = ThreadedHTTPServer(inAddress, self.requestHandler)
        ThreadLoop.__init__(self)
        self.start()
        self.do()
        
    def funLoop(self):
        print("server started")
        self.server.serve_forever()
        self._isActive=False
    @property
    def isActive(self):return self._isActive
    @isActive.setter
    def isActive(self,inB):
        self.server.shutdown()
        self._isActive=inB
        if not self._isActive : self.eProcess.set()
    def destroy(self):
        self.isActive=False
        self.requestHandler.isActive=False
        self.requestHandler.destroy()
        self.server.socket.close()
        self.server.server_close()
