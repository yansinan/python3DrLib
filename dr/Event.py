# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 13:08:43 2018

@author: yansinan
"""
#import multiprocessing as base
#from multiprocessing import Processing as baseClass
import threading as base
from threading import Thread as baseClass

class Event( object ):
    OnStart='EventOnStartBase'
    OnEnd='EventOnEndBase'
    def __init__(self, event_type,inTarget, data=None):#The constructor accepts an event type as string and a custom data
        self._type = event_type
        self._target = inTarget
        self._data = data
    #Returns the event type
    @property
    def type(self):return self._type
    #Returns the data associated to the event
    @property
    def data(self):return self._data
    @property
    def target(self):return self._target

class EventDispatcher(object):#multiprocessing.Process
    """
    Generic event dispatcher which listen and dispatch events
    """
    def __init__(self):
        self._events=dict()

    @property
    def nameClass(self):return str(self.__class__).split("\'")[1]+':'+self.name
    def __del__(self):
        #Remove all listener references at destruction time
        self._events = None
    def has_listener(self,event_type, listener):
        #Return true if listener is register to event_type
        # Check for event type and for the listener
        if event_type in self._events.keys():
            return listener in self._events[ event_type ]
        else:
            return False
    def dispatch(self,inType,inData=None):
        #Dispatch an instance of Event class
        # Dispatch the event to all the associated listeners
        if inType in self._events.keys():
            listeners = self._events[inType]
            for listener in listeners:
                #listener( event )
                listener(Event(inType,self,inData))        
    def dispatchEvent(self,event):
        #Dispatch an instance of Event class
        # Dispatch the event to all the associated listeners
        if event.type in self._events.keys():
            listeners = self._events[ event.type ]
            for listener in listeners:
                listener( event )
    def addEventListener(self,event_type, listener):
        #Add an event listener for an event type
        # Add listener to the event type
        if not self.has_listener( event_type, listener ):
            listeners = self._events.get( event_type, [] )
            listeners.append( listener )
            self._events[ event_type ] = listeners
    def removeEventListener(self,event_type, listener):
        #Remove event listener.
        # Remove the listener from the event type
        if self.has_listener( event_type, listener ):
            listeners = self._events[ event_type ]
            if len( listeners ) == 1:del self._events[ event_type ]# Only this listener remains so remove the key
            else:
                listeners.remove(listener)# Update listeners chain
                self._events[ event_type ] = listeners
    ####################Class Method
    '''
    @staticmethod
    def __del__(self):@s
        #Remove all listener references at destruction time
        EventDispatcher._events = None
    
    @staticmethod
    def has_listener(event_type, listener):
        #Return true if listener is register to event_type
        # Check for event type and for the listener
        if event_type in EventDispatcher._events.keys():
            return listener in EventDispatcher._events[ event_type ]
        else:
            return False
    @staticmethod
    def dispatch_event(event):
        #Dispatch an instance of Event class
        # Dispatch the event to all the associated listeners
        if event.type in EventDispatcher._events.keys():
            listeners = EventDispatcher._events[ event.type ]
            for listener in listeners:
                listener( event )
    @staticmethod
    def add_event_listener(event_type, listener):
        #Add an event listener for an event type
        # Add listener to the event type
        if not EventDispatcher.has_listener( event_type, listener ):
            listeners = EventDispatcher._events.get( event_type, [] )
            listeners.append( listener )
            EventDispatcher._events[ event_type ] = listeners
    @staticmethod
    def remove_event_listener(event_type, listener):
        #Remove event listener.
        # Remove the listener from the event type
        if EventDispatcher.has_listener( event_type, listener ):
            listeners = EventDispatcher._events[ event_type ]
            if len( listeners ) == 1:del EventDispatcher._events[ event_type ]# Only this listener remains so remove the key
            else:
                listeners.remove(listener)# Update listeners chain
                EventDispatcher._events[ event_type ] = listeners
    '''

class ThreadLoop(baseClass):#multiprocessing.Process
    """
    Generic event dispatcher which listen and dispatch events
    """
    #_events = dict()
    def __init__(self,group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        self._isActive=True
        self.eProcess=base.Event()
        self.eDestroy=base.Event()
        #self.queResult = base.Queue()
        #self.queFeed = base.Queue()
        #self.queDestroy = base.Queue()
        baseClass.__init__(self,group=group, target=target, name=name, args=args, kwargs=kwargs,daemon=daemon)
        self.daemon = True#因子进程设置了daemon属性，主进程结束，它们就随着结束了
        #self.join()#设置daemon执行完结束的方法
        #self.start()
        self.init()
    def run(self):
        #self.init()
        while self.isActive :
            self.eProcess.wait()
            if not self.isActive : break
            self.funLoop()
            self.eProcess.clear()
            #if self.queDestroy.qsize()>0 :self._isActive=self.queDestroy.get()
        print('\n'+self.nameClass+self.name+' is not active..wait for destroy!!')
        self.destroy()
        print('\n'+self.nameClass+self.name+' destroyed...')
            
    def do(self):self.eProcess.set()
    def init(self):pass
    def funLoop(self):pass
    def destroy(self):pass
    @property
    def isActive(self):return self._isActive
    @isActive.setter
    def isActive(self,inB):
        #self.queDestroy.put(inB)
        self._isActive=inB
        if not self._isActive : self.eProcess.set()
    @property
    def nameClass(self):return str(self.__class__).split("\'")[1]+':'+super().name
    
'''    
class EventDispatcherThread(ThreadLoop,EventDispatcher):#multiprocessing.Process
    def __init__(self,group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        ThreadLoop.__init__(self,group=group, target=target, name=name, args=args, kwargs=kwargs,daemon=daemon)
        self.name=super().name
        EventDispatcher.__init__(self,self.name)
'''
if __name__ == '__main__':
    pass