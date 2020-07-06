import platform
import os
__all__=['isWin']

def isWin():
    sysstr = platform.system()
    if(sysstr =="Windows"):
        return True
    elif(sysstr == "Linux"):
        #disp_no = os.getenv('DISPLAY')
        if 'DISPLAY' in os.environ : #Linux 下判断是否Xserver 有 DISPLAY
            return True#os.environ['DISPLAY']
        else : 
            return False
    else:
        return False
