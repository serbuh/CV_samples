#!/usr/bin/env python

'''
Optical Flow - Lucas-Kanade for sparse feature set
====================

Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2



class App:
    def __init__(self, video_src):
        self.cam = cv2.VideoCapture(video_src)
        self.frame_width = self.cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        self.frame_count = self.cam.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        

    def draw_flow(self, img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis
    
    
    def draw_hsv(self, flow):
        h, w = flow.shape[:2]
        fx, fy = flow[:,:,0], flow[:,:,1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx+fy*fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[...,0] = ang*(180/np.pi/2)
        hsv[...,1] = 255
        hsv[...,2] = np.minimum(v*4, 255)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr
    

    def warp_flow(self, img, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        return res
        
    def read_and_resize(self):
        ret, frame = self.cam.read()   
        frame = cv2.resize(frame,(int(self.frame_width/4),int(self.frame_height/4)), interpolation = cv2.INTER_CUBIC)        
        return ret, frame
        

    def run(self):
        ret, prev = self.read_and_resize()
        
        prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        show_hsv = False
        show_glitch = False
        cur_glitch = prev.copy()        
        
        while True:
            ret, img = self.read_and_resize()            
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
            prevgray = gray
    
            cv2.imshow('flow', self.draw_flow(gray, flow))
            if show_hsv:
                cv2.imshow('flow HSV', self.draw_hsv(flow))
            if show_glitch:
                cur_glitch = self.warp_flow(cur_glitch, flow)
                cv2.imshow('glitch', cur_glitch)
    
            ch = cv2.waitKey(5) & 0xFF
            if ch == 27:
                break
            if ch == ord('1'):
                show_hsv = not show_hsv
                print('HSV flow visualization is', ['off', 'on'][show_hsv])
            if ch == ord('2'):
                show_glitch = not show_glitch
                if show_glitch:
                    cur_glitch = img.copy()
                print('glitch is', ['off', 'on'][show_glitch])
        self.cam.release()
        cv2.destroyAllWindows()
        print("Bye!")


def main():
    video_src = "DJI_0004.MP4"
    print(__doc__)
    App(video_src).run()

if __name__ == '__main__':
    main()
