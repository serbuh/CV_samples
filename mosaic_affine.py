#!/usr/bin/env python

'''
Mosaic
====================

Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import time
import numpy as np
import cv2


class App:
    def __init__(self, video_src):
        self.detect_interval = 24 # interval for frame move detection
        self.cam = cv2.VideoCapture(video_src)
        self.frame_orig_width = self.cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        self.frame_orig_height = self.cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        self.frame_count = self.cam.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        self.frame_fps = self.cam.get(cv2.cv.CV_CAP_PROP_FPS)
        
        self.frame_resized_width = int(self.frame_orig_width/8)
        self.frame_resized_height = int(self.frame_orig_height/8)
        
        self.frame_idx = 0
        cv2.namedWindow('video')
        self.vis = None
        
    def print_text_on_frame(self, frame, text):
        y0, dy = 50, 4
        for i, line in enumerate(text.split('\n')):
            y = y0 + i*dy
            cv2.putText(frame, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    def read_and_resize(self):
        ret, frame = self.cam.read()
        if frame is None:
            raise EOFError
        frame = cv2.resize(frame, (self.frame_resized_width, self.frame_resized_height), interpolation = cv2.INTER_CUBIC)
        return ret, frame


    def run(self):
        try:
            ret, prev = self.read_and_resize()
        except Exception:
            print("No video")
            cv2.waitKey(0)
            self.close()
        
        # Create a mosaic image
        BIG_SIZE = 3
        big_size = (prev.shape[0]*BIG_SIZE, prev.shape[1]*BIG_SIZE, prev.shape[2])
        
        # Prepare big black window of the proper size for the mosaic
        M_id = np.array([[1., 0., prev.shape[1]], [0., 1., prev.shape[0]]])
        mosaic = cv2.warpAffine(prev, M_id, (prev.shape[1]*BIG_SIZE, prev.shape[0]*BIG_SIZE))
        first_frame = mosaic.copy()
        #cv2.imshow('mosaic', mosaic)
        #cv2.waitKey(0)
        
        while True:
            time.sleep(0.001)
            # Read frame
            try:
                ret, curr = self.read_and_resize()
            except Exception:
                print("The End")
                cv2.waitKey(0)
                self.close()
                break
            
            # Track transformation
            if(self.frame_idx % self.detect_interval == 0):
                M = cv2.estimateRigidTransform(curr, prev, False)
                #M1 = np.array([[0, 0, 100], [0, 0, 100]])
                if M is None:
                    print("Panic")
                else:
                    M[0,2] += curr.shape[1] # Translate to the center on the first axis
                    M[1,2] += curr.shape[0] # Translate to the center on the second axis
                    warped = cv2.warpAffine(curr, M, (curr.shape[1]*BIG_SIZE, curr.shape[0]*BIG_SIZE))
                    
                    # Now create a mask of logo and create its inverse mask also
                    warped2gray = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
                    ret_warped, warped_mask = cv2.threshold(warped2gray, 10, 255, cv2.THRESH_BINARY)
                    warped_mask_inv = cv2.bitwise_not(warped_mask)
                    
                    mosaic2gray = cv2.cvtColor(mosaic,cv2.COLOR_BGR2GRAY)
                    ret_mosaic, mosaic_mask = cv2.threshold(mosaic2gray, 10, 255, cv2.THRESH_BINARY)
                    mosaic_mask_inv = cv2.bitwise_not(mosaic_mask)
                    
                    merge_mask = cv2.bitwise_or(warped_mask, mosaic_mask)
                    #first_frame = cv2.bitwise_or(warped,mosaic,mask = mask)
                    #cv2.imshow('first_frame', first_frame)
                    #cv2.waitKey(0)
                
                # Take only region of logo from logo image.
                #mosaic = cv2.bitwise_or(warped,mosaic,mask = merge_mask)
                #import pdb; pdb.set_trace()  
                mosaic = np.where(mosaic > 0,mosaic, warped)
                
                cv2.imshow('mosaic', mosaic)
                
                #import pdb; pdb.set_trace()                
                #cv2.waitKey(0)
            
            cv2.imshow('video', curr)

            # Read keyboard
            ch = cv2.waitKey(1) & 0xFF
            if ch == 27:
                self.close()
                break
            
            self.frame_idx += 1

    
    def close(self):
        self.cam.release()
        cv2.destroyAllWindows()

def main():
    video_src = "DJI_0004.MP4"
    print(__doc__)
    App(video_src).run()
    #app = App("DJI_0004.MP4")

if __name__ == '__main__':
    main()
