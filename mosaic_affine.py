#!/usr/bin/env python

'''
Mosaic
====================

Keys
----
ESC - exit
'''


import time
import numpy as np
import cv2



class App:
    def __init__(self, video_src):
        self.detect_interval = 1 # interval for frame move detection
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

    def affine_composition(self, M1, M2):
        add_row = np.array([[0,0,1]])
        M1_ext = np.concatenate((M1, add_row), axis=0)
        M2_ext = np.concatenate((M2, add_row), axis=0)
        M12 = np.dot(M2_ext, M1_ext)
        return M12[:2,:]
        

    def run(self):
        try:
            ret, prev = self.read_and_resize()
        except Exception:
            print("No video")
            cv2.waitKey(0)
            self.close()
        
        # Create a mosaic image
        BIG_SIZE = 3
        
        # Prepare big black window of the proper size for the mosaic
        M_id = np.array([[1., 0., 0.], [0., 1., 0.]])
        M_integrate = M_id.copy()
        # Move to center
        M_integrate[0,2] += prev.shape[1] # Translate to the center on the first axis
        M_integrate[1,2] += prev.shape[0] # Translate to the center on the second axis
        
        mosaic = cv2.warpAffine(prev, M_integrate, (prev.shape[1]*BIG_SIZE, prev.shape[0]*BIG_SIZE))
        
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
            incr_track = 1.
            incr_track_mat = 1.
            if(self.frame_idx % self.detect_interval == 0):
                M_incr = cv2.estimateRigidTransform(curr, prev, False)
                incr_track *= M_incr[0,0]
                print "incr: {}".format(incr_track)
                if M_incr is None:
                    print("Panic")
                    
                else:
                    M_integrate = self.affine_composition(M_incr, M_integrate)
                    incr_track_mat = M_integrate[0,0]
                    print "incm {}".format(incr_track_mat)
                    #print M_integrate[0,0]
                    new_frame = cv2.warpAffine(curr, M_integrate, (curr.shape[1]*BIG_SIZE, curr.shape[0]*BIG_SIZE))
                    
                    mosaic = np.where(new_frame == 0, mosaic, new_frame)
                    cv2.imshow('mosaic', mosaic)
                    prev = curr
            

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
    video_src = "DJI_0002.MP4"
    print(__doc__)
    App(video_src).run()
    #app = App("DJI_0004.MP4")

if __name__ == '__main__':
    main()
