#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 15
        self.detect_interval = 5 # interval for frame move detection
        self.tracks = [] # list of descriptors which are tracked
        self.cam = cv2.VideoCapture(video_src)
        self.frame_orig_width = self.cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        self.frame_orig_height = self.cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        self.frame_count = self.cam.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        
        self.frame_resized_width = int(self.frame_orig_width/1)
        self.frame_resized_height = int(self.frame_orig_height/1)
        
        self.frame_idx = 0
        cv2.namedWindow('video')
        cv2.setMouseCallback('video', self.onmouse)
        self.click = None
        self.roi = dict (x1 = 0,
                         x2 = 0,
                         y1 = 0,
                         y2 = 0)
        self.roi_delta = 15
        self.new_roi = False # flag for the new roi from the mouse

    def onmouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.click = (x, y)
            self.roi["x1"] = x - self.roi_delta
            self.roi["x2"] = x + self.roi_delta
            self.roi["y1"] = y - self.roi_delta
            self.roi["y2"] = y + self.roi_delta
            self.new_roi = True
            print("Target center selected: {} {}".format(x, y))
    
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
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()
        return ret, frame, frame_gray, vis

    def run(self):
        while True:
            try:
                _ret, frame, frame_gray, self.vis = self.read_and_resize()
            except Exception:
                print("The End")
                cv2.waitKey(0)
                self.close()
                break

            # Calculate flow of existing tracks
            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                p_vec = p1-p0;
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                #import pdb; pdb.set_trace()                
                for tr, (x_new, y_new), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x_new, y_new))
                    if len(tr) > self.track_len: # cut the tail of the track
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(self.vis, (x_new, y_new), 2, (0, 255, 0), -1)
                #import pdb; pdb.set_trace()  
                
                (x_mean, y_mean) = sum(p1.reshape(-1, 2))/len(self.tracks)
                # ROI movement vector
                if self.new_roi is True:
                    self.new_roi = False
                else:
                    self.roi = dict (x1 = int(x_mean) - self.roi_delta,
                                     x2 = int(x_mean) + self.roi_delta,
                                     y1 = int(y_mean) - self.roi_delta,
                                     y2 = int(y_mean) + self.roi_delta)
                print(self.roi)
                #self.roi = { k: self.roi.get(k, 0) + delta_roi.get(k, 0) for k in set(self.roi) | set(delta_roi) }
                
                self.tracks = new_tracks
                cv2.polylines(self.vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0)) # plot track
                self.print_text_on_frame(self.vis, 'track count: {}'.format(len(self.tracks)))

            # Search for new features
            if self.frame_idx % self.detect_interval == 0:
                # Create mask for tracks
                mask = np.zeros_like(frame_gray)
                mask[self.roi["y1"]:self.roi["y2"], self.roi["x1"]:self.roi["x2"]] = 255
                #import pdb; pdb.set_trace()
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
            # Show the mask region
            cv2.rectangle(self.vis, (self.roi["x1"], self.roi["y1"]), (self.roi["x2"], self.roi["y2"]), (255,0,0), 2)

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('video', self.vis)

            ch = cv2.waitKey(400) & 0xFF
            if ch == 27:
                self.close()
                break
    
    def close(self):
        self.cam.release()
        cv2.destroyAllWindows()

def main():
    #video_src = "DJI_0004.MP4"
    video_src = "vtest.avi"

    print(__doc__)
    App(video_src).run()

if __name__ == '__main__':
    main()
