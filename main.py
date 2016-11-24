# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:31:49 2016

@author: zhuyiyan, bilodea4
"""
import numpy as np
import cv2
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

#Detect shots in the videos. A shot is a set of consecutive frames with a smooth camera
#motion.
#4. (Manually) Annotate shot boundaries in the video. How would you evaluate how well
#you are detecting the shots? Compute your performance

def detectShot(frame1, frame2):
    '''frame 1, frame 2 are frames to compare to see if shot transitions'''
    threshold = 5000000
    small1 = block_reduce(frame1, block_size=(16, 16), func=np.mean)
    small2 = block_reduce(frame2, block_size=(16, 16), func=np.mean)
    ssd = 0
    for i in range(small1.shape[0]):
        for j in range(small1.shape[1]):
            diff = (small1[i,j] - small2[i,j]) ** 2
            ssd += diff
    return ssd > threshold
    
if __name__ == '__main__':

    # Define VideoCapture object and parameters
    cap_in = cv2.VideoCapture('test.avi')
    fourcc = cv2.cv.CV_FOURCC('F', 'M', 'P', '4')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280,720))
    
    ret, frame = cap_in.read()
    prev_frame = frame
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while(cap_in.isOpened()):
        if ret == True:
            # process frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame == None:
                continue
            
            if detectShot(gray, prev_gray):
                zeros = np.zeros((frame.shape[0],frame.shape[1]), dtype="uint8")
                out_frame = cv2.merge([zeros, zeros, frame[:,:,0]])
            else:
                out_frame = frame
                
            print(detectShot(gray, prev_gray))
                
            # write frame
            out.write(out_frame)
            prev_frame = frame
            prev_gray = gray
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        
        # read next frame
        ret, frame = cap_in.read()
        
    cap_in.release()
    out.release()
    cv2.destroyAllWindows()
    