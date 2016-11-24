# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:31:49 2016

@author: zhuyiyan, bilodea4
"""
import numpy as np
import cv2

#Detect shots in the videos. A shot is a set of consecutive frames with a smooth camera
#motion.
#4. (Manually) Annotate shot boundaries in the video. How would you evaluate how well
#you are detecting the shots? Compute your performance



if __name__ == '__main__':

    # Define VideoCapture objects
    cap_in = cv2.VideoCapture('test.avi')
    fourcc = cv2.cv.CV_FOURCC('F', 'M', 'P', '4')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280,720))
    
    while(cap_in.isOpened()):
        # read frame
        ret, frame = cap_in.read()
        if ret == True:
            # process frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # write frame
            out.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        
    cap_in.release()
    out.release()
    cv2.destroyAllWindows()
    