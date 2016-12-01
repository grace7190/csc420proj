# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:31:49 2016

@author: zhuyiyan, bilodea4
"""
import numpy as np
import cv2
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import scipy
from fractions import gcd


#Detect shots in the videos. A shot is a set of consecutive frames with a smooth camera
#motion.
#4. (Manually) Annotate shot boundaries in the video. How would you evaluate how well
#you are detecting the shots? Compute your performance

def detectShot(frame1, frame2, option):
    '''frame 1, frame 2 are frames to compare to see if shot transitions'''
    # ssd doesn't seem to be very effective
    if option == 'ssd':
        threshold = 12000000
        small1 = block_reduce(frame1, block_size=(16, 16), func=np.mean)
        small2 = block_reduce(frame2, block_size=(16, 16), func=np.mean)
        ssd = 0
        for i in range(small1.shape[0]):
            for j in range(small1.shape[1]):
                diff = (small1[i,j] - small2[i,j]) ** 2
                ssd += diff
        return ssd > threshold
    # thought this would be faster than ssd, seems to have similar results
    elif option == 'norm':
        threshold = 600
        bsize = gcd(frame1.shape[0], frame1.shape[1])
        # reduce size of image
        small1 = block_reduce(frame1, block_size=(bsize, bsize), func=np.mean)
        small2 = block_reduce(frame2, block_size=(bsize, bsize), func=np.mean)
        #take the norm
        norm = np.linalg.norm(small1 - small2)
        print(norm)
        return norm > threshold
        
    else:
        raise Exception("detectShot takes 'ssd' or 'norm' as 3rd input")
    
def predict_location(frame, kp1, des1):
    '''use SIFT to find logo in given frame and predict location of logo'''
    frame_small = cv2.resize(frame, None, fx=1.0/SCALE_FACTOR, fy=1.0/SCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    kp2, des2 = sift.detectAndCompute(frame_small, None)
    matches = featurecorr(des1, des2)
    if matches == []:
        return ()
    # average the points where matches were found to get approximate coordinates
    xs = []
    ys = []
    for m in matches:
        xs.append(kp2[int(m[1])].pt[0])
        ys.append(kp2[int(m[1])].pt[1])
    size = min(max((max(xs)-min(xs) + max(ys)-min(ys))/2, 25), 600) # set min/max size
    return (sum(xs) / float(len(xs)), sum(ys) / float(len(ys)), size)
    

def featurecorr(obj_pca, scene_pca):
    '''given features of object and features of scene, determine matches)'''
    match = []
    threshold = 0.6
    for i in range(0,len(obj_pca)):
        f = obj_pca[i]
        distances = []
        for j in range(0,len(scene_pca)):
            g = scene_pca[j]
            distances.append(np.linalg.norm(f-g))
        if len(distances) < 3:
            continue
        min1 = min(distances) # closest
        min2 = min(n for n in distances if n!=min1)
        ratio = min1/min2
        if ratio < threshold:
            match.append((i, distances.index(min1), ratio))
    return match
    
    
def face_detect(frame):
    # followed tutorial here: https://realpython.com/blog/python/face-recognition-with-python/
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(gray, 
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(80,80),
                                         maxSize=(300,300),
                                         flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    print "Found {0} faces!".format(len(faces))
    return faces

    
if __name__ == '__main__':
    # make numbers human-readable
    np.set_printoptions(suppress=True)
    
    # set up SIFT parameters
    logo_im = cv2.imread('cbc_logo.png')
    logo_gray = cv2.cvtColor(logo_im, cv2.COLOR_BGR2GRAY)
    SCALE_FACTOR = 4 # downsample by a factor of SCALE_FACTOR
    SHOT_TRANSITION = 8 # number of estimated frames for shot transitions
    logo_small = cv2.resize(logo_gray, None, fx=1.0/SCALE_FACTOR, fy=1.0/SCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    sift = cv2.SIFT(contrastThreshold=0.1, edgeThreshold=5)
    kp_logo, des_logo = sift.detectAndCompute(logo_small, None)
    
    # Define VideoCapture object and parameters
    cap_in = cv2.VideoCapture('test.avi')
    fourcc = cv2.cv.CV_FOURCC('F', 'M', 'P', '4')
    out = cv2.VideoWriter('output_test.avi', fourcc, 30.0, (1280,720))

    # set up loop
    ret, frame = cap_in.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_frame = frame
    prev_frames = [gray] # keep a stack of previous frames, like maybe 5
    prev_gray = gray
    frame_counter=0 # number of frames processed
    
    while(cap_in.isOpened()):
        if ret == True:
            
            # process frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            out_frame = frame

            loc = predict_location(gray, kp_logo, des_logo)
            if len(loc) > 0: 
                x1 = int((loc[0]-loc[2]/2)*SCALE_FACTOR)
                x2 = int((loc[0]+loc[2]/2)*SCALE_FACTOR)
                y1 = int((loc[1]-loc[2]/2)*SCALE_FACTOR)
                y2 = int((loc[1]+loc[2]/2)*SCALE_FACTOR)
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), (255,255,255), 4)
            
            if detectShot(gray, prev_frames[0], 'norm'):
                zeros = np.zeros((frame.shape[0],frame.shape[1]), dtype="uint8")
                out_frame = cv2.merge([zeros, zeros, out_frame[:,:,0]])
            
            faces = face_detect(out_frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(out_frame, (x, y), (x+w, y+h), (0,0,255), 2)
                
            # write frame
            out.write(out_frame)
            prev_frame = frame
            prev_frames.append(gray)
            if len(prev_frames) > SHOT_TRANSITION: # hopefully detect slow transitions
                prev_frames = prev_frames[1:]
            prev_gray = gray
            frame_counter+=1
            print(frame_counter),
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        
        # read next frame
        ret, frame = cap_in.read()
        
    cap_in.release()
    out.release()
    cv2.destroyAllWindows()
    