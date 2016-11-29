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
    '''frame 1, frame 2 are frames to compare to see if shot transitions
    unfortunately ncc requires different thresholds and heavier size reduction
    to run properly so... this function is kind of... mashed together'''

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
        
    elif option == 'norm':
        threshold = 400
        bsize = gcd(frame1.shape[0], frame1.shape[1])
        small1 = block_reduce(frame1, block_size=(bsize, bsize), func=np.mean)
        small2 = block_reduce(frame2, block_size=(bsize, bsize), func=np.mean)
        norm = np.linalg.norm(small1 - small2)
        print norm
        return norm > threshold
        
    else:
        raise Exception("detectShot takes 'ssd' or 'norm' as 3rd input")
        
        
        

def affine_solver(kp1, kp2, matches, k):
    '''given two sets of keypoints, sorted matches, and k, calculate affine 
    transformation for top k matches'''
    top_k = matches[0:k]
    match1 = []
    match2 = []
    for feat in top_k:
        match1.append(kp1[int(feat[0])].pt)
        match2.append(kp2[int(feat[1])].pt)
        
    #construct matrices for computing affine transformation
    p1 = []
    p2 = []
    for pt1 in match1:
        p1.append([pt1[0],pt1[1],0,0,1,0])
        p1.append([0,0,pt1[0],pt1[1],0,1])
    for pt2 in match2:
        p2.append([pt2[0]])
        p2.append([pt2[1]])
    
    p1 = np.array(p1)
    p2 = np.array(p2)
    a = np.dot(np.dot(scipy.linalg.pinv(np.dot(np.transpose(p1),p1)),np.transpose(p1)),p2)
    return a
    

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
        
    
def findLogo(frame, kp1, des1):
    '''use SIFT to find logo in given frame'''
    frame_small = cv2.resize(frame, None, fx=1.0/SCALE_FACTOR, fy=1.0/SCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    kp2, des2 = sift.detectAndCompute(frame_small, None)
    matches = featurecorr(des1, des2)
    sorted_matches = sorted(matches, key=lambda tup: tup[2])
    A = np.array([])
    if len(matches) > 3:
        A = affine_solver(kp1, kp2, sorted_matches, 4)
    return A
    
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
    cap_in = cv2.VideoCapture('Alec.avi')
    fourcc = cv2.cv.CV_FOURCC('F', 'M', 'P', '4')
    out = cv2.VideoWriter('output_alec.avi', fourcc, 30.0, (1280,720))

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
            
            affine = findLogo(gray, kp_logo, des_logo)
            
            if len(affine) > 0:
                M = np.float32([[affine[0][0],affine[1][0],affine[4][0]],[affine[2][0],affine[3][0],affine[5][0]]])
                logo_tl = np.dot(M, np.transpose(np.array([[0,0,1]])))
                logo_br = np.dot(M, np.transpose(np.array([[logo_small.shape[1],logo_small.shape[0],1]])))
                x1 = int(logo_tl[0][0])*SCALE_FACTOR
                y1 = int(logo_tl[1][0])*SCALE_FACTOR
                x2 = int(logo_br[0][0])*SCALE_FACTOR
                y2 = int(logo_br[1][0])*SCALE_FACTOR
                # print((x1, y1, x2, y2))
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), (255,255,255), 4)
            
            if detectShot(gray, prev_frames[0], 'norm'):
                zeros = np.zeros((frame.shape[0],frame.shape[1]), dtype="uint8")
                out_frame = cv2.merge([zeros, zeros, out_frame[:,:,0]])
                
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
    