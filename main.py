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
import random


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
    frame = cv2.resize(frame, None, fx=1.0/SCALE_FACTOR, fy=1.0/SCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    MIN = int(0.05 * (frame.shape[0] + frame.shape[1])/2) # min = around 5% of video size
    MAX = int(0.3 * (frame.shape[0] + frame.shape[1])/2) # max = around 30% of video size
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = faceCascade.detectMultiScale(frame, 
                                         scaleFactor=1.2,
                                         minNeighbors=3,
                                         minSize=(MIN,MIN),
                                         maxSize=(MAX,MAX),
                                         flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    faceCascade2 = cv2.CascadeClassifier('haarcascade_profileface.xml')
    faces2 = faceCascade2.detectMultiScale(frame, 
                                         scaleFactor=1.1,
                                         minNeighbors=3,
                                         minSize=(MIN,MIN),
                                         maxSize=(MAX,MAX),
                                         flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    flipped = cv2.flip(frame,1)
    faceCascade3 = cv2.CascadeClassifier('haarcascade_profileface.xml')
    faces3 = faceCascade3.detectMultiScale(flipped, 
                                         scaleFactor=1.06,
                                         minNeighbors=3,
                                         minSize=(MIN,MIN),
                                         maxSize=(MAX,MAX),
                                         flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    faces4 = []
    for f in faces3:
        faces4.append([frame.shape[1]-f[0]-f[2],f[1],f[2],f[3]])
    faces4 = np.array(faces4)
    
    print "Found {0}+{1}+{2} faces!".format(len(faces),len(faces2), len(faces4))
    
    output = np.zeros((0,4))
    
    if len(faces) > 0: # the 'numpy arrays are dumb' check
        output = np.concatenate((output, faces), axis=0)
    if len(faces2) > 0:
        output = np.concatenate((output, faces2), axis=0) 
    if len(faces4) > 0:
        output = np.concatenate((output, faces4), axis=0)
    
    filtered_output = []
    for face in output:
        if len(filtered_output) < 1:
            filtered_output.append(face)
        else:
            similar_faces = list(filter(lambda x: is_similar(x, face), filtered_output))
            if len(similar_faces) < 1:
                filtered_output.append(face)
        
    return filtered_output

def face_track(prev_faces_list, curr_faces):
    '''prev_faces_list is a list of lists of faces in (x,y,w,h) format, from previous frames
    curr_faces is output of face detection on current frame'''
    
    if len(prev_faces_list) < 1:
        curr_faces_colored = []
        for face in curr_faces:
            # add colour label to face
            colour = (random.randint(1,256), random.randint(1,256), random.randint(1,256))
            col_face = face.tolist()
            col_face.append(colour) 
            curr_faces_colored.append(col_face)
        return curr_faces_colored
        
    output_faces = []
        
    real_faces = []
    for face in curr_faces:
        for pface in prev_faces_list[-1]:
            if is_similar(face, pface):
                real_faces.append(face)
                
    face_counts = {}
                         
    for prev_faces in prev_faces_list[::-1]:
        for face in prev_faces:
            # find similar faces already in face_counts, if any
            similar_faces = list(filter(lambda x: is_similar(x[:4], face), face_counts.keys()))
            if len(similar_faces) > 0:
                # add a count to the face if found
                face_counts[tuple(similar_faces[0])] += 1 
            else:
                face_counts[tuple(face)] = 1 # add face if not found
                             
    for face in curr_faces:
        similar_faces = list(filter(lambda x: is_similar(x[:4], face), face_counts.keys()))
        if len(similar_faces) > 0:
            # if there's already a similar face in prev_faces, update coordinate in face_counts
            num_faces = face_counts.pop(similar_faces[0])
            colour = similar_faces[0][4]
            col_face = face.tolist()
            col_face.append(colour)
            face_counts[tuple(col_face)] = num_faces + 1
        else:
            # otherwise, add face to face_counts. probably useless code for now
            colour = (random.randint(1,256), random.randint(1,256), random.randint(1,256))
            col_face = face.tolist()
            col_face.append(colour) # add colour label to face
            face_counts[tuple(face)] = 1
    
    for bface, count in face_counts.items():
        # only significant if face appears more than half the times in previous x frames
        if count > len(prev_faces_list)/2:
            output_faces.append(bface)

    return output_faces
    
def is_similar(face1, face2):
    '''face: (x,y,w,h)'''
    threshold = 30
    for i in range(len(face1)):
        if abs(face1[i] - face2[i]) > threshold:
            return False
    return True
    
def average_face(face1, face2):
    '''face: (x,y,w,h)'''
    x = (face1[0]+face2[0])/2
    y = (face1[1]+face2[1])/2
    w = (face1[2]+face2[2])/2
    h = (face1[3]+face2[3])/2
    return (x,y,w,h)
    
if __name__ == '__main__':
    # make numbers human-readable
    np.set_printoptions(suppress=True)
    
    # set up SIFT parameters
    logo_im = cv2.imread('cbc_logo.png')
    logo_gray = cv2.cvtColor(logo_im, cv2.COLOR_BGR2GRAY)
    SCALE_FACTOR = 4 # downsample by a factor of SCALE_FACTOR
    SHOT_TRANSITION = 8 # number of estimated frames for shot transitions
    FACE_KEEP = 10
    logo_small = cv2.resize(logo_gray, None, fx=1.0/SCALE_FACTOR, fy=1.0/SCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    sift = cv2.SIFT(contrastThreshold=0.1, edgeThreshold=5)
    kp_logo, des_logo = sift.detectAndCompute(logo_small, None)
    
    # Define VideoCapture object and parameters
    cap_in = cv2.VideoCapture('Alec.avi')
    fourcc = cv2.cv.CV_FOURCC('F', 'M', 'P', '4')
    out = cv2.VideoWriter('output_alec_2.avi', fourcc, 30.0, (1280,720))

    # set up loop
    ret, frame = cap_in.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_frame = frame
    prev_frames = [gray] # keep a stack of previous frames, like maybe 5
    prev_gray = gray
    frame_counter=0 # number of frames processed
    prev_faces = []
    prev_face = ()
    
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
                prev_faces = []
            
            faces_detected = face_detect(gray)
            if len(faces_detected) > 0:
                faces = face_track(prev_faces, faces_detected)
            else:
                faces = faces_detected
            for face in faces:
                print(face)
                x, y, w, h, colour = face
                cv2.rectangle(out_frame, (int(x*SCALE_FACTOR), int(y*SCALE_FACTOR)), (int(x+w)*SCALE_FACTOR, int(y+h)*SCALE_FACTOR), colour, 2)
            prev_face = faces
            prev_faces.append(faces)
            
            # write frame
            out.write(out_frame)
            prev_frame = frame
            prev_frames.append(gray)
            if len(prev_frames) > SHOT_TRANSITION: # hopefully detect slow transitions
                prev_frames = prev_frames[1:]
            if len(prev_faces) > FACE_KEEP:
                prev_faces = prev_faces[1:]
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
    