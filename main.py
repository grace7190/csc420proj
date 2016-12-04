# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 14:31:49 2016

@author: zhuyiyan, bilodea4
"""
import numpy as np
import cv2
from skimage.measure import block_reduce
from fractions import gcd
import random
from sklearn import svm
from sklearn.externals import joblib
import glob

SCALE_FACTOR = 4 # downsample by a factor of SCALE_FACTOR
SHOT_TRANSITION = 5 # number of estimated frames for shot transitions
FACE_KEEP = 10
hog = cv2.HOGDescriptor()

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
    if obj_pca == None or scene_pca == None:
        print("no features")
        return []
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
                                         scaleFactor=1.16,
                                         minNeighbors=3,
                                         minSize=(MIN,MIN),
                                         maxSize=(MAX,MAX),
                                         flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    faceCascade2 = cv2.CascadeClassifier('haarcascade_profileface.xml')
    faces2 = faceCascade2.detectMultiScale(frame,
                                         scaleFactor=1.06,
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


def train_HOG_SVM():
    clf = svm.SVC(gamma=0.001, C=100.)
    features = []
    labels = []

    for filename in glob.glob("AdienceDB/female_faces/*.jpg"):
        face = cv2.imread(filename)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (156, 156))
        features.append(hog.compute(face).flatten())
        labels.append("female")
    print "done females"
    for filename in glob.glob("AdienceDB/male_faces/*.jpg"):
        face = cv2.imread(filename)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (156, 156))
        features.append(hog.compute(face).flatten())
        labels.append("male")
    print "done males"

    clf.fit(features, labels)
    print "done training"

    joblib.dump(clf, "clf.pkl")


def face_track2(frame_faces_list):
    # list of lists of faces, grouped by similarity
    similar_faces = []

    frame_faces_list = [[z.tolist() for z in x] for x in frame_faces_list]
    # initialise list with first frame faces
    if frame_faces_list != []:
        for face in frame_faces_list[0]:
            similar_faces.append([face])

    # compare the faces from subsequent frames
    for i in range(len(frame_faces_list)-1):
        frame_faces = frame_faces_list[i+1]
        for face_list in similar_faces:
            found_similar = False
            # get the latest non-False face
            
            latest_face = np.trim_zeros(face_list)[-1]
            # compare latest face to frame faces
            for frame_face in frame_faces[::-1]:
                if is_similar(latest_face, frame_face):
                    # found similar face, add to group
                    face_list.append(frame_face)
                    frame_faces.remove(frame_face)
                    found_similar = True
                    break
            if not found_similar:
                # didn't find similar face in frame
                face_list.append(False)
        # matched faces have been removed from frame_faces
        # if a new face appeared in frame, create new list in similar_faces
        for leftover in frame_faces:
            temp = []
            for j in range(i+1):
                # face didn't appear in previous i frames...
                temp.append(False)
            temp.append(leftover)
            similar_faces.append(temp)

    # by now, similar_faces has all faces that have appeared in set
    # of frames, in lists by similarity, indexed by frame number

    for face_list in similar_faces[::-1]:
        non_null_faces = filter(lambda x: x, face_list)
        # remove "faces" that don't appear often enough
        if len(non_null_faces) < 3:
            similar_faces.remove(face_list)

        # assign face colour
        face_list.append((random.randint(1,256),
                          random.randint(1,256),
                          random.randint(1,256)))

    return similar_faces


def decide_gender(face_list, frames):
    '''Decide face gender based on random sampling'''
    NUM_SAMPLES = 20
    range = min(len(face_list)/2, NUM_SAMPLES)
    male = 0
    female = 0
    i = 0
    while i <= range:
        idx = random.randint(0, len(face_list)-1)
        face = face_list[idx]
        if not face:
            # try another index
            continue
        x, y, w, h = face
        x = int(x * SCALE_FACTOR)
        y = int(y * SCALE_FACTOR)
        w = int(w * SCALE_FACTOR)
        h = int(h * SCALE_FACTOR)
        gray_face = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
        face_image = gray_face[y:y + h, x:x + w]
        face_image = cv2.resize(face_image, (156, 156))

        # classify using pre-trained svm classifier
        if clf.predict([hog.compute(face_image).flatten()]) == "male":
            male += 1
        else:
            female += 1
        i += 1
    return "male" if male > female else "female"



if __name__ == '__main__':

    # for i in range(9):
    #     frame = cv2.imread("faces_{0}.png".format(i+1))
    #     out_frame = frame[:,:,:]
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     faces = face_detect(gray)
    #
    #     for (x, y, w, h) in faces:
    #         x = int(x)*SCALE_FACTOR
    #         y = int(y)*SCALE_FACTOR
    #         w = int(w)*SCALE_FACTOR
    #         h = int(h)*SCALE_FACTOR
    #         print (h,w)
    #         face = out_frame[y:y+h, x:x+w, :]
    #         face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    #         face = cv2.resize(face, (156,156))
    #         print face.shape
    #         cv2.imshow("face", face)
    #         cv2.waitKey(0)
    #         hog_face = hog.compute(face)
    #         print(hog_face)
    #         print(hog_face.shape)
    #         cv2.rectangle(out_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    #     cv2.imshow("we", out_frame)
    #     cv2.waitKey(0)


    # Load in pretrained SVM
    clf = joblib.load("clf.pkl")

    # make numbers human-readable
    np.set_printoptions(suppress=True)

    # set up SIFT parameters
    logo_im = cv2.imread('cbc_logo.png')
    logo_gray = cv2.cvtColor(logo_im, cv2.COLOR_BGR2GRAY)
    logo_small = cv2.resize(logo_gray, None, fx=1.0/SCALE_FACTOR, fy=1.0/SCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    sift = cv2.SIFT(contrastThreshold=0.1, edgeThreshold=5)
    kp_logo, des_logo = sift.detectAndCompute(logo_small, None)

    # Define VideoCapture object and parameters
    cap_in = cv2.VideoCapture('Alec.avi')
    fourcc = cv2.cv.CV_FOURCC('F', 'M', 'P', '4')
    out = cv2.VideoWriter('output_alec_shotexample.avi', fourcc, 30.0, (1280,720))

    # set up loop
    ret, frame = cap_in.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_frames = [gray] # keep a stack of previous frames, like maybe 5
    frame_counter=0 # number of frames processed
    out_frames = []
    shot_faces = []

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
                # cv2.rectangle(out_frame, (x1, y1), (x2, y2), (255,255,255), 4)

            faces_detected = face_detect(gray)
            # save up shot's-worth of out_frames and detected faces
            out_frames.append(out_frame)
            shot_faces.append(faces_detected)

            if detectShot(gray, prev_frames[0], 'norm'):
                # detected shot! process saved up frames/faces
                zeros = np.zeros((frame.shape[0],frame.shape[1]), dtype="uint8")
                out_frame = cv2.merge([zeros, zeros, out_frame[:,:,0]])
                # replace last one with new red frame..
                out_frames[-1] = out_frame

                # get lists of real faces, colour, and gender
                faces = face_track2(shot_faces)
                for face_list in faces:
                    # face_list = [(face), (face), ..., (face), colour]
                    colour = face_list[-2]
                    gender = decide_gender(face_list[:-1], out_frames)
                    for i in range(len(out_frames)):
                        # if face was present in frame, draw it into output
                        if face_list[i]:
                            x, y, w, h = face_list[i]
                            x = int(x * SCALE_FACTOR)
                            y = int(y * SCALE_FACTOR)
                            w = int(w * SCALE_FACTOR)
                            h = int(h * SCALE_FACTOR)
                            cv2.rectangle(out_frames[i], (x, y), (x + w, y + h), colour, 2)
                            gender_label_colour = (255, 128, 128) if gender == 'male' else (128, 128, 255)
                            cv2.putText(out_frames[i], gender, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, gender_label_colour, 2)

                # draw out shot frames
                for temp in out_frames:
                    out.write(temp)
                out_frames = []
                shot_faces = []

            prev_frames.append(gray)
            if len(prev_frames) > SHOT_TRANSITION:  # hopefully detect slow transitions
                prev_frames = prev_frames[1:]
            frame_counter+=1
            print(frame_counter),
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # get lists of real faces, colour, and gender
            faces = face_track2(shot_faces)
            for face_list in faces:
                # face_list = [(face), (face), ..., (face), colour]
                colour = face_list[-2]
                gender = decide_gender(face_list[:-1], out_frames)
                for i in range(len(out_frames)):
                    # if face was present in frame, draw it into output
                    if face_list[i]:
                        x, y, w, h = face_list[i]
                        x = int(x * SCALE_FACTOR)
                        y = int(y * SCALE_FACTOR)
                        w = int(w * SCALE_FACTOR)
                        h = int(h * SCALE_FACTOR)
                        cv2.rectangle(out_frames[i], (x, y), (x + w, y + h), colour, 2)
                        gender_label_colour = (255, 128, 128) if gender == 'male' else (128, 128, 255)
                        cv2.putText(out_frames[i], gender, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, gender_label_colour)

            # draw out shot frames
            for temp in out_frames:
                out.write(temp)
            out_frames = []
            shot_faces = []
            break

        # read next frame
        ret, frame = cap_in.read()
    
        
    
    cap_in.release()
    out.release()
    cv2.destroyAllWindows()
