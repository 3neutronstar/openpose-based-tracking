import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import matplotlib
import os


from random import randint
from multi_pose_estimation import getKeypoints, getPersonwiseKeypoints, getValidPairs
from utils import get_accuracy


if __name__=='__main__':
    
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    device = '0'

    nPoints = 18
    # COCO Output Format
    keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                        'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 
                        'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

    pose_pairs = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
                [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
                [1,0], [0,14], [14,16], [0,15], [15,17],
                [2,17], [5,16] ]

    # index of pafs correspoding to the pose_pairs
    mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], 
            [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], 
            [47,48], [49,50], [53,54], [51,52], [55,56], 
            [37,38], [45,46]]

    colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
            [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
            [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]

    # Read the network into memory
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    cap=cv2.VideoCapture(os.path.join('video','TmaxDataset2.mp4'))
    idx_t=0
    while(cap.isOpened()):
        t=time.time()
        ret, frame = cap.read()
                
        # Fix the input Height and get the width according to the Aspect Ratio
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        inHeight = 368
        inWidth = int((inHeight/frameHeight)*frameWidth)

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)

        output = net.forward()
        H = output.shape[1]
        W = output.shape[2]
        #print(output.shape)
        inft=time.time()-t


        # if 1:
        #     i = 0
        #     probMap = output[0, i, :, :]
        #     probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        #     plt.figure(figsize=[14,10])
        #     plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #     plt.imshow(probMap, alpha=0.6)
        #     plt.colorbar()
        #     plt.axis("off")
        #     plt.savefig('./images/{}_test.jpg'.format(idx_t))
        #     plt.clf()
        #     plt.close()
        #     #exit()
        # cv2.resize()
        detected_keypoints = []
        keypoints_list = np.zeros((0,3))
        keypoint_id = 0
        threshold = 0.1

        for part in range(nPoints):
            probMap = output[0,part,:,:]
            probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
        #     plt.figure()
        #     plt.imshow(255*np.uint8(probMap>threshold))
            keypoints = getKeypoints(probMap, threshold)
            #print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
            keypoints_with_id = []
            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                keypoint_id += 1

            detected_keypoints.append(keypoints_with_id)



        valid_pairs, invalid_pairs = getValidPairs(output,mapIdx,detected_keypoints,frameWidth, frameHeight,pose_pairs)
        #print(valid_pairs)
        #print('invalid!!!!!!:' , invalid_pairs)

        personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs,mapIdx,pose_pairs,keypoints_list)

        for n in range(len(personwiseKeypoints)):
            for i in range(17):
            #for n in range(1):
                index = personwiseKeypoints[n][np.array(pose_pairs[i])]
                if -1 in index:
                    continue
                B = np.int32(keypoints_list[index.astype(int), 0])
                #print(B)
                A = np.int32(keypoints_list[index.astype(int), 1])
                cv2.line(frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
        #     plt.figure(figsize=[15,15])
        #     plt.imshow(frame)
        #     plt.savefig('./images/{}_test2.jpg'.format(idx_t))
        #     plt.clf()
        #     plt.close()
    
        dicts = {}
        for idx in keypointsMapping:
            dicts[idx] = 0

        for n in range(len(personwiseKeypoints)):
            boole = personwiseKeypoints[n][:18] >= 0
            #print(boole)
            keylist = [i for i in range(18)]
            watch = np.array(keypointsMapping)[boole]

            for idx in watch:
                dicts[idx] +=1

        endt=time.time() - t

        cv2.putText(frame, 'Number of People:{}'.format(max(dicts.values())),(int(W/2),int(H*3/4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, 'FPS: {:.3f}/ Proc Time: {:.3f}'.format(1./endt,endt),(int(W/2),int(H*3/4+40)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, 'INF Time: {:.3f}'.format(inft),(int(W/2),int(H*3/4+80)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.imshow('frame',frame)
        idx_t+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

