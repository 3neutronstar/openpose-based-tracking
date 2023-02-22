import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import matplotlib

from random import randint
from utils import get_accuracy



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# Find the Keypoints using Non Maximum Suppression on the Confidence Map
def getKeypoints(probMap, threshold=0.1):
    
    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)
    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []
    
    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints

def getValidPairs(output,mapIdx,detected_keypoints,frameWidth, frameHeight,pose_pairs):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.1
    conf_th = 0.7
    # 모든 POSE_PAIR에 대해
    for k in range(len(mapIdx)):
        # limb의 한쪽 조인트 A, 다른쪽 조인트B 각각에 대한 paf
        pafA = output[0, mapIdx[k][0], :, :]
        pafB = output[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))

        # limb의 양쪽 keypoint 후보들
        candA = detected_keypoints[pose_pairs[k][0]]
        candB = detected_keypoints[pose_pairs[k][1]]
        nA = len(candA)
        nB = len(candB)

        # joint keypoint가 detect된 경우에 한해
        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            # 조인트들 사이의 모든 interpolated points 집합에 대해 PAF를 계산한다
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):         
                    # 두 joint 사이의 distance vector(d_ij)계산
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    # distance vector(d_ij)를 normalize
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # p(u)
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
                    # L(p(u))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ]) 
                    # E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)

                    # Connection이 valid한지 체크하기 위해 threshold로 걸러낸다  
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Valid한 connection들 저장
                if found:            
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

            # valid_pair과 valid_pairs 다름 주의 - valid_pairs는 global variable
            valid_pairs.append(valid_pair)
            
        # Keypoint가 detect되지 않았을 경우
        else: 
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    #print(valid_pairs)
    return valid_pairs, invalid_pairs

# for each detected valid pair, it assigns the joint(s) to a person
# It finds the person and index at which the joint should be added. This can be done since we have an id for each joint
def getPersonwiseKeypoints(valid_pairs, invalid_pairs,mapIdx,pose_pairs,keypoints_list): 
    #각 사람의 keypoints를 저장하기 위한 empty list 생성
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(pose_pairs[k])
            #valid_pair 각각에 대해, partA가 personwiseKeypoints에 있는지 확인
            for i in range(len(valid_pairs[k])): 
                found = 0
                person_idx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break
                # 만약 partA가 있다면, partB도 personwiseKeypoints에 추가
                if found:
                    personwiseKeypoints[person_idx][indexB] = partBs[i]
                    personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

                # 만댝 partA가 모든 list에 없다면, list에 없는 새로운 사람의 part라는 뜻
                elif not found and k < 17:
                    # partA-partB 전체를 한 row로 생성해서 추가
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    # 각 keypoint의 keypoint_scores와 paf_score도 추가
                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints

if __name__ == '__main__':
    protoFile = "./coco/pose_deploy_linevec.prototxt"
    weightsFile = "./coco/pose_iter_440000.caffemodel"
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


    accs=AverageMeter('Acc', ':6.2f')
    image_accs=AverageMeter('ImageAcc', ':6.2f')
    # Get annotation
    numofperson, keys = get_accuracy()
    boole = np.array(numofperson) > 0
    numofperson_sel = np.array(numofperson)[boole]
    #print(sum(numofperson_sel))
    #exit()
    keys_sel = np.array(keys)[boole]

    # Load val dataset
    filelist = os.listdir('..\..\../data/dataset/coco2017/val2017/')
    filelist_sel = np.array(filelist)[boole]
    print('Total dataset :', len(filelist_sel))

    total_accuracy = 0.0
    total_person = 0.0
    for idx_t, key in enumerate(keys_sel):
        print('idx :', idx_t)
        num_zero = 12 - len(str(key))
        pt = '0' * num_zero + str(key) + '.jpg'
    #for idx_t, pt in enumerate(filelist_sel):
        image1 = cv2.imread('..\..\../data/dataset/coco2017/val2017/' + str(pt))
        #print(image1)
        #print(image1.shape)
        #exit()
        frameWidth = image1.shape[1]
        frameHeight = image1.shape[0]

        t = time.time()

        # Read the network into memory
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

        # Fix the input Height and get the width according to the Aspect Ratio
        inHeight = 368
        inWidth = int((inHeight/frameHeight)*frameWidth)

        inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)

        net.setInput(inpBlob)

        output = net.forward()
        H = output.shape[2]
        W = output.shape[3]
        #print(output.shape)

        print("Time Taken = {}".format(time.time() - t))

        if 1:
            i = 0
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (frameWidth, frameHeight))

            # plt.figure(figsize=[14,10])
            # plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            # plt.imshow(probMap, alpha=0.6)
            # plt.colorbar()
            # plt.axis("off")
            # # make images dir
            # if not os.path.isdir('images'):
            #     os.mkdir('images')
            # plt.savefig('./images/{}_test.jpg'.format(idx_t))
            # plt.clf()
            #exit()
    
        detected_keypoints = []
        keypoints_list = np.zeros((0,3))
        keypoint_id = 0
        threshold = 0.1
    
        for part in range(nPoints):
            probMap = output[0,part,:,:]
            probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
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

        frameClone = image1.copy()
        # if 1:
        #     for i in range(nPoints):
        #         for j in range(len(detected_keypoints[i])):
        #             cv2.circle(frameClone, detected_keypoints[i][j][0:2], 3, [0,0,255], -1, cv2.LINE_AA)
        #     plt.figure(figsize=[15,15])
        #     plt.imshow(frameClone[:,:,[2,1,0]])
        #     plt.savefig('./images/{}_test2.jpg'.format(idx_t))
        #     plt.clf()
    

        valid_pairs, invalid_pairs = getValidPairs(output,mapIdx,detected_keypoints,frameWidth,frameHeight,pose_pairs)
        #print(valid_pairs)
        #print('invalid!!!!!!:' , invalid_pairs)

        personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs,mapIdx,pose_pairs,keypoints_list)
        #print(personwiseKeypoints)
        #print(len(personwiseKeypoints[0]))
        #print(len(personwiseKeypoints))
        #print('Pose Pair Len :', len(pose_pairs))
        #print(np.array(pose_pairs[0]))
        #print(personwiseKeypoints[0][np.array(pose_pairs[0])])

        num = []
        if 1:
            for n in range(len(personwiseKeypoints)):
                for i in range(17):
                #for n in range(1):
                    index = personwiseKeypoints[n][np.array(pose_pairs[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypoints_list[index.astype(int), 0])
                    #print(B)
                    A = np.int32(keypoints_list[index.astype(int), 1])
                    # cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
                    num.append(n)
            
            # plt.figure(figsize=[15,15])
            # plt.imshow(frameClone[:,:,[2,1,0]])
            # plt.savefig('./images/{}_test_paf.jpg'.format(idx_t))
            # plt.clf()
        
    
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

            #print(dicts)
            #print('Number of person :', max(dicts.values()))
            this_num_person=numofperson_sel[idx_t]
            counted_person=max(dicts.values())

            total_person += counted_person
            accuracy = abs((counted_person - this_num_person)) / this_num_person 
            if abs(accuracy) > 1:
                accuracy = 1            
                print('accuracy :', 1 - accuracy)
            total_accuracy += (1 - accuracy)
            print('AVG_accruracy :',total_accuracy / (idx_t + 1)*100.0)
            peoples=this_num_person
            if this_num_person != 0:
                total_person += counted_person
                if counted_person == this_num_person:
                    acc=1
                elif counted_person > this_num_person:
                    acc=this_num_person/counted_person
                    peoples=counted_person
                else:
                    acc=counted_person/this_num_person
            else:
                if counted_person == 0:
                    acc=1
                else:
                    total_person += counted_person
                    acc = 0
            if counted_person == this_num_person:
                image_acc=1
            else:
                image_acc=0
            print('Total Person :',total_person)
            #plt.savefig('./images/{}_test_paf.jpg'.format(idx_t))
            #exit()
        accs.update(acc,peoples)
        image_accs.update(image_acc,1)
        print('Count Accuracy :',accs.avg*100.0)
        print('Image Accuracy :',image_accs.avg*100.0)
        #print(len(personwiseKeypoints) - num)
    
    total_accuracy /= (idx_t + 1)
    print(total_accuracy)
