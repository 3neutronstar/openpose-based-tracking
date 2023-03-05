import numpy as np
cimport numpy as np
import copy
import cv2


def getKeypoints(np.ndarray probMap, float threshold=0.1):
    cdef np.ndarray mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
    cdef np.ndarray mapMask = np.uint8(mapSmooth > threshold)
    cdef np.ndarray mapMasked = mapSmooth * mapMask
    cdef list keypoints = []
    cdef tuple contours
    contours = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    # cv2.findContours(mapMasked, contours, hierarchy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cdef np.ndarray blobMask = np.zeros_like(mapMasked)
    for cnt in contours:
        cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapMasked * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))
    return keypoints

def getValidPairs(np.ndarray[float, ndim=4] output, 
                     np.ndarray[int, ndim=2] mapIdx, 
                     list detected_keypoints, 
                     int frameWidth, int frameHeight, 
                     np.ndarray[int, ndim=2] pose_pairs):

    cdef list valid_pairs = []
    cdef list invalid_pairs = []
    cdef int n_interp_samples = 10
    cdef float paf_score_th = 0.1
    cdef float conf_th = 0.7
    cdef int k, i, j, nA, nB, max_j
    cdef float norm, maxScore, 
    cdef double avg_paf_score
    cdef np.ndarray[float, ndim=2] pafA
    cdef np.ndarray[float, ndim=2] pafB
    cdef np.ndarray[int, ndim=2] interp_coord
    cdef np.ndarray[float, ndim=2] paf_interp =None
    cdef np.ndarray[double, ndim=1] paf_scores
    cdef np.ndarray[double, ndim=2] valid_pair

    cdef np.ndarray d_ij=np.zeros(2,dtype=np.float64)
    
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
            valid_pair = np.zeros((0,3),dtype=np.float64)
            # 조인트들 사이의 모든 interpolated points 집합에 대해 PAF를 계산한다
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):         
                    # 두 joint 사이의 distance vector(d_ij)계산
                    d_ij = np.array([candB[j][:2][0]-candA[i][:2][0],candB[j][:2][1]-candA[i][:2][1]])
                    norm = np.linalg.norm(d_ij)
                    # distance vector(d_ij)를 normalize
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    # p(u)
                    interp_coord = np.array(list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                                            np.linspace(candA[i][1], candB[j][1], num=n_interp_samples))),dtype=np.int32)
                    # L(p(u))
                    paf_interp = np.zeros((n_interp_samples, 2),dtype=np.float32)
                    for k in range(len(interp_coord)):
                        paf_interp[k]=np.array([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ],np.float32) 
                    # E
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/float(len(paf_scores))

                    # Connection이 valid한지 체크하기 위해 threshold로 걸러낸다  
                    if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1
                # Valid한 connection들 저장
                if found:     
                    valid_pair = np.append(valid_pair, [[float(candA[i][3]), float(candB[max_j][3]), float(maxScore)]], axis=0)

            # valid_pair과 valid_pairs 다름 주의 - valid_pairs는 global variable
            valid_pairs.append(valid_pair)
            
        # Keypoint가 detect되지 않았을 경우
        else: 
            print("No Connection : k = {}".format(k))
            invalid_pairs.append(k)
            valid_pairs.append([])
    #print(valid_pairs)
    return np.array(valid_pairs), np.array(invalid_pairs)

def getPersonwiseKeypoints(np.ndarray valid_pairs, np.ndarray invalid_pairs, np.ndarray[int,ndim=2] mapIdx, np.ndarray[int, ndim=2] pose_pairs, np.ndarray[double, ndim=2] keypoints_list):

    cdef np.ndarray[int,ndim=2] valid_pairs_arr, mapIdx_arr
    cdef np.ndarray[int] pose_pairs_arr
    cdef int k, i, j, found, person_idx, indexA, indexB, partAs_len, partBs_len, keypoints_len
    cdef np.ndarray[double,ndim=2] personwiseKeypoints
    cdef double paf_score, keypoint_score, partAs_i, partBs_i
    cdef np.ndarray[double] row

    personwiseKeypoints = -1 * np.ones((0, 19), dtype=np.float64)

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            valid_pairs_arr = np.array(valid_pairs[k], dtype=np.int32)
            mapIdx_arr = np.array(mapIdx, dtype=np.int32)
            pose_pairs_arr = np.array(pose_pairs[k], dtype=np.int32)
            partAs = valid_pairs_arr[:, 0]
            partBs = valid_pairs_arr[:, 1]
            indexA, indexB = pose_pairs_arr
            partAs_len = len(partAs)
            partBs_len = len(partBs)

            for i in range(partAs_len):
                found = 0
                person_idx = -1

                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        person_idx = j
                        found = 1
                        break

                if found:
                    partAs_i = partAs[i]
                    partBs_i = partBs[i]
                    keypoint_score = np.sum(keypoints_list[int(partBs_i), 2])
                    paf_score = valid_pairs_arr[i, 2]
                    personwiseKeypoints[person_idx][indexB] = partBs_i
                    personwiseKeypoints[person_idx][-1] += keypoint_score + paf_score

                elif not found and k < 17:
                    row = -1 * np.ones(19, dtype=np.float64)
                    partAs_i = partAs[i]
                    partBs_i = partBs[i]
                    keypoint_score = np.sum(keypoints_list[valid_pairs_arr[i, :2].astype(np.int32), 2])
                    paf_score = valid_pairs_arr[i, 2]
                    row[indexA] = partAs_i
                    row[indexB] = partBs_i
                    row[-1] = keypoint_score + paf_score
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])

    return personwiseKeypoints