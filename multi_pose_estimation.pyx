import numpy as np
cimport numpy as np
import copy
import cv2


cdef getKeypoints(np.ndarray probMap, float threshold=0.1):
    cdef np.ndarray mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
    cdef np.ndarray mapMask = np.uint8(mapSmooth > threshold)
    cdef np.ndarray mapMasked = mapSmooth * mapMask
    cdef np.ndarray keypoints = None
    cdef np.ndarray contours
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.findContours(mapMasked, contours, hierarchy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cdef np.ndarray blobMask = np.zero_like(mapMasked)
    for cnt in contours:
        cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapMasked * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        if keypoints in None:
            keypoints = np.array([maxLoc + (probMap[maxLoc[1], maxLoc[0]],)])
        else:
            keypoints = np.concatenate((keypoints, maxLoc + (probMap[maxLoc[1], maxLoc[0]],)))
        # keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))
    return keypoints

cpdef getValidPairs(np.ndarray[np.float64_t, ndim=4] output, 
                     np.ndarray[np.int_t, ndim=2] mapIdx, 
                     list detected_keypoints, 
                     int frameWidth, int frameHeight, 
                     np.ndarray[np.int_t, ndim=2] pose_pairs):

    cdef list valid_pairs = []
    cdef list invalid_pairs = []
    cdef int n_interp_samples = 10
    cdef double paf_score_th = 0.1
    cdef double conf_th = 0.7
    cdef int k, i, j, nA, nB, max_j
    cdef double norm, maxScore, avg_paf_score
    cdef np.ndarray[np.float64_t, ndim=2] pafA
    cdef np.ndarray[np.float64_t, ndim=2] pafB
    cdef np.ndarray[np.int_t, ndim=2] interp_coord
    cdef np.ndarray[np.float64_t, ndim=2] paf_interp =None
    cdef np.ndarray[np.float64_t, ndim=1] paf_scores
    cdef np.ndarray[np.float64_t, ndim=2] valid_pair

    cdef np.ndarray d_ij=np.zeros(2)
    
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
                    d_ij = candB[j][:2] - candA[i][:2]
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
                    paf_interp = np.zeros((n_interp_samples, 2))
                    for k in range(len(interp_coord)):
                        paf_interp[k]=np.array([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
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

def getPersonwiseKeypoints(list[list[int]] valid_pairs, list[int] invalid_pairs, list[int] mapIdx, list[list[int]] pose_pairs, np.ndarray[np.float32_t, ndim=3] keypoints_list):

    cdef int[:, :] valid_pairs_arr, pose_pairs_arr
    cdef int[:] invalid_pairs_arr, mapIdx_arr
    cdef int k, i, j, found, person_idx, indexA, indexB, partAs_len, partBs_len, keypoints_len
    cdef double[:, :] personwiseKeypoints
    cdef double paf_score, keypoint_score, partAs_i, partBs_i

    personwiseKeypoints = -1 * np.ones((0, 19), dtype=np.float32)

    for k in range(len(mapIdx)):
        if k not in invalid_pairs:
            valid_pairs_arr = np.array(valid_pairs[k], dtype=np.int32)
            invalid_pairs_arr = np.array(invalid_pairs, dtype=np.int32)
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
                    keypoint_score = sum(keypoints_list[partBs_i.astype(np.int32), 2])
                    paf_score = valid_pairs_arr[i, 2]
                    personwiseKeypoints[person_idx][indexB] = partBs_i
                    personwiseKeypoints[person_idx][-1] += keypoint_score + paf_score

                elif not found and k < 17:
                    row = -1 * np.ones(19, dtype=np.float32)
                    partAs_i = partAs[i]
                    partBs_i = partBs[i]
                    keypoint_score = sum(keypoints_list[valid_pairs_arr[i, :2].astype(np.int32), 2])
                    paf_score = valid_pairs_arr[i, 2]
                    row[indexA] = partAs_i
                    row[indexB] = partBs_i
                    row[-1] = keypoint_score + paf_score
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])

    return personwiseKeypoints