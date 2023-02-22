import json
import numpy as np
import math
import os
COCO_TO_OURS = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                        'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 
                        'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

def get_accuracy():
    file_path = os.path.join('..\..\..\data\dataset\coco2017\\annotations','person_keypoints_val2017.json')

    data = {}

    with open(file_path, 'r') as outfile:
        data = json.load(outfile)

    # id vs filename



    #print(data.keys())    
    image = {}
    info = {}
    for i in range(len(data['images'])):
        image[data['images'][i]['id']] = {}
        info[data['images'][i]['id']] = []
        #data['images']['id']

    #print(info)
    #exit()
    #for idx in data['images']:
        #print(idx['id'])
    #print(len(data['annotations']))
    #print(data['annotations'][0].keys())
    #print(data['annotations'][0]['keypoints'])
    #print(data['annotations'][1]['keypoints'])

    persons = {}
    shit = []
    for i in range(len(data['annotations'])):
        persons[data['annotations'][i]['image_id']] = []
      
    person_centers = []
    #info = {}rs = []
    for i in range(len(data['annotations'])):
        dic = {}
        dic['keypoints'] = np.zeros((17,3)).tolist()
        name = data['annotations'][i]['image_id']
        #name_2 = data['images'][i]['id']
        #info[name_2] = []

        if data['annotations'][i]['num_keypoints'] < 5 or data['annotations'][i]['area'] < 32 * 32:
            continue
        
        person_center = [data['annotations'][i]['bbox'][0] + data['annotations'][i]['bbox'][2] / 2.0, data['annotations'][i]['bbox'][1] + data['annotations'][i]['bbox'][3] / 2.0]
        scale = data['annotations'][i]['bbox'][3] / 368.0

        flag = 0
        for pc in person_centers:
            dis = math.sqrt((person_center[0] - pc[0]) * (person_center[0] - pc[0]) + (person_center[1] - pc[1]) * (person_center[1] - pc[1]))
            if dis < pc[2] * 0.3:
                flag = 1;
                break
        if flag == 1:
            continue
        
        for part in range(17):
            dic['keypoints'][part][0] = data['annotations'][i]['keypoints'][part * 3]
            dic['keypoints'][part][1] = data['annotations'][i]['keypoints'][part * 3 + 1]
            if data['annotations'][i]['keypoints'][part * 3 + 2] == 2:
                dic['keypoints'][part][2] = 1
            elif data['annotations'][i]['keypoints'][part * 3 + 2] == 1:
                dic['keypoints'][part][2] = 0
            else:
                dic['keypoints'][part][2] = 2

        persons[name].append(dic)

    for name, person in persons.items():
        for p in person:
            dic = {}
            #dic['pos'] = person['objpos']
            dic['keypoints'] = np.zeros((18,3)).tolist()
            #dic['scale'] = person['scale']
            for i in range(17):
                dic['keypoints'][COCO_TO_OURS[i]][0] = p['keypoints'][i][0]
                dic['keypoints'][COCO_TO_OURS[i]][1] = p['keypoints'][i][1]
                dic['keypoints'][COCO_TO_OURS[i]][2] = p['keypoints'][i][2]
            dic['keypoints'][1][0] = (p['keypoints'][5][0] + p['keypoints'][6][0]) * 0.5
            dic['keypoints'][1][1] = (p['keypoints'][5][1] + p['keypoints'][6][1]) * 0.5
            if p['keypoints'][5][2] == p['keypoints'][6][2]:
                dic['keypoints'][1][2] = p['keypoints'][5][2]
            elif p['keypoints'][5][2] == 2 or p['keypoints'][6][2] == 2:
                dic['keypoints'][1][2] = 2
            else:
                dic['keypoints'][1][2] = 0
            #print(dic)
            info[name].append(dic)

    #print(list(info.keys()))
    #exit()
    #exit()
    #print(info)
    #print(len(info[0]['keypoints']))
    numofperson = []

    #for i in ra

    for k, val in info.items():
        #print(val)
        final = {}
        for idx in keypointsMapping:
            final[idx] = 0
        for v in val:
            #print(v)
            for key in v.values():
                #print(key)
                for idx, m in enumerate(key):
                    if m[2] != 2:
                        final[keypointsMapping[idx]] += 1

    
        person = max(final.values())
        numofperson.append(person)

    #print(numofperson)
    #np.array()
    #np.save('./data/number_of_people.npy', np.array(numofperson))
    #print(len(numofperson))

    return numofperson, list(info.keys())
