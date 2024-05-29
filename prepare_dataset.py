import os
import json
import numpy as np
import math
import random
import pickle 


PATH = '../../datasets/ue4'
actions = ['crouch', 'walk', 'jump', 'stand', 'death', 'reload', 'fire']
norm = (-1.0, 1.0)
np.random.seed(8)

ANNOTATIONS_PATH = 'datasets/action_classification/'
DATASET_TRAIN = '{}train_data_1_in.data'.format(ANNOTATIONS_PATH)
DATASET_VAL = '{}val_data_1_in.data'.format(ANNOTATIONS_PATH)

point_keys = ['point_{}'.format(i) for i in range(4, 16)]

files = os.listdir(PATH)

make_vector = lambda data, point_keys : '; '.join(['; '.join([str(element) for element in i]) for i in data])

def rotate_pitch(matrix, angle):
    rotation_matrix_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    matrix = np.dot(matrix, rotation_matrix_pitch)
    return matrix


def rotate_yaw(matrix, angle):
    rotation_matrix_yaw = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [np.sin(angle), 0, np.cos(angle)]
    ])
    matrix = np.dot(matrix, rotation_matrix_yaw)
    return matrix


def rotate_roll(matrix, angle):
    rotation_matrix_roll = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    matrix = np.dot(matrix, rotation_matrix_roll)
    return matrix

def reculc_coordinates(value, point_keys):
    skeleton = np.array([value[i] for i in point_keys])
    skeleton -= skeleton[point_keys.index('point_7')]

    x, y = skeleton[point_keys.index('point_6')][0], skeleton[point_keys.index('point_6')][1]
    roll = np.round(np.arctan2(y, x), 4)
    skeleton = rotate_roll(skeleton, roll)

    x, y, z = skeleton[point_keys.index('point_6')]
    yaw = np.round(-np.arctan2(z, np.sqrt(x**2 + y**2)), 4)
    skeleton = rotate_yaw(skeleton, yaw)

    x, y, z = skeleton[point_keys.index('point_4')]
    roll = -np.round(np.arctan2(y, z), 4)
    skeleton = rotate_pitch(skeleton, roll)
    skeleton = np.where(np.abs(skeleton) < 0.007, 0, skeleton)
    skeleton = np.round(skeleton, 3)
    return skeleton

def normalize(arr, t_min, t_max):
    diff = t_max - t_min
    diff_arr = np.max(arr) - np.min(arr)
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            coord = arr[i][j]
            coord = (((coord - np.min(arr))*diff)/diff_arr) + t_min
    return arr

def save_data(data, file_path):
    keypoints_data = []
    labels_data = []

    for keypoints, lable in data:
        keypoints_data.append([keypoints])
        labels_data.append(lable)
        
    keypoints_data = np.array(keypoints_data)
    labels_data = np.array(labels_data)

    with open(file_path, 'wb') as file: 
        pickle.dump([keypoints_data, labels_data], file)

data = []
    
for file_name in files:
    with open('{}/{}'.format(PATH, file_name), 'r') as f:
        file_data = json.load(f)

    file_name = file_name.lower()
    
    for ind, action in enumerate(actions):
        if action in file_name:
            for key, value in file_data.items():
                vector = reculc_coordinates(value, point_keys)
                vector = normalize(vector,
                                   norm[0],
                                   norm[1])
                data.append([vector, ind])

random.shuffle(data)

save_data(data[:int(len(data)*0.8)], DATASET_TRAIN)
save_data(data[int(len(data)*0.8):], DATASET_VAL)



