from asyncore import read
from dis import dis
import cv2 
import numpy as np 
import os 
import glob 

from detect import Detect



def calculate(h_matrix, detected_points):
    
    all_points = detected_points

    if 'edge' not in all_points.keys():
        print("edge is not detected. Need to try again!")
    else:
        bbox_edge_1, bbox_edge_2 = all_points['edge'][0], all_points['edge'][1]
        if bbox_edge_1[0][0] < bbox_edge_2[0][0]:
            point_1 = bbox_edge_1[2].reshape(2,1)
            point_2 = bbox_edge_2[3].reshape(2,1)
        else:
            point_1 = bbox_edge_2[2].reshape(2,1)
            point_2 = bbox_edge_1[3].reshape(2,1)
        p3 = (point_1[1] + point_2[1])/2
        point_1 = np.concatenate((point_1, np.ones((1, 1))), axis=0)
        point_2 = np.concatenate((point_2, np.ones((1, 1))), axis=0)
        homo_mat_inv = np.linalg.inv(h_matrix)
        world_p1 = np.matmul(homo_mat_inv, point_1)
        world_p2 = np.matmul(homo_mat_inv, point_2)
        world_p1 = world_p1 / world_p1[2]
        world_p2 = world_p2 / world_p2[2]
        distance_width = np.linalg.norm(world_p1 - world_p2)
        # print(distance_width)

    if 'scale' not in all_points.keys():
        print("scale is not detected. Need to try again!")
    else:
        bbox_edge = all_points['scale'].reshape(4,2)
        point_4 = bbox_edge[3].reshape(2,1)
        point_3 = np.array([bbox_edge[0][0], p3[0]]).reshape(2,1)
        point_3 = np.concatenate((point_3, np.ones((1, 1))), axis=0)
        point_4 = np.concatenate((point_4, np.ones((1, 1))), axis=0)
        world_p3 = np.matmul(homo_mat_inv, point_3)
        world_p4 = np.matmul(homo_mat_inv, point_4)
        world_p3 = world_p3 / world_p3[2]
        world_p4 = world_p4 / world_p4[2]
        distance_height = np.linalg.norm(world_p3 - world_p4)
        # print(distance_height)


        
    
    return distance_width, distance_height

        
    





