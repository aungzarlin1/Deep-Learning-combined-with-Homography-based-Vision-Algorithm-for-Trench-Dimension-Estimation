from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


def points(output):
    all_points = {}
    class_name = []
    for i in range(output.shape[0]):
        c1 = output[i, 1:3].detach().cpu().numpy()
        c2 = output[i, 3:5].detach().cpu().numpy()
        point1 = c1
        point2 = c2[0], c1[1]
        point3 = c2
        point4 = c1[0], c2 [1]
        pts = np.array([point1, point2, point3, point4]).reshape(1,4,2)
        cls = int(output[i][-1])
        label = "{0}".format(cfg_classes[cls])
        if label not in class_name:
            class_name.append(label)
            all_points[label] = pts
        else:
            temp = all_points[label]
            temp = np.concatenate((temp, pts), axis=0)
        #     temp.append(pts)
            all_points[label] = temp
            # all_points[label] += pts
    return all_points


def img_write(output, img):    
    
    for det in output:         
        class_id = det[-1].int().detach().cpu().numpy()
        c1 = det[1:3].int().detach().cpu().numpy()
        c2 = det[3:5].int().detach().cpu().numpy()   
           
        color = colors[int(class_id) % len(colors)]
        cv2.rectangle(img, c1, c2, color, 1)          
        cv2.putText(img, cfg_classes[class_id], (c1[0], c1[1] - 6), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)        

    return img


def run_yolov4(image_file, image_name, cfg_path, weight_path, backend='opencv'):

    global cfg_classes, colors 

## CONFIGURE ##
    # image_file = "../data/images/"
    # image_name = 'frame-001'
    # cfg_path = "../cfg/yolov4-custom.cfg" 
    # weight_path = "../weight/yolov4-custom_best.weights"
    model_width = 608
    model_height = 608
    batch_size = 1
    confidence = 0.3
    nms_thesh = 0.4
    num_classes = 3
    cfg_classes = load_classes("../data/obj_scale.names")
    colors = [[255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 0, 255]]
    # backend = "opencv"  # opencv or pytorch
    ## CONFIGURE ##


    img = cv2.imread(image_file + image_name +'.png')   ## Change file name

    if not os.path.exists("../result"):
        os.makedirs("../result")

    if backend == "opencv":
        print("Backend opencv")
        net = cv2.dnn.readNet(weight_path, cfg_path)
        #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(model_width, model_height), scale=1/255, swapRB=True)
        classes, scores, boxes = model.detect(img, confidence, nms_thesh)
        prediction = []
        labels = []
        for (classid, score, box) in zip(classes, scores, boxes):
            # print("class id",classid)
            # print("score",score )
            # print("box",box )
            label = cfg_classes[classid]
            labels.append(label)
            det_box = [0, box[0], box[1], box[0]+box[2], box[1]+box[3],score,score, classid ]
            prediction.append(det_box)
        prediction = torch.tensor(prediction)
        output =  prediction
        print("Detected classes :", labels)
        # print(prediction)
        img = img_write(output, img)

    else:
        print("Backend pytorch")
        CUDA = torch.cuda.is_available()
        model = Darknet(cfg_path)
        model.load_weights(weight_path)
        model.net_info["height"] = model_height
        model.net_info["width"] = model_width
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0 
        assert inp_dim > 32

        #If there's a GPU availible, put the model on GPU

        if CUDA:
            model.cuda()

        model.eval()

        img = letterbox_image(img, (inp_dim, inp_dim))
        # cv2.imwrite('test.jpg', img)
        img = cv2.imread('test.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        batch = prep_image(img, inp_dim)   # single image batch

        print("type ", type(batch), batch.size())

        if CUDA:
            batch = batch.cuda()
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

        # previois prediction shape ([1, 22743, 85])
        prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)
        output =  prediction
        # now prediction shape is [3, 8] 
        print("prediciton is ", prediction)

        if CUDA:
            torch.cuda.synchronize()       
        try:
            output
        except NameError:
            print ("No detections were made")
            exit()
        img = img_write(output, img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    cv2.imwrite("../result/res" + image_name+".jpg", img)

    return output



def Detect(image_file, image_name, cfg_path, weight_path, backend):
    output = run_yolov4(image_file, image_name, cfg_path, weight_path, backend)
    all_points = points(output)
    torch.cuda.empty_cache()
    return all_points 