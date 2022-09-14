from distance import calculate
from homography import homography 
# from distance import * 
from detect import Detect

image_file = '../data/images/'
image_name = 'frame-001'
image = image_file + image_name + ".png"
cfg_path = "../cfg/yolov4-custom.cfg" 
weight_path = "../weight/yolov4-custom_best.weights"
backend = 'opencv'

detection = Detect(image_file, image_name, cfg_path, weight_path, backend)
print('detection points:', detection)

# method = 'auto', 'manual', 'yolo'
# h_matrix = homography(image, detection, method='yolo')
# print("h matrix:",h_matrix)
h_matrix = homography(image, detection)
print(h_matrix)
# h_matrix = homography(image, detection, method='manual')
# print(h_matrix)
width, height = calculate(h_matrix, detection)
print('width:', width)
print('height', height)