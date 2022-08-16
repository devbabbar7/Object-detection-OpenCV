import cv2
from my_utils import get_yolo_preds
import torch

with open("model_data/coco.names","r", encoding = "utf-8") as f:
    labels = f.read().strip().split("\n")

yolo_config_path = "model_data/yolov4.cfg"
yolo_weights_path = "model_data/yolov4.weights"

input_vid_path = "input_video/test2.mp4"

cuda = False
show_display = True

write_output = True
output_vid_path = "output_video/test2_yolov4.avi"

confidence_threshold = 0.5

overlapping_threshold = 0.3

net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)
if cuda:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

if __name__ == '__main__':
    get_yolo_preds(net, input_vid_path, output_vid_path, confidence_threshold
    , overlapping_threshold, write_output, show_display, labels)
    
