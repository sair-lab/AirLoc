import os
import cv2
import copy
import torch
import numpy as np
import pickle


def overlay_points(images, points_output, color=(255,0,0)):
  new_images = []
  for image, points_o in zip(images, points_output):
    points = points_o['points'].cpu().numpy()
    if len(points) == 0:
      print("no points")
    for i in range(len(points)):
      x = points[i][1].astype(int)
      y = points[i][0].astype(int)
      if x < 0:
        continue
      cv2.circle(image, (x,y), 1, color, thickness=-1)
    new_images.append(image)
  return new_images


pkl_file = "/home/aryan/Mp3d_dataset/x_view/mp3d/qoiz87JEwZ2/rooms/room2/points/0.pkl"
img_path = "/home/aryan/Mp3d_dataset/x_view/mp3d/qoiz87JEwZ2/rooms/room2/raw_data/0_rgb.png"

src = [cv2.imread(img_path)]


with open(pkl_file, 'rb') as f:
    data = [pickle.load(f)]
    
img = overlay_points(src,data)
# print(img[0].type)
cv2.imshow('image',img[0])

cv2.waitKey(0)
cv2.destroyAllWindows()

    
