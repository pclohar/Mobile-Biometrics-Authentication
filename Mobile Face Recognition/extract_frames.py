# -*- coding: utf-8 -*-

#/Users/pallavilohar/Documents/Academic Material/Semester 3/Biometrics Auth/Face Data Collection/.Task#1.mp4.icloud

import os
import cv2

root = '/Volumes/Documents/BIOMETRICS_AUTH/Face Recognition Project/E_Molina' # Replace it with your path 

base_dir = 'F20DataImages'
for path, subdirs, files in os.walk(root):
    for name in files:
      frame_count = 1
      folders = path.split('/')
      folder = folders[-1]
      filepath = os.path.join(path, name)
      #print(filepath)
      vidcap = cv2.VideoCapture(filepath)
      success,image = vidcap.read()
      

      if not vidcap.isOpened():
          print("Error opening video :" + filepath)


      count = 0
      directory = os.path.join('F20DataImages' , folder)
      if not os.path.exists(directory):
        os.makedirs(directory)
      while success:
        cv2.imwrite(os.path.join('F20DataImages' , folder, name.split('.mp4')[0]+"-Frame%d.png" % frame_count), image)        
        success,image = vidcap.read()
        count += 35
        frame_count += 1
        vidcap.set(1, count)
        if frame_count > 29:
          break