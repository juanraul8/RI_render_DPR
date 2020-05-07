'''
    python2, use pytorch_conda environment
'''
import numpy as np
import cv2
from generate_node import *
import sys
import os

data_folder = "/mnt/raid/juan/StyleGANRelightingData/"

for root, dirs, files in os.walk(data_folder):

    print ("Walking folders")

    dirs.sort()

    depth = root.count(os.sep) - data_folder.count(os.sep)

    #Debugging
    #print(root)
    #print (dirs)
    #print (files)
    #print (depth)

    if root == data_folder or depth > 0:
        continue

    for dir in dirs:
        folder = os.path.join(root, dir)

        print ("Processing folder: {}".format(folder))

        subFolder = os.path.join(folder, 'render')

        if not os.path.exists(subFolder):
            print("Folder {} does not exist".format(subFolder))

            continue

        # Load Data

        id = int(dir.split("_")[1])
        name = "face_{:02d}".format(id)

        file = os.path.join(subFolder, 'albedo.png')
        img = cv2.imread(file)
        file = os.path.join(folder, name + '_detected.txt')
        face_landmark = np.loadtxt(file).T

        # Create Node
        file = os.path.join(subFolder, 'albedo_detected.txt')
        albedo_landmark = np.loadtxt(file).T
        get_node(albedo_landmark, face_landmark, img.shape[1], img.shape[0], subFolder)

        # ARAP
        triangle_path = os.path.join(subFolder, 'triangle.txt')
        correspondence_path = os.path.join(subFolder, 'correspondence.txt')
        saveName = os.path.join(subFolder, 'arap.obj')

        cmd = '../utils/libigl_arap/my_arap ' + \
              triangle_path + ' ' + correspondence_path + ' ' \
              + saveName + ' ' + str(img.shape[1]) + ' ' + str(img.shape[0])
        os.system(cmd)

        print ("Test Done")
        input()

print ("Generating Node is Done")
#input()