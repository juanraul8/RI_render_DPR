from detect_landmark import *
import numpy as np
import cv2

data_folder = "/mnt/raid/juan/StyleGANRelightingData/"

detect_landmark = detect_landmark()

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

        #print (file)

        subFolder = os.path.join(folder, 'render')

        if not os.path.exists(subFolder):

            print ("Folder {} does not exist".format(subFolder))

            continue

        # Load albedo image

        file = os.path.join(subFolder, 'albedo_3DFFA.png')

        if os.path.exists(file):
            print("Albedo already exist")
            continue

        file = os.path.join(subFolder, 'albedo.png')
        img = cv2.imread(file)

        # Landmarks Detection
        albedo_landmark = detect_landmark.detect(img)

        if albedo_landmark is None:
            continue
        else:
            detect_landmark.save_landmark(albedo_landmark,
                                          os.path.join(subFolder, 'albedo_detected.txt'))
            #detect_landmark.draw_landmark(albedo_landmark, img,
            #                              os.path.join(subFolder, 'albedo_3DDFA.png'))

        input()

print ("Landmarks Detection is Done")