from fit_3DDFA import *
import cv2
import os

data_folder = "/mnt/raid/juan/StyleGANRelightingData/"

fit_3DMM = fit_3DDFA('gpu')

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

        #Load image
        id = int(dir.split("_")[1])
        name = "face_{:02d}".format(id)

        file = os.path.join(folder, "{}.png".format(name))

        #print (file)

        img = cv2.imread(file)

        #print (img.shape)

        # Fit 3DMM Model
        fit_3DMM.forward(img, folder, name)
        #input()

print ("Fitting 3DMM is Done")
#input()
