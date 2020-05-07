from my_warp import *
from my_loadMesh import *
import os
import numpy as np
import cv2

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

        saveSubFolder = os.path.join(folder, 'warp')

        if not os.path.exists(saveSubFolder):
            os.makedirs(saveSubFolder)

        #Load mesh
        id = int(dir.split("_")[1])
        name = "face_{:02d}".format(id)
        file = os.path.join(folder, "{}.png".format(name))
        img = cv2.imread(file)

        meshName = os.path.join(subFolder, 'arap.obj')
        vertex, faces, tx_coord = load_mesh(meshName)

        #Warping function
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        vertex = np.transpose(vertex, (1, 0))
        faces = np.transpose(faces, (1, 0))
        tx_coord = np.transpose(tx_coord, (1, 0))

        UV, depth_buffer = get_warpedUV(vertex, faces - 1, tx_coord, imgHeight, imgWidth, c=3)
        vis_img = (UV * 255.0).astype(np.uint8)

        cv2.imwrite(os.path.join(saveSubFolder, 'UV_warp.png'), vis_img)

        #Get warped image
        UV = UV[:, :, 0:2]
        UV = np.reshape(UV, (-1, 2))

        get_warpedImage(UV, imgWidth, imgHeight, subFolder, saveSubFolder)
        albedo_img = cv2.imread(os.path.join(saveSubFolder, 'albedo.png')).astype(np.float)
        combine_img = 0.5 * img.astype(np.float) + 0.5 * albedo_img
        cv2.imwrite(os.path.join(saveSubFolder, 'combine_albedo.png'), combine_img.astype(np.uint8))

        print("Test Done")
        input()

print ("Warping is Done")
#input()
