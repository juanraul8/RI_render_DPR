import os
import numpy as np
from getObj_3DDFA import *
from my_render import *
import time

data_folder = "/mnt/raid/juan/StyleGANRelightingData/"

modelFolder = '../data/3DMM/'
triangle_info_path = os.path.join(modelFolder, 'model_info.mat')
objPath = os.path.join(modelFolder, 'BFM_UV.mat')
mtl_path = '../data/3DMM/3DMM_normal.obj.mtl'

getObj = getObj_3DDFA(triangle_info_path, objPath, mtl_path)

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

        id = int(dir.split("_")[1])
        name = "face_{:02d}".format(id)
        file = os.path.join(folder, "{}.png".format(name))

        #Create new object
        src = os.path.join(folder, name + '.obj')
        dst = os.path.join(folder, name + '_new.obj')
        getObj.create_newObj(src, dst)

        #Rendering
        saveFolder = os.path.join(folder, 'render')

        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)

        begin_time = time.time()

        my_render(file, dst, modelFolder, saveFolder)
        print('dealing with %s used %s seconds' % (file, time.time() - begin_time))
        input()

print ("Rendering is Done")
input()