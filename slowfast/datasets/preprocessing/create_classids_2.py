import json
import numpy as np
import os
import shutil

srclist = 'classids.json'

<<<<<<< HEAD
videodir = '/ws/data/miniKinetics400_slowfast_5/train_256'
=======
videodir = '/ws/data/train_256'
>>>>>>> endstop
outlist = 'trainlist.txt'

# videodir = 'YOUR_DATASET_FOLDER/val/'
# outlist = 'vallist.txt'


json_data = open(srclist).read()
clss_ids = json.loads(json_data)

folder_list = os.listdir(videodir)

was = np.zeros(1000)
new_clss_ids = dict()
for n, m in clss_ids.items():
    names = n
    new_name = "_".join(names.split())
    # new_name = ''
    # for j in range(len(names)):
    #     if j < len(names) - 1:
    #         new_name = new_name + '_'
    print(new_name)
    new_clss_ids[new_name] = m

with open('classids_miniKinetics200.json', 'w') as f:
    json.dump(new_clss_ids, f)