#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache


# In[2]:


# this path determines where downloaded data will be stored
manifest_path = os.path.join('E:/ecephys_cache_dir/', "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path,fetch_tries=1,timeout=2000*60)

print(cache.get_all_session_types())
sessions = cache.get_session_table()
brain_observatory_type_sessions = sessions[sessions["session_type"] == "brain_observatory_1.1"]
brain_observatory_type_sessions


# In[3]:



unit_dict = {}
num_id = 0
repeat_id = 0
def load_dict(session_id):
    global num_id
    global repeat_id
    path = "E:\\ecephys_cache_dir\\allen_spike_movie_3\\"
    structure_name = "CA1"
    file_name = path + "nm_unitid_" + str(session_id) + "_" + structure_name + ".npy"
    if not os.path.exists(file_name):
        return
    data = np.load(file_name)
    for id in data:
        if id not in unit_dict:
            unit_dict[id] = num_id
            #print(id)
            num_id+=1
        else :
            repeat_id+=1


# In[4]:


def load_spike(session_id):
    global num_id
    global spike_unit_sum
    global spike_unit_num
    path = "E:\\ecephys_cache_dir\\allen_spike_movie_3\\"
    structure_name = "CA1"
    spike_file_name = path + "nm_spike_" + str(session_id) + "_" + structure_name + ".npy"
    if not os.path.exists(spike_file_name):
        return
    spike_data = np.load(spike_file_name)
    unit_file_name = path + "nm_unitid_" + str(session_id) + "_" + structure_name + ".npy"
    unit_data = np.load(unit_file_name)
    for i in range(3600):
        for j in range(spike_data.shape[1]):
            unit_id = unit_dict[unit_data[j]]
            spike_unit_sum[i][unit_id]+=spike_data[i][j]
            #if spike_data[i][j] !=0 :
                #print(session_id, i, j, spike_data[i][j])
            spike_unit_num[i][unit_id]+=1
            


# In[5]:


#sessions_id = brain_observatory_type_sessions.index.values

sessions_id = [719161530]
for session_id in sessions_id:
    print(session_id)
    load_dict(session_id)
print(num_id)
print(repeat_id)

spike_unit_sum = np.zeros((3600, num_id))
spike_unit_num = np.zeros((3600, num_id))
spike_unit_ave = np.zeros((3600, num_id))
for session_id in sessions_id:
    print(session_id)
    load_spike(session_id)
for i in range(3600):
    for j in range(num_id):
        spike_unit_ave[i][j] = spike_unit_sum[i][j]/spike_unit_num[i][j]
print(spike_unit_ave[0])
np.save("E:\\ecephys_cache_dir\\allen_spike_ave_movie_3_one_session\\movie_03_train_spike_CA1_719161530.npy",spike_unit_ave)


# In[6]:


print(spike_unit_sum[0])


# In[7]:


import numpy as np
spike_unit_ave = np.load("E:\\ecephys_cache_dir\\allen_spike_ave_movie_3_one_session\\movie_03_train_spike_CA1_719161530.npy")
#spike_unit_ave = np.transpose(spike_unit_ave, [1,0])
num_id = spike_unit_ave.shape[1]
print(num_id)
print(spike_unit_ave.shape)
import random
list = [i for i in range(3600)]
random.shuffle(list)
spike_train = np.zeros((3000, num_id))
spike_test = np.zeros((600, num_id))
for i in range(3000):
    for j in range(num_id):
        spike_train[i][j] = spike_unit_ave[list[i]][j]
for i in range(600):
    for j in range(num_id):
        spike_test[i][j] = spike_unit_ave[list[3000+i]][j]


# In[8]:


import h5py
file_name = "E:\\ecephys_cache_dir\\natural_movie_templates\\natural_movie_3.h5"
data = np.load(file_name)
pic_train = np.zeros((3000, 256, 256))
pic_test = np.zeros((600, 256, 256))
for i in range(3000):
    pic_train[i] = data[list[i]][24:280,176:432]/255.0
for i in range(600):
    pic_test[i] = data[list[i+3000]][24:280,176:432]/255.0


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
print(pic_train.shape)
for i in range(10):
    pic = pic_test[i]
    plt.imshow(pic,cmap="gray")
    plt.show()


# In[10]:


np.save("F:\\20201015_decodingDataset_code\\simulated data1\\movie_03_train_spike_os_CA1_719161530.npy",spike_train)
np.save("F:\\20201015_decodingDataset_code\\simulated data1\\movie_03_test_spike_os_CA1_719161530.npy", spike_test)
np.save("F:\\20201015_decodingDataset_code\\simulated data1\\movie_03_train_pic_os_CA1_719161530.npy", pic_train)
np.save("F:\\20201015_decodingDataset_code\\simulated data1\\movie_03_test_pic_os_CA1_719161530.npy", pic_test)


# In[43]:


import os
path = "E:\\ecephys_cache_dir\\allen_spike_movie_3"
name_list = set()
dataname = os.listdir(path)
for data in dataname :
    data = data.split("_")[3]
    name = data.split(".")[0]
    print(name)
    if name not in name_list :
        name_list.add(name)
print(name_list)


# In[3]:


import numpy as np
#structure_name_list = [ "CA1", "CA3", "DG", "SUB", "ProS", "LGd", "LP", "APN"]
#structure_name_list = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam","CA1", "CA3", "DG", "SUB", "ProS", "LGd", "LP", "APN"]
#structure_name_list = name_list
#structure_name_list = ["LGd","LP","APN"]
structure_name_list = ["VISp"]
#structure_name_list = ["SUB"]
file_name = "E:\\ecephys_cache_dir\\allen_spike_ave_movie_3\\movie_03_train_spike_seq_"
size_sum = 0
for structure_name in structure_name_list :
    data = np.load(file_name + structure_name + ".npy")
    size = data.shape[1]
    print(structure_name, size)
    size_sum += size
print("Total Num : ", size_sum)
spike_ave = np.zeros((3600, size_sum))
size_sum = 0
for structure_name in structure_name_list :
    data = np.load(file_name + structure_name + ".npy")
    size = data.shape[1]
    for i in range(3600):
        for j in range(size):
            spike_ave[i][size_sum + j] = data[i][j]
    #print(size)
    size_sum += size
print(np.mean(spike_ave))
print(np.std(spike_ave))


# In[ ]:





# In[4]:


print(spike_ave[3599][100:110])
print(np.mean(spike_ave))
print(np.std(spike_ave))


# In[5]:


import random
#unit_num = size_sum
unit_num = 100
list = [i for i in range( 3600)]
list2 = [i for i in range(size_sum)]
#list3 = [i for i in range(3600)]
random.seed(19990320)
random.shuffle(list)
random.shuffle(list2)
#random.shuffle(list3)
print(list[:30])
print(list2[:30])
#print(list3[:30])
spike_train = np.zeros((3200, unit_num))
spike_test = np.zeros((400, unit_num))
for i in range(3200):
    for j in range(unit_num):
        spike_train[i][j] = spike_ave[list[i]][list2[j%size_sum]]
for i in range(400):
    for j in range(unit_num):
        spike_test[i][j] = spike_ave[list[3200+i]][list2[j%size_sum]]


# In[6]:


import h5py
from PIL import Image as Img
file_name = "E:\\ecephys_cache_dir\\natural_movie_templates\\natural_movie_3.h5"
data = np.load(file_name)
pic_train = np.zeros((3200, 256, 256))
pic_test = np.zeros((400, 256, 256))
for i in range(3200):
    #img = Img.fromarray(data[list[i]])
    #img = img.resize((256,256))
    #pic_train[i] = np.asarray(img)/255.0
    pic_train[i] = data[list[i]][24:280,176:432]/255.0
    
for i in range(400):
    #img = Img.fromarray(data[list[i+3200]])
    #img = img.resize((256,256))
    #pic_test[i] = np.asarray(img)/255.0
    pic_test[i] = data[list[i+3200]][24:280,176:432]/255.0


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
print(pic_train.shape)
for i in range(10,30):
    pic = pic_test[i]
    plt.imshow(pic,cmap="gray")
    plt.show()


# In[8]:


np.save("F:\\20201015_decodingDataset_code\\simulated data1\\movie_03_train_spike_19990320_VISp_100.npy",spike_train)
np.save("F:\\20201015_decodingDataset_code\\simulated data1\\movie_03_test_spike_19990320_VISp_100.npy", spike_test)
np.save("F:\\20201015_decodingDataset_code\\simulated data1\\movie_03_train_pic_19990320_VISp_100.npy", pic_train)
np.save("F:\\20201015_decodingDataset_code\\simulated data1\\movie_03_test_pic_19990320_VISp_100.npy", pic_test)

