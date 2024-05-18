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


# In[3]:


analysis_metrics= cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1')


# In[4]:


pd.set_option('display.max_rows',10)
pd.set_option('display.max_columns',10)
pd.set_option('display.width',10)
print(analysis_metrics.columns.tolist())


# In[5]:


print(analysis_metrics.shape)


# In[6]:


units = cache.get_units()


# In[7]:


print(units.columns.tolist())


# In[8]:


print(units["ecephys_structure_acronym"])


# In[9]:


def get_metrics_by_structure(structure_name, metrics_name):
    num = 0
    metrics_sum = 0.0
    indexes = analysis_metrics[metrics_name].index
    values = analysis_metrics[metrics_name].values
    for i in range(len(indexes)):
        structure = units[units.index == indexes[i]]["ecephys_structure_acronym"].tolist()[0]
        if structure == structure_name:
            #print(structure)
            num+=1
            metrics_sum += np.float64(values[i])
    return metrics_sum/num, num
    #print(num, metrics_sum)


# In[10]:


structure_name_list = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam", "LGd", "LP", "APN","CA1", "DG", "SUB", "CA3"]
metrics_name = "g_osi_sg"
for structure_name in structure_name_list :
    metrics, num = get_metrics_by_structure(structure_name,metrics_name)
    print(metrics_name + " : "+ structure_name + " "+str(metrics)+" "+str(num))


# In[ ]:





# In[ ]:




