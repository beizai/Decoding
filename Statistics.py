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


sessions = cache.get_session_table()
brain_observatory_type_sessions = sessions[sessions["session_type"] == "brain_observatory_1.1"]
brain_observatory_type_sessions.head()


# In[4]:


brain_observatory_type_sessions.brain_observatory_type_sessions()


# In[5]:


analysis_metrics1 = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1')


# In[6]:



list1 = analysis_metrics1.columns.tolist()


# In[7]:


list1


# In[8]:


analysis_metrics1["fano_ns"]


# In[9]:


session_id = 715093703
session = cache.get_session_data(session_id)


# In[10]:


session.metadata


# In[11]:


session.structurewise_unit_counts


# In[12]:


scene_presentations = session.get_stimulus_table("natural_movie_three")
visp_units = session.units[session.units["ecephys_structure_acronym"] == "VISp"]

spikes = session.presentationwise_spike_times(
    stimulus_presentation_ids=scene_presentations.index.values,
    unit_ids=visp_units.index.values[:]
)

spikes


# In[20]:


cache.get_unit_analysis_metrics()


# In[21]:


units = cache.get_units()


# In[22]:


len(units)


# In[31]:


units[analysis_metrics1]["ecephys_structure_acronym"]


# In[30]:


units.columns


# In[32]:


analysis_metrics1["fano_ns"]


# In[33]:


df3 = pd.merge(units, analysis_metrics1["fano_ns"],left_index = True, right_index = True)


# In[35]:


df3.columns


# In[39]:


df4 = df3[["ecephys_structure_acronym","fano_ns"]]


# In[40]:


df4


# In[43]:


design = pd.pivot_table(
    df4,
    values="fano_ns", 
    index="ecephys_structure_acronym", 
    fill_value=0.0,
    aggfunc=np.average
)


# In[44]:


design


# In[50]:


scene_presentations = session.get_stimulus_table("natural_movie_three")
visp_units = session.units[session.units["ecephys_structure_acronym"] == "VISp"]

spikes = session.presentationwise_spike_times(
    stimulus_presentation_ids=scene_presentations.index.values,
    unit_ids=visp_units.index.values[:]
)

spikes


# In[46]:


spikes["unit_id"]


# In[147]:



    scene_presentations = session.get_stimulus_table("natural_movie_three")
    visp_units = session.units[session.units["ecephys_structure_acronym"] == "CA3"]

    spikes = session.presentationwise_spike_times(
    stimulus_presentation_ids=scene_presentations.index.values,
    unit_ids=visp_units.index.values[:]
)

    spikes
    


# In[148]:



spikes["stimulus_presentation_id"] = pd.to_numeric(spikes["stimulus_presentation_id"],errors='coerce').fillna(0.0)
spikes["unit_id"] = pd.to_numeric(spikes["unit_id"],errors='coerce').fillna(0.0)


# In[ ]:





# In[149]:


spikes = spikes[spikes["stimulus_presentation_id"] < 10000]


# In[150]:


last_time = {}
sum = 0
num = 0
for index, row in spikes.iterrows():
    #print(index, int(row["unit_id"]))
    if int(row["unit_id"])  not in last_time:
        #print(int(row["unit_id"]))
        last_time[int(row["unit_id"])] = index
        #print(index)
    else :
        sum = sum + index - last_time[int(row["unit_id"])]
        num += 1
        last_time[int(row["unit_id"])] = index
print(sum/num)
        


# In[ ]:




