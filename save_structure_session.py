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
manifest_path = os.path.join('F:/ecephys_cache_dir/', "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path,fetch_tries=1,timeout=2000*60)

print(cache.get_all_session_types())


# In[3]:


sessions = cache.get_session_table()
brain_observatory_type_sessions = sessions[sessions["session_type"] == "brain_observatory_1.1"]
brain_observatory_type_sessions.head()


# In[4]:


session_id = 715093703
session = cache.get_session_data(session_id)


# In[5]:


session.metadata


# In[6]:


session.structurewise_unit_counts


# In[82]:


scene_presentations = session.get_stimulus_table("natural_movie_three")
visp_units = session.units[session.units["ecephys_structure_acronym"] == "VISp"]

spikes = session.presentationwise_spike_times(
    stimulus_presentation_ids=scene_presentations.index.values,
    unit_ids=visp_units.index.values[:]
)

spikes


# In[83]:


spikes["count"] = np.zeros(spikes.shape[0])
spikes = spikes.groupby(["stimulus_presentation_id", "unit_id"]).count()
design = pd.pivot_table(
    spikes, 
    values="count", 
    index="stimulus_presentation_id", 
    columns="unit_id", 
    fill_value=0.0,
    aggfunc=np.sum
)
design


# In[84]:


targets = scene_presentations.loc[design.index.values, "frame"]
targets


# In[9]:


def save_structure(structure_name, session):

    scene_presentations = session.get_stimulus_table("natural_movie_one")
    visp_units = session.units[session.units["ecephys_structure_acronym"] == structure_name]

    spikes = session.presentationwise_spike_times(
        stimulus_presentation_ids=scene_presentations.index.values,
        unit_ids=visp_units.index.values[:]
    )
    print(spikes.shape)
    if spikes.shape[0] == 0:
        return np.zeros((900, 0 )), np.zeros(0, dtype = np.int)
    spikes["count"] = np.zeros(spikes.shape[0])
    spikes = spikes.groupby(["stimulus_presentation_id", "unit_id"]).count()
    
    design = pd.pivot_table(
        spikes, 
        values="count", 
        index="stimulus_presentation_id", 
        columns="unit_id", 
        fill_value=0.0,
        aggfunc=np.sum
    )
    print(design.shape)
    targets = scene_presentations.loc[design.index.values, "frame"]
    
    sti_num = design.shape[0]
    units_num = design.shape[1]
    design_value = design.to_numpy()
    pic_unit_sum = np.zeros((900, units_num))
    pic_unit_num = np.zeros((900, units_num))
    pic_unit_ave = np.zeros((900, units_num))
    unit_id = np.zeros(units_num, dtype=np.int)
    for i in range(sti_num):
        for j in range(units_num):
            pic_id = int(targets[design.index.values[i]])
            if pic_id != -1:
                pic_unit_sum[pic_id][j]+=design_value[i][j]
                pic_unit_num[pic_id][j]+=1
    for i in range(900):
        for j in range(design.columns.values.size):
            pic_unit_ave[i][j] = pic_unit_sum[i][j] / pic_unit_num[i][j]
    for i in range(units_num):
        unit_id[i] = int(design.columns.values[i])
    print(pic_unit_ave.shape)
    print(unit_id)
    return pic_unit_ave, unit_id


# In[10]:


def save_session(session_id):
    path = "F:\\ecephys_cache_dir\\allen_spike_movie_1\\"
    session = cache.get_session_data(session_id)
    structures = session.structurewise_unit_counts.index.values[:]
    for structure in structures:
        print(structure)
        pic_unit_ave, unit_id = save_structure(structure, session)
        np.save( path + "nm_spike_"+ str(session_id) + "_" + structure + ".npy", pic_unit_ave)
        np.save( path + "nm_unitid_"+ str(session_id) + "_" + structure + ".npy", unit_id)


# In[11]:



sessions_id = brain_observatory_type_sessions.index.values
for session_id in sessions_id:
    print(session_id)
    save_session(session_id)


# In[ ]:




