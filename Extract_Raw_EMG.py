import ezc3d
import pyomeca
import scipy.io as sio
# from scipy import signal
# from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pickle
import glob

# for f, file in enumerate(files):

Participants = ['Pilote' + str(iP) for iP in range(1, 25)] # Define Participants using comprehension list
Muscles = ['DeltA', 'DeltM', 'DeltP', 'Bi', 'Tri', 'TrapSup', 'TrapMed', 'TrapInf', 'Dent']
Muscles2 = ['DeltA.IM EMG1', 'DeltM.IM EMG2', 'DeltP.IM EMG3', 'Bi.IM EMG4', 'Tri.IM EMG5', 'TrapSup.IM EMG6', 'TrapMed.IM EMG7', 'TrapInf.IM EMG8', 'Dent.IM EMG9']
Freq = 2000

# for iP in range(len(Participants)):
iP = 2
StrucData = {}
StrucData2 = {}

# Read all files in the folder
path = r'J:/IRSST_DavidsTea/Raw_Data/' + Participants[iP] + '/EMG'
for iFiles, files in enumerate(glob.glob(path+"/**c3d")):
    # data2 = pyomeca.Analogs.from_c3d(files, usecols=Muscles)
    data2 = pyomeca.Analogs.from_c3d(files, usecols=Muscles2)

# for iFiles in range(len(files)):
    TrialName = ['Trial' + str(iFiles)]
    TrialName2 = 'Trial' + str(iFiles)

    # Import .c3d raw file
    c3d = ezc3d.c3d(files)

    # Print the parameters
    NumParam = c3d['parameters']['ANALOG']['USED']['value'][0]  # Number of parameters recorded
    NameParam = c3d['parameters']['ANALOG']['LABELS']['value']  # Name of parameters recorded

    # Store what we want into data variable
    data = c3d['data']['analogs']

    # Get the index of wanted parameters -> Muscles
    id_Muscles = [iNameParam for iNameParam in range(NumParam) for iM in range(len(Muscles))
                  if NameParam[iNameParam].find(Muscles[iM]) == 0 and NameParam[iNameParam].find('IM') != -1]

    Name_id_Muscles = [Muscles[iM] for iNameParam in range(NumParam) for iM in range(len(Muscles))
                       if NameParam[iNameParam].find(Muscles[iM]) == 0 and NameParam[iNameParam].find('IM') != -1]

    data_to_keep = np.array(data[0, id_Muscles, :])

    for n in TrialName:
        StrucData[n] = {x: list(y) for x, y in zip(Name_id_Muscles, zip(*data_to_keep.transpose()))}

    StrucData2[TrialName2] = {x.split(".")[0]: list(y) for x, y in zip(Muscles2, zip(*data2.data.transpose()))}


    # Plot the wanted data
    if True:
        plt.figure(figsize=(9, 5))
        for iM in range(len(Name_id_Muscles)):
            plt.subplot(5, 2, iM + 1)
            plt.plot(StrucData['Trial0'][Name_id_Muscles[iM]])
            plt.title(Name_id_Muscles[iM])
            plt.suptitle('Raw EMG signals', fontsize=16)

        plt.figure(figsize=(9, 5))
        for iM in range(len(Muscles2)):
            plt.subplot(5, 2, iM + 1)
            plt.plot(StrucData2['Trial0'][Muscles[iM]])
            plt.title(Muscles2[iM])
            plt.suptitle('Raw EMG signals', fontsize=16)

        plt.show()
        input("Press Enter to continue...")

# Save data as .pickle
add_data_to_pickle(StrucData, "test_pick")

import pickle
with open('test_pick', 'rb') as f:
    x = pickle.load(f)

# Save as .mat file
sio.savemat("J:/IRSST_DavidsTea/Data_exported/EMG/Raw/" + Participants[iP] + "_EMG_raw.mat", {'Data': StrucData})

start = time.time()
sio.savemat("C:/Users/p1098713/Desktop/" + Participants[iP] + "_EMG_raw.mat", {'Data': StrucData})
stop = time.time()
print('time to run', stop-start)

# Store the wanted data into pickle files
# f = open("J:/IRSST_DavidsTea/Data_exported/EMG/Raw/" + Participants[iP] + "_EMG_raw.p", "wb")
start = time.time()
f = open("C:/Users/p1098713/Desktop/" + Participants[iP] + "_EMG_raw.p", "wb")
pickle.dump(StrucData, f)
f.close()
stop = time.time()
print('temps de sauvegarde', stop-start)

# Store the wanted data into Json files
import json
start = time.time()
json.dump(StrucData, open("C:/Users/p1098713/Desktop/" + Participants[iP] + "_EMG_raw.json", 'w' ) )
stop = time.time()
print('temps de sauvegarde', stop-start)

# Test Load Data
import pickle
import time
Participants = ['Pilote' + str(iP) for iP in range(1, 25)] # Define Participants using comprehension list
iP = 2
start = time.time()
f = open("C:/Users/p1098713/Desktop/" + Participants[iP] + "_EMG_raw.p", "rb")
zzz = pickle.load(f)
f.close()
stop = time.time()
print('temps de sauvegarde', stop-start)
import matplotlib.pyplot as plt

import json
start = time.time()
data = json.load(open("C:/Users/p1098713/Desktop/" + Participants[iP] + "_EMG_raw.json" ) )
stop = time.time()
print('temps de sauvegarde', stop-start)

import scipy.io as sio
import time
import matplotlib.pyplot as plt
Participants = ['Pilote' + str(iP) for iP in range(1, 25)] # Define Participants using comprehension list
iP = 2
start = time.time()
# StrucData = sio.loadmat("C:/Users/p1098713/Desktop/" + Participants[iP] + "_EMG_raw.mat")
Data = sio.loadmat("C:/Users/p1098713/Desktop/" + Participants[iP] + "_EMG_Raw.mat")
xxx = Data['Data'][0][0][0][0][0][1]
# StrucData =

stop = time.time()
print('temps de sauvegarde', stop-start)

start = time.time()
c3d = ezc3d.c3d('C:/Users/p1098713/Desktop/Contractions_Alexis01.c3d')
stop = time.time()
print('time to run', stop - start)
