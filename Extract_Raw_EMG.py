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
from biosiglive.io.save_data import add_data_to_pickle

# for f, file in enumerate(files):

Participants = ['Pilote' + str(iP) for iP in range(1, 25)] # Define Participants using comprehension list
Raw_Muscles_Names = ['DeltA.IM EMG1', 'DeltM.IM EMG2', 'DeltP.IM EMG3', 'Bi.IM EMG4', 'Tri.IM EMG5', 'TrapSup.IM EMG6', 'TrapMed.IM EMG7', 'TrapInf.IM EMG8', 'Dent.IM EMG9']
Muscles = ['DeltA', 'DeltM', 'DeltP', 'Bi', 'Tri', 'TrapSup', 'TrapMed', 'TrapInf', 'Dent']
Freq = 2000

# for iP in range(len(Participants)):
iP = 2
StrucData = {}

# Read all files in the folder
path = r'J:/IRSST_DavidsTea/Raw_Data/' + Participants[iP] + '/EMG'
for iFiles, files in enumerate(glob.glob(path+"/**c3d")):
    start = time.time()
    data = pyomeca.Analogs.from_c3d(files, usecols=Raw_Muscles_Names)
    end = time.time()
    print('Time to load c3d', end-start)
    TrialName = 'Trial' + str(iFiles)
    StrucData[TrialName] = {x.split(".")[0]: np.array(y) for x, y in zip(Raw_Muscles_Names, zip(*data.data.transpose()))}

    # Plot the wanted data
    if False:
        plt.figure(figsize=(9, 5))
        for iM, Muscle in enumerate(Muscles):
            plt.subplot(5, 2, iM + 1)
            plt.plot(StrucData[TrialName][Muscle])
            plt.title(Muscle)
            plt.suptitle('Raw EMG signals', fontsize=16)

        plt.show()
        input("Press Enter to continue...")

# Save data as .pickle
start = time.time()
add_data_to_pickle(StrucData, "test_pick")
end = time.time()
print('Time to save pickle', end-start)

# start = time.time()
# with open('test_pick', 'rb') as f:
#     x = pickle.load(f)
# end = time.time()
# print('Time to save data', end - start)

