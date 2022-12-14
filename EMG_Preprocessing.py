import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pyomeca import Analogs
import pickle
import time

# Define variables
Participants = ['Pilote' + str(iP) for iP in range(1, 25)] # Define Participants using comprehension list
Muscles = ['DeltA', 'DeltM', 'DeltP', 'Bi', 'Tri', 'TrapSup', 'TrapMed', 'TrapInf', 'Dent']
EMG_Freq = 2000 # Acquisition frequency 2000 Hz

# for iP in range(len(Participants)):
iP = 2

# ---Import File---
start = time.time()
with open('test_pick', 'rb') as f:
    StrucData = pickle.load(f)
end = time.time()
print('Time to load pickle', end-start)
Keys = StrucData.keys()

for key in Keys:
    # Data_mat = np.array([StrucData[key][Muscles[iM]] for iM in range(len(Muscles))])
    # EMG = Analogs(Data_mat[:, :])
    EMG = Analogs(np.array([StrucData[key][Muscle] for Muscle in Muscles]))
    EMG['channel'] = Muscles

    # ---Raw signals---
    fft_EMG = EMG.meca.fft(freq=EMG_Freq)

    EMG.plot(x="time", col="channel", col_wrap=5)
    plt.suptitle('Raw EMG signals' + key, fontsize=16)

    plt.figure(2)
    plt.suptitle('FFT Raw EMG signals' + key, fontsize=16)
    for iM in range(len(Muscles)):
        plt.subplot(2, 5, iM + 1)
        fft_EMG[iM].plot.line(x="freq")
        plt.title(Muscles[iM])

    # ---Band-pass filter signals---
    low_cut = 10
    high_cut = 400
    EMGBP = EMG.meca.band_pass(order=2, cutoff=[low_cut, high_cut], freq=EMG_Freq)

    # fft_EMGBP = EMGBP.meca.fft(EMG_Freq)
    #
    # EMGBP.plot(x="time", col="channel", col_wrap=5)
    # plt.suptitle('band-pass filtered EMG signals', fontsize=16)
    #
    # plt.figure(4)
    # plt.suptitle('FFT band-pass filtered EMG signals', fontsize=16)
    # for iM in range(len(Muscles)):
    #     plt.subplot(2, 5, iM + 1)
    #     fft_EMGBP[iM].plot.line(x="freq")
    #     plt.title(Muscles[iM])

    # ---stop-band filter signals---
    # low_cut = 59
    # high_cut = 61
    # EMGBS = EMGBP.meca.band_stop(order=2, cutoff=[low_cut, high_cut], freq=EMG_Freq)

    # fft_EMGBS = EMGBS.meca.fft(EMG_Freq)
    #
    # EMGBS.plot(x="time", col="channel", col_wrap=5)
    # plt.suptitle('stop-band filtered EMG signals', fontsize=16)
    #
    # plt.figure(6)
    # plt.suptitle('FFT stop-band filtered EMG signals', fontsize=16)
    # for iM in range(len(Muscles)):
    #     plt.subplot(2, 5, iM + 1)
    #     fft_EMGBS[iM].plot.line(x="freq")
    #     plt.title(Muscles[iM])

    # ---Remove mean---
    EMGBL = EMGBP
    EMGBL.meca.center()

    fft_EMGBL = EMGBL.meca.fft(EMG_Freq)

    EMGBL.plot(x="time", col="channel", col_wrap=5)
    plt.suptitle('Detrended & filtered EMG signals' + key, fontsize=16)

    plt.figure(8)
    plt.suptitle('FFT detrended & filtered EMG signals' + key, fontsize=16)
    for iM in range(len(Muscles)):
        plt.subplot(2, 5, iM + 1)
        fft_EMGBL[iM].plot.line(x="freq")
        plt.title(Muscles[iM])

    plt.show()
    input("Press Enter to continue...")

# # Store the filtered data into a dictionnary and save as .mat file
# data_to_keep = EMGBL.data
# EMG_filtered = {x:list(y) for x,y in zip(Muscles, zip(*data_to_keep.transpose()))}
# sio.savemat("J:/IRSST_DavidsTea/Data_exported/EMG/Filtered/" + Participants[iP] + "_EMG_filtered.mat", EMG_filtered)
