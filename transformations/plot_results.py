from sktime.utils.data_io import load_from_tsfile_to_dataframe
import random
import numpy as np
import matplotlib.pyplot as plt

EvalGpWaDtw = np.array([0.58, 0.69, 0.54, 0.83, 0.76, 0.41])


RobusGpWaDtw = np.array([0.88, 0.92, 0.94,0.93, 0.93, 0.87])
RobusGpWaBoss = np.array([0.57, 0.86, 0.96,0.97, 0.91, 0.82])
RobusGpWaST = np.array([0.75, 0.93, 0.97,0.98, 0.98, 0.98])


RobusGpScDtw = np.array([0.28, 0.36, 0.53,0.54, 0.37, 0.30])
RobusGpScBoss = np.array([0.73, 0.88, 0.96,0.98, 0.90, 0.84])

RobusGpNoDtw = np.array([0.84, 0.67, 0.55,0.40, 0.33])
RobusGpNoBoss = np.array([0.99, 0.89, 0.67,0.56, 0.54])

RobusCoWaDtw = np.array([1,1, 1,1,1, 1])
RobusCoWaBoss = np.array([0.67, 0.71, 0.96,0.96, 0.96, 0.92])


plt.figure()
plt.scatter(RobusGpWaDtw,RobusGpWaBoss, label="GunPoint")
plt.scatter(RobusCoWaDtw,RobusCoWaBoss, label="Coffee")
plt.title("Warp")
plt.xlabel("Dtw")
plt.ylabel("Boss")
plt.legend()


plt.figure()
plt.plot(RobusGpWaDtw,label="Warp Dtw")
plt.plot(RobusGpWaBoss,label="Warp Boss")
plt.plot(RobusGpWaST,label="Warp ST")
plt.plot(RobusGpScDtw,label="Scale Dtw")
plt.plot(RobusGpScBoss,label="Scale Boss")
plt.legend()

plt.figure()
plt.plot(RobusGpNoDtw,label="Noise Dtw")
plt.plot(RobusGpNoBoss,label="Noise Boss")
plt.legend()
