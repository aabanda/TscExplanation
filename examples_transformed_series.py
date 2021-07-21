from sktime.utils.data_io import load_from_tsfile_to_dataframe
import matplotlib.pyplot as plt
from functions.transformations import warp, scale, noise, slice


input_dir = "../Univariate2018"
db_name = "ArrowHead"

train_x, train_y = load_from_tsfile_to_dataframe("%s/%s/%s_TRAIN.ts" % (input_dir, db_name, db_name))
test_x, test_y = load_from_tsfile_to_dataframe("%s/%s/%s_TRAIN.ts" % (input_dir, db_name, db_name))



ind=0
ref = test_x.values[ind,:][0].values


#Warp
plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(warp(ref,100,200,1.2),linewidth=3, label="Warped TS")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.ylim(-2,2)
plt.xlim(-10,280)
plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(warp(ref,30,100,0.7),linewidth=3, label="Warped TS")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.ylim(-2,2)
plt.xlim(-10,280)
plt.tight_layout()




#Scale
plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(scale(ref,180,230,0.7),linewidth=3, label="Scaled TS")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.ylim(-2,2)
plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(scale(ref,30,80,1.2),linewidth=3, label="Scaled TS")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.ylim(-2,2)
plt.tight_layout()


#Noise
plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(noise(ref,80,170,2),linewidth=3, label="TS with Noise")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.ylim(-2,2)
plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(noise(ref,160,200,7),linewidth=3, label="TS with Noise")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.ylim(-2,2)
plt.legend(fontsize=20)
plt.tight_layout()



#Slice
plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(slice(ref,5,len(ref)),linewidth=3, label="Shifted TS")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.ylim(-2,2)
plt.legend(fontsize=20)
plt.tight_layout()

plt.figure(figsize=(7,5))
plt.plot(ref,linewidth=3, label="Reference TS")
plt.plot(slice(ref,20,230),linewidth=3, label="Shifted TS")
plt.xlabel("t", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.ylim(-2,2)
plt.legend(fontsize=20)
plt.tight_layout()
