import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import numpy as np
import matplotlib.pyplot as plt
from transformations.warp_function import warp
from sktime.distances.elastic import dtw_distance
import  matplotlib


root = tk.Tk()

canvas1 = tk.Canvas(root, width=800, height=300)
canvas1.pack()

label1 = tk.Label(root, text='Graphical User Interface')
label1.config(font=('Arial', 20))
canvas1.create_window(400, 50, window=label1)

entry1 = tk.Entry(root)
canvas1.create_window(400, 100, window=entry1)

entry2 = tk.Entry(root)
canvas1.create_window(400, 120, window=entry2)

entry3 = tk.Entry(root)
canvas1.create_window(400, 140, window=entry3)




inter = np.loadtxt("inter_warp_3000.txt")
neig_y = np.loadtxt("neig_y_seg_warp_3000.txt")

inter.shape



train_x, train_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TRAIN.ts")
test_x, test_y = load_from_tsfile_to_dataframe("../datasets/Univariate_ts/CBF/CBF_TEST.ts")


ind = 5 #Class 3

ref = test_x.values[ind,:][0].values
print(test_y[ind])
figure1 = Figure(figsize=(4,3), dpi=100)
subplot1 = figure1.add_subplot(111)
subplot1.plot(ref)
canvas = FigureCanvasTkAgg(figure1, master=root)
canvas.get_tk_widget().pack()
canvas.draw()

# figure1 = Figure(figsize=(4,3), dpi=100)
# subplot1 = figure1.add_subplot(111)


ind_class = []
for i in np.unique(train_y):
    ind_class.append(np.where(train_y==i)[0])


per_class =3
matplotlib.use('Agg')  # turn off gui
fig, axs = plt.subplots(len(np.unique(train_y)),1, figsize=(5, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)

axs = axs.ravel()

for c in range(len(np.unique(train_y))):
    for j in range(per_class):
        ind = ind_class[c][j]
        axs[c].plot(train_x.values[ind,:][0].values)

# subplot1.plot(ref)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()
canvas.draw()




def create_charts():
    global x1
    global x2
    global x3
    global bar1
    global pie2
    start = int(entry1.get())
    end = int(entry2.get())
    warp_level = float(entry3.get())


    neig = warp(ref, start, end, warp_level)
    figure1 = Figure(figsize=(4, 3), dpi=100)
    subplot1 = figure1.add_subplot(111)

    subplot1.plot(neig)
    if len(neig)<len(ref):
        subplot1.set_xlim(0,len(ref))

    bar1 = FigureCanvasTkAgg(figure1, root)
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=0)



    distance_matrix = np.zeros(train_x.shape[0])


    for j in range(0, train_x.shape[0]):
        distance_matrix[j] = dtw_distance(neig, np.asarray(train_x.values[j, :][0]))

    #print(train_y[np.argmin(distance_matrix)])
    #print(np.argmin(distance_matrix))
    #plt.plot(np.asarray(train_x.values[np.argmin(distance_matrix), :][0]))
    #print(train_y[2])

    # plt.plot(ref, c='grey')
    # plt.plot(vec1)
    # plt.plot(vec2)
    # plt.xlim(0, 130)



    figure2 = Figure(figsize=(4, 3), dpi=100)
    subplot2 = figure2.add_subplot(111)
    # labels2 = 'Label1', 'Label2', 'Label3'
    # pieSizes = [float(x1), float(x2), float(x3)]
    # my_colors2 = ['lightblue', 'lightsteelblue', 'silver']
    # explode2 = (0, 0.1, 0)
    # subplot2.pie(pieSizes, colors=my_colors2, explode=explode2, labels=labels2, autopct='%1.1f%%', shadow=True,
    #              startangle=90)

    subplot2.plot(np.asarray(train_x.values[np.argmin(distance_matrix), :][0]))

    #subplot2.axis('equal')
    pie2 = FigureCanvasTkAgg(figure2, root)
    pie2.get_tk_widget().pack()


def clear_charts():
    bar1.get_tk_widget().pack_forget()
    pie2.get_tk_widget().pack_forget()


button1 = tk.Button(root, text=' Create Charts ', command=create_charts, bg='palegreen2', font=('Arial', 11, 'bold'))
canvas1.create_window(400, 180, window=button1)

button2 = tk.Button(root, text='  Clear Charts  ', command=clear_charts, bg='lightskyblue2', font=('Arial', 11, 'bold'))
canvas1.create_window(400, 220, window=button2)

button3 = tk.Button(root, text='Exit Application', command=root.destroy, bg='lightsteelblue2',
                    font=('Arial', 11, 'bold'))
canvas1.create_window(400, 260, window=button3)

root.mainloop()