from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import os

main = tkinter.Tk()
main.title("Human Action Recognition from depth maps and Postures using Deep Learning")
main.geometry("1300x1200")

global filename
global classifier
global X, labels, datas, graph
class_labels = ['high arm wave', 'horizontal arm wave', 'hammer', 'hand catch', 'forward punch', 'high throw', 'draw x', 'draw tick', 'draw circle', 'hand clap',
          'two hand wave', 'side-boxing', 'bend', 'forward kick', 'side kick', 'jogging', 'tennis swing', 'tennis serve', 'golf swing', 'pick up & throw']


def upload():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir = ".")
    pathlabel.config(text=filename)
    text.insert(END,filename+" loaded\n")

def animate(i):
    graph._offsets3d = (datas[i,:,0], datas[i,:,2], datas[i,:,1])
    return graph

def loadData(action, subject, instance):
    ins = np.loadtxt(filename+'/a%02i_s%02i_e%02i_skeleton.txt'%(action, subject, instance))
    print("ins.shape[0]",ins.shape[0])
    ins = ins.reshape((ins.shape[0]//20, 20, 4))
    return ins
    
def featuresExtraction():
    global X, labels, datas, graph
    text.delete('1.0', END)
    X = []
    data = np.load("model/data.txt.npy",allow_pickle=True)
    labels = np.load("model/labels.txt.npy")
    data = np.asarray(data)
    labels = np.asarray(labels)
    for i in range(len(data)):
        values = data[i][1]
        X.append(np.asarray(values))
    X = np.asarray(X)
    labels = np.asarray(labels)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    labels = labels[indices]
    labels = to_categorical(labels)
    X = X.reshape(X.shape[0],5,4,3)
    text.insert(END,"Total subject & action images found in dataset: "+str(data.shape[0])+"\n")
    text.insert(END,"Total actions found in dataset:\n\n")
    text.insert(END,str(class_labels))

    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_xlim3d([0.0, 300.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([0.0, 1400.0])
    ax.set_ylabel('Z')
    ax.set_zlim3d([300.0, 0.0])
    ax.set_zlabel('Y')
    graph = ax.scatter([], [], [])
    datas = loadData(1,1,1)
    datas = np.asarray(datas)
    ani = FuncAnimation(fig, animate, frames=datas.shape[0], interval=100)
    plt.show()


def trainCNN():
    global X, labels,classifier
    text.delete('1.0', END)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()    
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()      
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 1, 1, input_shape = (X.shape[1], X.shape[2], X.shape[3]), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (1, 1)))
        classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (1, 1)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 256, activation = 'relu'))
        classifier.add(Dense(output_dim = labels.shape[1], activation = 'softmax'))
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(X, labels, batch_size=16, epochs=1000, shuffle=True, verbose=2)
        classifier.save_weights('model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()     
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
    predict = classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100    
    text.insert(END,'CNN Activity Recognition Accuracy  : '+str(a)+"\n")
    text.insert(END,'CNN Activity Recognition Precision : '+str(p)+"\n")
    text.insert(END,'CNN Activity Recognition Recall    : '+str(r)+"\n")
    text.insert(END,'CNN Activity Recognition FMeasure  : '+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = class_labels, yticklabels = class_labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,10])
    plt.title("CNN Depth Maps Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def read(file):
    data, labels, lens, subjects = [], [], [], []
    action = np.loadtxt(file)[:, :3].flatten()
    frame_size = len(action) // 60  # 20 iskeleton num x,y,z 3D points
    action = np.asarray(action).reshape(frame_size, 60)
    new_act = []
    for frame in action:
        new_act.append(frame)
    data.append(new_act)
    data = np.asarray(data)
    return data

def loadTestData(filename):
    ins = np.loadtxt(filename)
    print("ins.shape[0]",ins.shape[0])
    ins = ins.reshape((ins.shape[0]//20, 20, 4))
    return ins

def predictAction():
    global datas, graph
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="TestFiles")
    test = []
    data = read(filename)
    for i in range(len(data)):
        values = data[i][1]
        test.append(values)
    test = np.asarray(test)
    test = test.reshape(test.shape[0],5,4,3)
    preds = classifier.predict(test)
    predict = np.argmax(preds)
    text.insert(END,"Skeleton values : "+str(test)+" ACTIVITY RECOGNIZE AS: "+class_labels[predict])
    text.update_idletasks()
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_xlim3d([0.0, 300.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([0.0, 1400.0])
    ax.set_ylabel('Z')
    ax.set_zlim3d([300.0, 0.0])
    ax.set_zlabel('Y')
    graph = ax.scatter([], [], [])
    datas = loadTestData(filename)
    datas = np.asarray(datas)
    ani = FuncAnimation(fig, animate, frames=datas.shape[0], interval=100)
    plt.show()
    

font = ('times', 16, 'bold')
title = Label(main, text='Human Action Recognition from depth maps and Postures using Deep Learning')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload MSRAction3D Image", command=upload)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=350,y=100)

feButton = Button(main, text="Features Extraction", command=featuresExtraction)
feButton.place(x=50,y=150)
feButton.config(font=font1) 

cnnButton = Button(main, text="Train CNN Algorithm", command=trainCNN)
cnnButton.place(x=280,y=150)
cnnButton.config(font=font1) 

predictButton = Button(main, text="Predict Action from Test Data", command=predictAction)
predictButton.place(x=500,y=150)
predictButton.config(font=font1) 

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
