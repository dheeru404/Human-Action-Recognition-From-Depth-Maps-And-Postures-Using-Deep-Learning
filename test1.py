from os import listdir
from os.path import join
import numpy as np
import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn.preprocessing import normalize

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import os
from keras.layers import Convolution2D
'''
def full_fname2_str(data_dir, fname, sep_char):
    fnametostr = ''.join(fname).replace(data_dir, '')
    ind = int(fnametostr.index(sep_char))
    label = int(fnametostr[ind + 1:ind + 3])
    return label

def frame_normalizer(frame, frame_size):
    assert frame.shape[0] == frame_size
    frame = frame.reshape(frame_size // 3, 3)
    spine_mid = frame[1]
    new_frame = []
    j = 0
    for joint in frame:
        new_frame.append(joint - spine_mid)
        j += 1
    new_frame = np.asarray(new_frame)
    return (list(new_frame.flatten()))


def read(data_dir):
    print('Loading MSR 3D Data, data directory %s' % data_dir)
    data, labels, lens, subjects = [], [], [], []
    filenames = []
    documents = [join(data_dir, d)
                 for d in sorted(listdir(data_dir))]
    filenames.extend(documents)
    filenames = np.array(filenames)

    for file in filenames:
        #print("file",file)
        action = np.loadtxt(file)[:, :3].flatten()
        #print("action",action)

        labels.append(full_fname2_str(data_dir, file, 'a'))
        frame_size = len(action) // 60  # 20 iskeleton num x,y,z 3D points
        lens.append(frame_size)
        action = np.asarray(action).reshape(frame_size, 60)
        #print("action shape",action.shape)
        
        new_act = []
        for frame in action:
            new_act.append(frame)

        data.append(new_act)
        subjects.append(full_fname2_str(data_dir, file, 's'))
        
    data = np.asarray(data)
    labels = np.asarray(labels) -1
    lens = np.asarray(lens)
    
    subjects = np.asarray(subjects)
    print("All files read!")
    print('initial shapes [data label len]: %s %s %s' % (data.shape, labels.shape, lens.shape))
    return data,labels,lens



MSR_data_dir = 'MSRAction3DSkeleton(20joints)'

data,labels,lens = read(MSR_data_dir)
np.save("model/data.txt",data)
np.save("model/labels.txt",labels)
'''
X = []
data = np.load("model/data.txt.npy",allow_pickle=True)
labels = np.load("model/labels.txt.npy")
data = np.asarray(data)
labels = np.asarray(labels)
for i in range(len(data)):
    values = data[i][1]
    X.append(np.asarray(values))
X = np.asarray(X)
#X = normalize(X)
labels = np.asarray(labels)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
labels = labels[indices]
labels = to_categorical(labels)
X = X.reshape(X.shape[0],5,4,3)
print(X.shape)

if os.path.exists('model/model.json'):
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
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
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(X, labels, batch_size=16, epochs=1000, shuffle=True, verbose=2)
    classifier.save_weights('model/model_weights.h5')            
    model_json = classifier.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    print("Training Model Accuracy = "+str(accuracy))

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)
predict = classifier.predict(X_test)
predict = np.argmax(predict, axis=1)
testY = np.argmax(y_test, axis=1)
p = precision_score(testY, predict,average='macro') * 100
r = recall_score(testY, predict,average='macro') * 100
f = f1_score(testY, predict,average='macro') * 100
a = accuracy_score(testY,predict)*100    
print('CNN Activity Recognition Accuracy  : '+str(a)+"\n")

def read(file):
    data, labels, lens, subjects = [], [], [], []
    action = np.loadtxt('MSRAction3DSkeleton(20joints)/'+file)[:, :3].flatten()
    frame_size = len(action) // 60  # 20 iskeleton num x,y,z 3D points
    action = np.asarray(action).reshape(frame_size, 60)
    new_act = []
    for frame in action:
        new_act.append(frame)
    data.append(new_act)
    data = np.asarray(data)
    return data

class_labels = ['high arm wave', 'horizontal arm wave', 'hammer', 'hand catch', 'forward punch', 'high throw', 'draw x', 'draw tick', 'draw circle', 'hand clap',
          'two hand wave', 'side-boxing', 'bend', 'forward kick', 'side kick', 'jogging', 'tennis swing', 'tennis serve', 'golf swing', 'pick up & throw']


arr = ['a01_s01_e02_skeleton.txt','a01_s01_e02_skeleton.txt','a02_s08_e02_skeleton.txt','a08_s02_e02_skeleton.txt','a09_s04_e03_skeleton.txt','a11_s05_e02_skeleton.txt']
for j in range(len(arr)):
    data = read(arr[j])
    test = []
    for i in range(len(data)):
        values = data[i][1]
        test.append(values)
    test = np.asarray(test)
    test = test.reshape(test.shape[0],5,4,3)
    preds = classifier.predict(test)
    predict = np.argmax(preds)
    print(str(predict)+" "+class_labels[predict]+" "+arr[j])




