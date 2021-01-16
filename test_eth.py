import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential  
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

test_data_dir = 'custom\\AWEeth\\test'
img_width, img_height = 96, 96
class_names = ['1','2','3','4','5','6','99']

def evaluate(model):
    predictions = []
    ranks = []
    for i in range(0, 7):
        ranks.append([])
    gt = []

    for i in range(0,len(class_names)):
        files = os.listdir(test_data_dir + '\\' + class_names[i])
        
        for f in files:
            gt.append(class_names[i])
            img = image.load_img(test_data_dir + '\\' + class_names[i] + '\\' + f, target_size=(96, 96))

            
            img_batch = np.expand_dims(img, axis=0)
            prediction = model.predict(img_batch, batch_size=1)

            prediction = tf.nn.sigmoid(prediction)
            prediction = np.array(prediction)
            indices = (-prediction).argsort()[0]
            idx = np.argmax(prediction)
            
            for j in range(1,8):
                
                if i in indices[0:j]:
                    ranks[j-1].append(class_names[i])
                else:
                    
                    ranks[j-1].append(class_names[idx])
            prediction = class_names[idx]
            predictions.append(prediction)
    predictions = np.array(predictions)
    gt = np.array(gt)
    ranks_acc = np.zeros((1,7))[0]
    for i in range(0,7):
        ranks_acc[i] = (np.sum(ranks[i]==gt)/len(predictions))

    return ranks_acc




model1 = tf.keras.models.load_model("resnet-eth")


ranks_acc1 = evaluate(model1)


print("RANK1")
print(ranks_acc1[0])
print("RANK5")
print(ranks_acc1[4])
print("AUC")
print(np.sum(ranks_acc1)/7)

plot1, = plt.plot(list(range(1,8)),ranks_acc1, label='Resnet-152-v2')


plt.ylabel("Recognition rate")
plt.xlabel("Rank")
plt.ylim([0.0, 1])
plt.legend(handles=[plot1])
plt.savefig("CMC-ethnicity")
plt.show()
