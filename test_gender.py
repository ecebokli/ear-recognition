import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential  
from tensorflow.keras.preprocessing import image
import os

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


model = tf.keras.models.load_model("resnet-gender")
test_data_dir = 'custom\\AWEgender\\test'

img_width, img_height = 96, 96
class_names = ['f','m']
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_dir,
    image_size=(img_width, img_height),
    batch_size=8)




file_count = 0
predictions = []
gt = []
TP = 0
FP = 0
TN = 0
FN = 0
male = []
for i in range(0,len(class_names)):
    files = os.listdir(test_data_dir + '\\' + class_names[i])
    for f in files:
        gt.append(class_names[i])
        img = image.load_img(test_data_dir + '\\' + class_names[i] + '\\' + f, target_size=(96, 96))
        #img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img, axis=0)
        #img_preprocessed = tf.keras.applications.resnet_v2.preprocess_input(img_batch)
        prediction = model.predict(img_batch)
        prediction = tf.nn.sigmoid(prediction)
        prediction = np.array(prediction)
        idx = np.argmax(prediction)
        prediction = class_names[idx]
        if prediction == gt[-1]:
            if prediction == "m":
                TP += 1
            else:
                TN += 1
        else:
            if prediction == "m":
                FP += 1
            else:
                FN += 1


        predictions.append(prediction)
print(predictions)
print("acc: ", (TP + TN)/(TP + TN + FP + FN))
print("sens: ", TP/(TP + FN))
print("spec: ", TN/(TN + FP))        
results = model.evaluate(test_ds)
print(results)

# for i in range(0,100):        
#     for j in range(1,11):
#         number = i*10+j
#         j = "{0:0=2d}".format(j)
        
#         if str(number) in train:
#             if gender == 'm':
#                 shutil.copy2('awecrop/'+class_names[i]+'/'+j+'.png', 'custom/AWEgender/train/m/'+str(file_count)+'.png')
#             else:
#                 shutil.copy2('awecrop/'+class_names[i]+'/'+j+'.png', 'custom/AWEgender/train/f/'+str(file_count)+'.png')
#         else:
#             if gender == 'm':
#                 shutil.copy2('awecrop/'+class_names[i]+'/'+j+'.png', 'custom/AWEgender/test/m/'+str(file_count)+'.png')
#             else:
#                 shutil.copy2('awecrop/'+class_names[i]+'/'+j+'.png', 'custom/AWEgender/test/f/'+str(file_count)+'.png')

#         file_count += 1



# # predictions = model.predict(img_array)
# # score = tf.nn.softmax(predictions[0])

# print(results)
