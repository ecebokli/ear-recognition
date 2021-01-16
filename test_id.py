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
test_data_dir = 'custom\\AWE\\test'
img_width, img_height = 96, 96
class_names = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '096', '097', '098', '099', '100']


def evaluate(model):
    predictions = []
    ranks = []
    for i in range(0, 100):
        ranks.append([])
    gt = []

    for i in range(0,len(class_names)):
        files = os.listdir(test_data_dir + '\\' + class_names[i])
        #print(files)
        for f in files:
            gt.append(class_names[i])
            img = image.load_img(test_data_dir + '\\' + class_names[i] + '\\' + f, target_size=(96, 96))

            #img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img, axis=0)
            #img_preprocessed = tf.keras.applications.resnet_v2.preprocess_input(img_batch)
            prediction = model.predict(img_batch, batch_size=1)

            # Apply a sigmoid since our model returns logits
            prediction = tf.nn.sigmoid(prediction)
            prediction = np.array(prediction)
            indices = (-prediction).argsort()[0]
            idx = np.argmax(prediction)
            
            for j in range(1,101):
                
                if i in indices[0:j]:
                    ranks[j-1].append(class_names[i])
                else:
                    
                    ranks[j-1].append(class_names[idx])
            #print(prediction)
            #idx = np.argmax(prediction)
            prediction = class_names[idx]
            predictions.append(prediction)
    predictions = np.array(predictions)
    gt = np.array(gt)
    #print(predictions)  
    #print(gt)
    ranks_acc = np.zeros((1,100))[0]
    print(np.sum(ranks[0]==gt), len(predictions))
    for i in range(0,100):
        #print(ranks[i])
        
        ranks_acc[i] = (np.sum(ranks[i]==gt)/len(predictions))

    return ranks_acc




model1 = tf.keras.models.load_model("mobilenet")
model2 = tf.keras.models.load_model("vgg19")
model3 = tf.keras.models.load_model("resnet-id")
model4 = tf.keras.models.load_model("xception")
model5 = tf.keras.models.load_model("inceptionresnet")

ranks_acc1 = evaluate(model1)
ranks_acc2 = evaluate(model2)
ranks_acc3 = evaluate(model3)
ranks_acc4 = evaluate(model4)
ranks_acc5 = evaluate(model5)

print("RANK1")
print(ranks_acc1[0],ranks_acc2[0],ranks_acc3[0],ranks_acc4[0],ranks_acc5[0])
print("RANK5")
print(ranks_acc1[4],ranks_acc2[4],ranks_acc3[4],ranks_acc4[4],ranks_acc5[4])
print("AUC")
print(np.sum(ranks_acc1)/100,np.sum(ranks_acc2)/100,np.sum(ranks_acc3)/100,np.sum(ranks_acc4)/100,np.sum(ranks_acc5)/100)

plot1, = plt.plot(list(range(1,101)), ranks_acc1, label='MobileNetV2')
plot2, = plt.plot(list(range(1,101)), ranks_acc2, label='VGG19')
plot3, = plt.plot(list(range(1,101)), ranks_acc3, label='Resnet-152-v2')
plot4, = plt.plot(list(range(1,101)), ranks_acc4, label='Xception')
plot5, = plt.plot(list(range(1,101)), ranks_acc5, label='Inception-Resnet')

plt.ylabel("Recognition rate")
plt.xlabel("Rank")
plt.legend(handles=[plot1,plot2,plot3,plot4,plot5])
plt.savefig("CMC")
plt.show()


# image_batch, label_batch = test_ds.as_numpy_iterator().next()
# predictions = model.predict_on_batch(image_batch).flatten()

# # Apply a sigmoid since our model returns logits
# predictions = tf.nn.sigmoid(predictions)
# print(predictions)


# print(np.sum(predictions==gt)/len(predictions))




# results = model.evaluate(test_ds)
# print(results)