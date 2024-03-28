import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from sklearn.metrics import confusion_matrix
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pathlib
import time

epochs =10

#training Data
train_dataset_url = "file:///content/drive/MyDrive/DeepLearning/train.tgz"
train_data_dir = tf.keras.utils.get_file('train', origin=train_dataset_url, untar=True)
train_data_dir = pathlib.Path(train_data_dir)

#Validation Data
val_dataset_url = "file:///content/drive/MyDrive/DeepLearning/validation.tar.gz"
val_data_dir = tf.keras.utils.get_file('validation', origin=val_dataset_url, untar=True)
val_data_dir = pathlib.Path(val_data_dir)

image_count = len(list(train_data_dir.glob('*/*.jpg')))
print("Training data count:", image_count)
count = len(list(val_data_dir.glob('*/*.jpg')))
print("Validation data count:", count)

train_empty = list(train_data_dir.glob('empty/*'))
PIL.Image.open(str(train_empty[100]))

train_still = list(train_data_dir.glob('still/*'))
PIL.Image.open(str(train_still[0]))

train_walking = list(train_data_dir.glob('walking/*'))
PIL.Image.open(str(train_walking[0]))

batch_size =32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_dir, validation_split=0.2, subset="training",
    seed=123, image_size=(img_height, img_width), batch_size = batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_data_dir, validation_split=0.2, subset="validation",
    seed=123, image_size=(img_height, img_width), batch_size = batch_size)

class_names = train_ds.class_names
print("CLASS NAMES: ",class_names)

#print("LABELS", labels)
for i in range(5):
  ax = plt.subplot(1, 5, i+1)
  img = plt.imread(str(train_empty[i+20]))
  plt.imshow(img, aspect='auto')
  plt.title("empty")
  plt.axis("off")
plt.show()

for i in range(5):
  ax = plt.subplot(1, 5, i+1)
  img = plt.imread(str(train_still[i+10]))
  plt.imshow(img, aspect='auto')
  plt.title("still")
  plt.axis("off")
plt.show()

for i in range(5):
  ax = plt.subplot(1, 5, i+1)
  img = plt.imread(str(train_walking[i+15]))
  plt.imshow(img, aspect='auto')
  plt.title("walking")
  plt.axis("off")

plt.show()

######################
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
#pixcel values are now in '[0,1]'
print("MIN and MAX",np.min(first_image), np.max(first_image))


############################
### MLP model/Neural Network
############################
print("****************MLP (MultiLayer Perceptron) Model ")
num_classes = len(class_names)


model = keras.Sequential([
keras.layers.Flatten(),
keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

"""
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('MLP Training and Validation accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training loss')
plt.plot(epochs_range, val_loss, label='Validation loss')
plt.legend(loc='lower right')
plt.title('MLP Training and Validation loss')
plt.show()
loss, accuracy = model.evaluate(val_ds)
print("MLP Validation accuracy: {:.2f}%".format(accuracy * 100))
"""
#################################################
# START OF A BASIC CNN MODEL with 1 HIDDEN Layer
#################################################
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("****************CNN MODEL with 1 HIDDEN Layer")
num_classes = len(class_names)
model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, padding='same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(num_classes, activation='softmax')
    ])
model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

model.summary()

# Train the Model for 10 epochs with the Leras model.fit

start_time = time.time()
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
end_time = time.time()

print("Training time CNN-1L: {:.2f} seconds".format(end_time - start_time))

#Visulalize the training Results
acc=history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training accuracy')
plt.plot(epochs_range, val_acc, label='validation accuracy')
plt.legend(loc='lower right')
plt.title('CNN 1 Layer Train & Val Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training loss')
plt.plot(epochs_range, val_loss, label='validation loss')
plt.legend(loc='upper right')
plt.title('CNN 1 Layer Train & Val  loss')
plt.show()

#confusion Matrix
loss, accuracy = model.evaluate(val_ds)
print("Validation accuracy: {:.2f}%".format(accuracy * 100))

y_predict_prob = model.predict(val_ds)
y_predict = np.argmax(y_predict_prob, axis=1)
y_true = np.concatenate([y for x, y in val_ds], axis=0)
accuracy = np.mean(y_predict == y_true) * 100
print("Test accuracy: {:.2f}%".format(accuracy))
cm = confusion_matrix(y_true, y_predict)
plt.figure(figsize=(6, 6))

sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

#######################
print("*************CNN MODEL with MULTIPLE HIDDEN Layer")
#######################
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)
model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, padding='same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(num_classes, activation='softmax')
    ])
model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

model.summary()
#Visulalize the training Results
# Train the Model for 10 epochs with the Leras model.fit

start_time = time.time()
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
end_time = time.time()

print("Training time CNN-M: {:.2f} seconds".format(end_time - start_time))

acc=history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training accuracy')
plt.plot(epochs_range, val_acc, label='validation accuracy')
plt.legend(loc='lower right')
plt.title('CNN Multi-L Train & Val Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training loss')
plt.plot(epochs_range, val_loss, label='validation loss')
plt.legend(loc='upper right')
plt.title('CNN Multi-L Train & Val  loss')
plt.show()

#confusion Matrix
loss, accuracy = model.evaluate(val_ds)
print("Validation accuracy: {:.2f}%".format(accuracy * 100))

y_predict_prob = model.predict(val_ds)
y_predict = np.argmax(y_predict_prob, axis=1)
y_true = np.concatenate([y for x, y in val_ds], axis=0)
accuracy = np.mean(y_predict == y_true) * 100
print("Test accuracy: {:.2f}%".format(accuracy))
cm = confusion_matrix(y_true, y_predict)
plt.figure(figsize=(6, 6))

sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
#########################
# END of BASIC CNN MOEL
#########################


#########################################
##STATE of the ART CNN model Resetnet50
#########################################
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("*************CNN MODEL with Resnet50")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Flatten,Dense,MaxPool2D,BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.preprocessing import image



resnet_model = ResNet50(include_top=False, weights='imagenet')

for layer in resnet_model.layers:
    layer.trainable = False

model = keras.Sequential([
    resnet_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
"""
base_model = ResNet50(include_top=False, weights= 'imagenet')
x= base_model.output
x= GlobalAveragePooling2D()(x)
x= Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)
model =Model(inputs=base_model.input, outputs=predictions)

for layers in base_model.layers:
  layers.trainable = False

  model.compile(optimizer='adam', loss='catagorical_crossentrophy', metrics=['accuracy'])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
"""
#Visulalize the training Results
# Train the Model for 10 epochs with the Leras model.fit
start_time = time.time()
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
end_time = time.time()

print("Training time ResNet-50: {:.2f} seconds".format(end_time - start_time))

acc=history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training accuracy')
plt.plot(epochs_range, val_acc, label='validation accuracy')
plt.legend(loc='lower right')
plt.title('ResNet50 Train & Val Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training loss')
plt.plot(epochs_range, val_loss, label='validation loss')
plt.legend(loc='upper right')
plt.title('ResNet50 Train & Val  loss')
plt.show()

#confusion Matrix
loss, accuracy = model.evaluate(val_ds)
print("Validation accuracy: {:.2f}%".format(accuracy * 100))

y_predict_prob = model.predict(val_ds)
y_predict = np.argmax(y_predict_prob, axis=1)
y_true = np.concatenate([y for x, y in val_ds], axis=0)
accuracy = np.mean(y_predict == y_true) * 100
print("Test accuracy: {:.2f}%".format(accuracy))
cm = confusion_matrix(y_true, y_predict)
plt.figure(figsize=(6, 6))

sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
