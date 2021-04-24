import pandas as pd
import numpy as np
train_csv = pd.read_csv('sign_mnist_train.csv')
test_csv = pd.read_csv('sign_mnist_test.csv')


#Create X_train (images) and Y_train (labels)

labels = []
images = []
for row in train_csv.iterrows():
    label = row[1][0]
    image = np.array_split(row[1][1:],28)
    labels.append(label)
    images.append(image)

num_classes = len(np.unique(labels))
print('Unique Labels: ' + str(np.unique(labels)))
labels = np.array(labels)
images = np.array(images)
print(labels.shape)
print(images.shape)


#Expand dims for these 2 np arrays so that they can be set as input to TF model
labels = np.expand_dims(labels,axis=1)
images = np.expand_dims(images,axis=3)
print(labels.shape)
print(images.shape)


#Create X_test (images) and Y_test (labels); we will also use this for Validation

labels_test = []
images_test = []
for row in test_csv.iterrows():
    label = row[1][0]
    image = np.array_split(row[1][1:],28)
    labels_test.append(label)
    images_test.append(image)

labels_test = np.array(labels_test)
images_test = np.array(images_test)
print(labels_test.shape)
print(images_test.shape)


#Expand dims for these 2 np arrays so that they can be set as input to TF model
labels_test = np.expand_dims(labels_test,axis=1)
images_test = np.expand_dims(images_test,axis=3)
print(labels_test.shape)
print(images_test.shape)


X_train = images.astype(float)
Y_train = labels.astype(float)
X_test = images_test.astype(float)
Y_test = labels_test.astype(float)


#Split the training and test sets
from sklearn.model_selection import train_test_split
X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 12345)


print(X_train.shape)
print(X_validate.shape)
print(Y_train.shape)
print(Y_validate.shape)


#Create a Image Generator for X_train
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255,rotation_range=45, width_shift_range=0.25,
    height_shift_range=0.15,shear_range=0.15, zoom_range=0.2, fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1/255)
valid_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow(X_train, Y_train, batch_size=32)
test_generator =  test_datagen.flow(X_test,Y_test,batch_size=32)
valid_generator = valid_datagen.flow(X_validate,Y_validate,batch_size=32)


#Define and compile the TF Model
import tensorflow as tf

model = tf.keras.Sequential(
[
    tf.keras.layers.Conv2D(16, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2)),
     tf.keras.layers.Conv2D(128, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
])

#get_ipython().getoutput("!!!! IF YOU USE ADAM , THE ACCURACY STAYS AT 0 , USE SGD OPTIMIZER !!!")
model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics = ['accuracy'])
model.summary()


history = model.fit(train_generator,
                    epochs=500,
                    validation_data =valid_generator,
                    callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
])


# Plot the chart for accuracy and loss on both training and validation
get_ipython().run_line_magic("matplotlib", " inline")
import matplotlib.pyplot as plt
acc = history.history['accuracy']# Your Code Here
val_acc =history.history['val_accuracy'] # Your Code Here
loss = history.history['loss']# Your Code Here
val_loss = history.history['val_loss']# Your Code Here

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


score = model.evaluate(test_generator, verbose = 0) 
print('Test loss:', score[0])
print('Test accuracy:', score[1])
