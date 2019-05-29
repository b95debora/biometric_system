import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D, Flatten, Dropout
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils
from keras_vggface.vggface import VGGFace
import pickle
from pickle_read_write import *

num_classes = 5
epochs = 5
IMG_SIZE = 150

#pickle_in = open("x_training_race.pickle","rb")
#x_train = pickle.load(pickle_in)

#pickle_in = open("y_training_race.pickle","rb")
#y_train = pickle.load(pickle_in)

x_train = read_pickle("x_training_race.pickle")
y_train = read_pickle("y_training_race.pickle")

#Modello da cui partire:
base_model=VGGFace(model='vgg16', weights='vggface', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg') 

x = base_model.get_layer('pool5').output
x = Flatten()(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(1024,activation='relu')(x)
x = Dense(512,activation='relu')(x)
preds = Dense(num_classes,activation='softmax')(x) #final layer with softmax activation


#devono essere addestrati solo i nuovi livelli
for layer in base_model.layers:
    layer.trainable = False


# "fondo" il modello addestrato con quello che abbiamo creato
model=Model(inputs=base_model.input,outputs=preds)

model.summary()

optimizer = Adam(lr=0.0001)

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])


history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_split=0.3)

#salvo il modello
model.save('race.model')

#Costruisco grafici per vedere accuratezza e loss sul training e validation set

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

