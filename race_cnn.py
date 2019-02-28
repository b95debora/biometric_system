from organized_dataset import * 
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D, Flatten
from keras.applications import MobileNet
from keras.applications import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.utils import np_utils
from keras_vggface.vggface import VGGFace
import pickle

training_data, validation_data, test_data = create_train_validation_test("race")

num_classes = 5
epochs = 3
IMG_SIZE = 300

x_train = []
y_train = []

x_validation = []
y_validation = []

x_test = []
y_test = []

for features,label in training_data:
    x_train.append(features)
    y_train.append(label)

x_train = np.array(x_train).reshape(-3, IMG_SIZE, IMG_SIZE, 3)
x_train = x_train/255.0
y_train = np_utils.to_categorical(y_train, num_classes)

for features,label in validation_data:
    x_validation.append(features)
    y_validation.append(label)

x_validation = np.array(x_validation).reshape(-3, IMG_SIZE, IMG_SIZE, 3)
x_validation = x_validation/255.0
y_validation = np_utils.to_categorical(y_validation, num_classes)

for features,label in test_data:
    x_test.append(features)
    y_test.append(label)

x_test = np.array(x_test).reshape(-3, IMG_SIZE, IMG_SIZE, 3)
x_test = x_test/255.0
y_test = np_utils.to_categorical(y_test, num_classes)

pickle_out = open("x_test_race.pickle","wb")
pickle.dump(x_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test_race.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

#Modello da cui partire:
base_model=VGGFace(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg') 

x = base_model.get_layer('pool5').output
x = Flatten()(x)
x = Dense(1024,activation='relu')(x)
x = Dense(1024,activation='relu')(x)
x = Dense(512,activation='relu')(x)
preds = Dense(num_classes,activation='softmax')(x) #final layer with softmax activation


#devono essere addestrati solo i nuovi livelli
for layer in base_model.layers:
    layer.trainable = False


# "fondo" il modello addestrato con quello che abbiamo creato
model=Model(inputs=base_model.input,outputs=preds)

model.summary()


train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow(x_train, y_train, batch_size=32)
valid_datagen = ImageDataGenerator()
valid_generator = valid_datagen.flow(x_validation, y_validation, batch_size=32)


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size


history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator,
                              validation_steps=STEP_SIZE_VALID,
                              epochs=epochs)

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

#salvo il modello
model.save('race.model')

