from organized_dataset import * 
from keras.utils import np_utils
import pickle

training_data, validation_data, test_data = create_train_validation_test("gender")

num_classes = 2

IMG_SIZE = 150

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

pickle_out = open("x_training_gender.pickle","wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()

pickle_out = open("y_training_gender.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open("x_validation_gender.pickle","wb")
pickle.dump(x_validation, pickle_out)
pickle_out.close()

pickle_out = open("y_validation_gender.pickle","wb")
pickle.dump(y_validation, pickle_out)
pickle_out.close()

pickle_out = open("x_test_gender.pickle","wb")
pickle.dump(x_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test_gender.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()


