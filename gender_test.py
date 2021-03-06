from keras.models import load_model
from keras.utils import np_utils
import numpy as np
import keras
import pickle

num_classes = 2
# load model
model = load_model("gender.model")
    
pickle_in = open("x_test_gender.pickle","rb")
x_test = pickle.load(pickle_in)

pickle_in = open("y_test_gender.pickle","rb")
y_test = pickle.load(pickle_in)



val_loss, val_acc = model.evaluate(x_test, y_test)
print("================================")
print("Valutazione modello su test set:")
print("Loss: " + str(val_loss))
print("Accuratezza: "+str(val_acc))
