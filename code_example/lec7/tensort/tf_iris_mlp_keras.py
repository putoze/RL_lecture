#
# Use Multi-layer Perceptron (MLP, or feed-forward dense DNN) with
#  one 4-node hidden layer to classify all 3 classes in the complete Iris dataset
#  
# Uses TensorFlow Keras
#

#
# 1. Import and prepare the complete 3-class, 4-feature Iris dataset
# 2. Create TensorFlow Keras model for the MLP (DNN)
# 3. Train the model
# 4. Validate with test data, generate confusion matrices
#

from sklearn import datasets
from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()  #turn on Eager mode...
tf.set_random_seed(123)      #let's make things repeatable!


############################################
# 1. Import and prepare the Iris dataset
#

#load the Iris dataset
iris=datasets.load_iris()
iris_data=iris.data
iris_target=iris.target

#scale the dataset using standardization
#
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
std_scaler.fit(iris_data)
ss_iris_data=std_scaler.transform(iris_data)


#create a test dataset which is 15% of total
#
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(ss_iris_data, iris_target, test_size=0.15, random_state=11, stratify=iris_target)

#convert targets to binary one-hot format
#
from sklearn.preprocessing import LabelBinarizer
lb_enc=LabelBinarizer()
lb_enc.fit(list(range(3)))
target_train_1h=lb_enc.transform(target_train)
target_test_1h=lb_enc.transform(target_test)

#
#
############################################


############################################
# 2. Create TF Keras model for MLP
#    1 hidden layer with 4 nodes (ReLU), 1 output layer with 3 nodes (Softmax)
#

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(4, activation='relu', input_dim=4))
model.add(Dense(3, activation='softmax'))

#
#
############################################

############################################
# 3. Train the model
#

#instantiate the optimizer
from tensorflow.keras.optimizers import SGD
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

#compile the Keras model
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#train the model
model.fit(data_train, target_train_1h,
          epochs=100,
          batch_size=10)

#
#
############################################


############################################
# 4. Plot training & validation metrics 
#

#simple function to predict classes from Keras model
def predict(feature_input):
    return tf.argmax(model.predict(feature_input),axis=1).numpy()

#make classification predictions (training data)
# 
pred=predict(data_train)

#impedance matching...
target_train.shape=(-1,)
pred.shape=(-1,)

#Generate metrics report using sklearn
#accuracy report & confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print('%%%%%%%%%%%%%%%%%%%%\n')

print("Classification Report (training data):")
print(classification_report(target_train, pred))
print("Confusion Matrix (training data):")
print(confusion_matrix(target_train, pred))
print('%%%%%%%%%%%%%%%%%%%%\n')

#make classification predictions (testing data)
# 
pred=predict(data_test)

#impedance matching...
target_test.shape=(-1,)
pred.shape=(-1,)

#generate metrics report using sklearn
#accuracy report & confusion matrix
print("Classification Report (testing data):")
print(classification_report(target_test,pred))
print("Confusion Matrix (testing data):")
print(confusion_matrix(target_test, pred))

#
#
############################################
