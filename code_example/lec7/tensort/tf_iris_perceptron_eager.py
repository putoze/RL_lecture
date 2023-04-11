#
# Train Perceptron to classify Iris data
#   using TensorFlow Eager mode
#
#  (Note: This example also demonstrate usage of
#                   sklearn & tensorflow features together)
#

#
# 1. Extract only the first two linearly separable classes
#     from the dataset.  Use only Petal & Sepal Length features
#     for classification and plotting, Scale the data using Standardization,
#     Create test & training datasets (test dataset 15% of total)
# 2. Create TensorFlow model for Perceptron
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

#extract subset of data, only the first two classes
# and only two features petal & sepal length
iris_subset=iris.data[:100,[0,2]]
target_subset=iris.target[:100]

#scale the dataset using standardization
#
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
std_scaler.fit(iris_subset)
ss_iris_subset=std_scaler.transform(iris_subset)

#create a test dataset which is 15% of total
#
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(ss_iris_subset, target_subset, test_size=0.15, random_state=11, stratify=target_subset)

#a bit of Tensor "impedance" matching for our TF model
target_train.shape=(-1,1)

#
#
############################################


############################################
# 2. Create TF model for Perceptron
#

#Build single Perceptron model using TF
#
weight=tf.Variable(tf.random.normal(shape=(2,1),stddev=0.25))
bias=tf.Variable(0.0)

#the basic perceptron cell
def perceptron(feature_input):
    feature_input=tf.convert_to_tensor(feature_input,dtype=tf.float32)
    
    temp=tf.matmul(feature_input,weight)+bias
    output=tf.sigmoid(temp)
    
    return output

#function to make classification predictions with perceptron
def predict(feature_input):
    predict=tf.round(perceptron(feature_input))
    
    return predict.numpy()

#the cost function, let's just use MSE to keep things simple
#  (note, we're implementing manually, but we could also use tf.losses.mean_squared_error here)
def cost(model_out, target_input):
    target_input=tf.convert_to_tensor(target_input, dtype=tf.float32)
    cost=tf.reduce_mean(tf.square(model_out - target_input))
    
    return cost

#
#
############################################


############################################
# 3. Train the model (notice there is no Session here, we're using eager mode!)
#

#instantiate the optimizer
optim = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# train the model for n_epochs
#
n_epochs=601
training_costs=[]
for e in range(n_epochs):
    with tf.GradientTape() as tape:
        cost_tmp=cost(perceptron(data_train),target_train)
        grads=tape.gradient(cost_tmp,[weight,bias])
        optim.apply_gradients(zip(grads, [weight,bias]))
        training_costs.append(cost_tmp)
        if not e % 50:
            print('Epoch %4d: %.4f' % (e, cost_tmp))

#plot cost vs. epochs
plt.plot(training_costs)
plt.show()

#
#
############################################


############################################
# 4. Plot metrics (generate confusion matrices
#     for training & testing datasets
#

#make classification predictions (training data)
# 
pred=predict(data_train)

#impedance matching...
target_train.shape=(-1,)
pred.shape=(-1,)

#generate confusion matrix for training data using TF 
#
confuse=tf.confusion_matrix(labels=target_train, predictions=pred)
print('%%%%%%%%%%%%%%%%%%%%\n')
print("TensorFlow Confusion Matrix (training data):")
print(confuse)
print()

#...or, we can also generate metrics report using sklearn
#accuracy report & confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
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

#generate confusion matrix for testing data using TF 
#
confuse=tf.confusion_matrix(labels=target_test, predictions=pred)
print("TensorFlow Confusion Matrix (testing data):")
print(confuse)
print()

#...or, we can also generate metrics report using sklearn
#accuracy report & confusion matrix
print("Classification Report (testing data):")
print(classification_report(target_test,pred))
print("Confusion Matrix (testing data):")
print(confusion_matrix(target_test, pred))

#
#
############################################
