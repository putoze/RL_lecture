#
# Use Multi-layer Perceptron (MLP, or feed-forward dense DNN) with
#  one 4-node hidden layer to classify all 3 classes in the complete Iris dataset
#  
# TensorFlow eager mode, Core API version
#

#
# 1. Import and prepare the complete 3-class, 4-feature Iris dataset
# 2. Create TensorFlow model for the MLP (DNN)
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
# 2. Create TF model for MLP
#    1 hidden layer with 4 nodes (ReLU), 1 output layer with 3 nodes (Softmax)
#

#Build simple MLP model using TF
#
weight_hid=tf.Variable(tf.random.normal(shape=(4,4),stddev=0.25))
bias_hid=tf.Variable([[0.0]*4],shape=(1,4))
weight_out=tf.Variable(tf.random.normal(shape=(4,3),stddev=0.25))
bias_out=tf.Variable([[0.0]*3],shape=(1,3))

#the MLP
def mlp(feature_input):
    feature_input=tf.convert_to_tensor(feature_input,dtype=tf.float32)
    
    hid_out=tf.nn.relu(tf.matmul(feature_input,weight_hid)+bias_hid)
    output=tf.matmul(hid_out,weight_out)+bias_out
    
    #note, we do not apply softmax here, since it will be included
    # in our cost and prediction functions later
    
    return output

#function to make classification predictions with MLP
def predict(feature_input):
    predict=tf.argmax(tf.nn.softmax(mlp(feature_input)),axis=1)
    
    return predict.numpy()

#the cost function, cross-entropy for classification
def cost(model_out, target_input):
    target_input=tf.convert_to_tensor(target_input, dtype=tf.float32)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_input,logits=model_out))
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
for e in range(n_epochs+1):
    with tf.GradientTape() as tape:
        cost_tmp=cost(mlp(data_train),target_train_1h)
        grads=tape.gradient(cost_tmp,[weight_hid,bias_hid,weight_out,bias_out])
        optim.apply_gradients(zip(grads, [weight_hid,bias_hid,weight_out,bias_out]))
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
# 4. Plot training & validation metrics 
#

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
