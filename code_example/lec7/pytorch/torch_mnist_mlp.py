#
# Use Multi-layer Perceptron (MLP, or feed-forward dense DNN) with
#  one 50-node hidden layer to classify MNIST digits dataset
#  
# Uses PyTorch torch.nn layers API's
#

#
# 1. Import and prepare the complete MNIST dataset
# 2. Create PyTorch  model for the MLP (DNN)
# 3. Train the model
# 4. Validate with test data, generate confusion matrices
#

from sklearn import datasets
from matplotlib import pyplot as plt
import torch

torch.manual_seed(123)      #let's make things repeatable!


############################################
# 1. Import and prepare the MNIST digits dataset
#

print('Preparing MNIST dataset...')

#Download/import original MNIST dataset, save in local disk cache
# The MNIST database contains a total of 70000 examples of handwritten
#  digits of size 28x28 pixels, 8-bit grayscale, labeled from 0 to 9
#
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False, cache=True, data_home='./mnist_dataset')

#
# Scale & center image data (range, -1 to 1)
#
mnist_scaled=((mnist.data/255.0) - 0.5)*2

#
# Split training, testing datasets
#   (test dataset which is 15% of total)
#
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(mnist_scaled, mnist.target, test_size=0.15, random_state=11, stratify=mnist.target)

#openml targets are strings, need to convert to floats
import numpy as np
target_train=target_train.astype(np.float32)
target_test=target_test.astype(np.float32)


print('...done!\n')

#
#
############################################


############################################
# 2. Create PyTorch torch.nn layers model for MLP
#    1 hidden layer with 50 nodes (ReLU), 1 output layer with 10 nodes (Softmax)
#

print('Creating MLP model...')

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dense1=torch.nn.Linear(784,50)
        self.relu1=torch.nn.ReLU()
        self.dense2=torch.nn.Linear(50,10)
    
    #forward computation through MLP    
    def forward(self, feature_input):
        feature_input=torch.tensor(feature_input,dtype=torch.float32)
    
        #forward propagate input through network layers
        output=self.dense1(feature_input)
        output=self.relu1(output)
        output=self.dense2(output)
    
        #note, we do not apply softmax here, since it will be included
        # in our cost and prediction functions later        
       
        return output
    
    #function to make classification predictions with MLP
    def predict(self, feature_input):
        predict=torch.argmax(torch.softmax(self.forward(feature_input),dim=1),axis=1)
    
        return predict.detach().numpy()


#the cost function, cross-entropy for classification
loss_func=torch.nn.CrossEntropyLoss()
def cost(model_out, target_input):
    target_input=torch.tensor(target_input, dtype=torch.long)
    cost=loss_func(model_out,target_input)
    
    return cost

print('...done!\n')

#
#
############################################


############################################
# 3. Train the model
#

print('Training the MLP...')

#instantiate an MLP model
mlp=MLP()

#instantiate the optimizer
optim = torch.optim.SGD(mlp.parameters(), lr=0.075, momentum=0.9)

# train the model for n_epochs
#
n_epochs=800
training_costs=[]
for e in range(n_epochs+1):
    optim.zero_grad()
    cost_tmp=cost(mlp.forward(data_train),target_train)
    cost_tmp.backward()
    optim.step()
        
    training_costs.append(float(cost_tmp))
    if not e % 50:
        print('Epoch %4d: %.4f' % (e, float(cost_tmp)))

#plot cost vs. epochs
plt.plot(training_costs)
plt.show()

print('...done!\n')

#
#
############################################


############################################
# 4. Plot training & validation metrics 
#

print('Plotting validation metrics...\n')

#make classification predictions (training data)
# 
with torch.no_grad():
    pred=mlp.predict(data_train)

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
with torch.no_grad(): 
    pred=mlp.predict(data_test)

#impedance matching...
target_test.shape=(-1,)
pred.shape=(-1,)

#generate metrics report using sklearn
#accuracy report & confusion matrix
print("Classification Report (testing data):")
print(classification_report(target_test,pred))
print("Confusion Matrix (testing data):")
print(confusion_matrix(target_test, pred))

print('\n...done!')

#
#
############################################
