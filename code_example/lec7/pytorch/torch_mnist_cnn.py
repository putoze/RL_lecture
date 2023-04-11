#
# Use CNN to classify MNIST digits dataset
#  - CNN Parameters based on associated TF-Keras CNN/MNIST code example 
#  - Uses PyTorch torch.nn layers API's
#

#
# 1. Import and prepare the complete MNIST dataset
# 2. Create PyTorch  model for the CNN
# 3. Train the model
# 4. Validate with test data, generate confusion matrices
#

from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(123)      #let's make things repeatable!

#GPU acceleration available? If so, let's use it.
#
has_cuda=torch.cuda.is_available()
if has_cuda: tdevice='cuda'
else: tdevice='cpu'


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

#"un-flatten" the images (restore them to 2-D)
mnist.data=mnist.data.reshape(mnist.data.shape[0],28,28)

#
# Scale & center image data (range, -1 to 1)
#
mnist_scaled=((mnist.data/255.0) - 0.5)*2

#PyTorch Conv2d operator expects 4-D array (n_samples, n_channels, height, width)
# so we need to insert a dummy dimension for the channel index (grayscale images have 1 channel)
mnist_scaled=np.expand_dims(mnist_scaled, 1)

#
# Split training, testing datasets
#   (test dataset which is 15% of total)
#
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(mnist_scaled, mnist.target, test_size=0.15, random_state=11, stratify=mnist.target)

#let's convert the data arrays to PyTorch tensors up front, so we
# don't need to do the conversions during every function call
data_train=torch.tensor(data_train,dtype=torch.float32,device=tdevice)
data_test=torch.tensor(data_test,dtype=torch.float32,device=tdevice)

#openml targets are strings, need to convert to floats
target_train=target_train.astype(np.float32)
target_test=target_test.astype(np.float32)


print('...done!\n')

#
#
############################################


############################################
# 2. Create PyTorch torch.nn layers model for CNN
#    2 Conv2D layers
#    2 Dense (Linear) layers, output layer has 10 nodes (Softmax)
#

print('Creating CNN model...')
    
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cnn1=torch.nn.Conv2d(1,32,(3,3))
        self.cnn2=torch.nn.Conv2d(32,64,(3,3))
        self.drop1=torch.nn.Dropout2d(0.25)
        self.pool1=torch.nn.MaxPool2d((2,2))
        self.flat1=torch.nn.Flatten()
        self.dense1=torch.nn.Linear(9216,128)
        self.drop2=torch.nn.Dropout(0.5)
        self.dense2=torch.nn.Linear(128,10)
    
    #forward computation through CNN    
    def forward(self, feature_input):
        #forward propagate input through network layers
        output=F.relu(self.cnn1(feature_input))
        output=F.relu(self.cnn2(output))
        output=self.pool1(output)
        output=self.drop1(output)
        output=self.flat1(output)
        output=F.relu(self.dense1(output))
        output=self.drop2(output)
        output=self.dense2(output)
    
        #note, we do not apply softmax here, since it will be included
        # in our cost and prediction functions later        
       
        return output
    
    #function to make classification predictions with CNN
    def predict(self, feature_input):
        predict=torch.argmax(torch.softmax(self.forward(feature_input),dim=1),axis=1)
    
        return predict.detach().cpu().numpy()


#the cost function, cross-entropy for classification
loss_func=torch.nn.CrossEntropyLoss()
def cost(model_out, target_input):
    target_input=torch.tensor(target_input, dtype=torch.long, device=tdevice)
    cost=loss_func(model_out,target_input)
    
    return cost

print('...done!\n')

#
#
############################################


############################################
# 3. Train the model
#

print('Training the CNN...')

#instantiate an CNN model
cnn=CNN()
if has_cuda: cnn.cuda()

#instantiate the optimizer
optim = torch.optim.SGD(cnn.parameters(),lr=0.05,momentum=0.9)

# train the model for n_epochs, using batches
#
batch_size=100
batches=range(0,len(data_train),batch_size)
n_epochs=12
training_costs=[]
for e in range(n_epochs):
    for b in batches:
        optim.zero_grad()
        cost_tmp=cost(cnn.forward(data_train[b:b+batch_size]),target_train[b:b+batch_size])
        cost_tmp.backward()
        optim.step()
            
        training_costs.append(float(cost_tmp))
        print('Epoch: {}/{}  Cost: {}'.format(e, b+batch_size, float(cost_tmp)))
    print('------------------------\n')
     


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

#switch model from training to evaluation mode
# (our model has dropout layers, so explicitly need to do this here)
cnn.eval()

#make classification predictions (training data)
# (note, doing this incrementally and also with no_grad to reduce memory overhead.)
with torch.no_grad():
    pred=np.array([],dtype=np.float32)
    for b in batches:
        pred=np.concatenate((pred, cnn.predict(data_train[b:b+batch_size])),axis=0)

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
    pred=cnn.predict(data_test)

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
