#
# Use Multi-layer Perceptron (MLP, or feed-forward dense DNN) with
#  one 4-node hidden layer to classify all 3 classes in the complete Iris dataset
#  
#  (PyTorch version)
#

#
# 1. Import and prepare the complete 3-class, 4-feature Iris dataset
# 2. Create model for the MLP (DNN)
# 3. Train the model
# 4. Validate with test data, generate confusion matrices
#

from sklearn import datasets
from matplotlib import pyplot as plt
import torch

torch.manual_seed(123)      #let's make things repeatable!


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


#
#
############################################


############################################
# 2. Create model for MLP
#    1 hidden layer with 4 nodes (ReLU), 1 output layer with 3 nodes (Softmax)
#

#Build simple MLP model
#
with torch.no_grad():
    #note: we don't want initialization calculations to be included in gradient tracking
    weight_hid=(0.25*torch.randn(size=(4,4), dtype=torch.float32)).requires_grad_(True)
    bias_hid=torch.tensor([0.0]*4, dtype=torch.float32, requires_grad=True)
    weight_out=(0.25*torch.randn(size=(4,3), dtype=torch.float32)).requires_grad_(True)
    bias_out=torch.tensor([0.0]*3, dtype=torch.float32, requires_grad=True)


#the MLP
def mlp(feature_input):
    feature_input=torch.tensor(feature_input,dtype=torch.float32)
    
    hid_out=torch.relu(torch.matmul(feature_input,weight_hid)+bias_hid)
    output=torch.matmul(hid_out,weight_out)+bias_out
    
    #note, we do not apply softmax here, since it will be included
    # in our cost and prediction functions later
    
    return output

#function to make classification predictions with MLP
def predict(feature_input):
    predict=torch.argmax(torch.softmax(mlp(feature_input),dim=1),axis=1)
    
    return predict.detach().numpy()

#the cost function, cross-entropy for classification
loss_func=torch.nn.CrossEntropyLoss()
def cost(model_out, target_input):
    target_input=torch.tensor(target_input, dtype=torch.long)
    cost=loss_func(model_out,target_input)
    
    return cost

#
#
############################################


############################################
# 3. Train the model
#

#instantiate the optimizer
optim = torch.optim.SGD([weight_hid,bias_hid,weight_out,bias_out], lr=0.1)

# train the model for n_epochs
#
n_epochs=601
training_costs=[]
for e in range(n_epochs+1):
    optim.zero_grad()
    cost_tmp=cost(mlp(data_train),target_train)
    cost_tmp.backward()
    optim.step()
    training_costs.append(float(cost_tmp))
    if not e % 50:
        print('Epoch %4d: %.4f' % (e, float(cost_tmp)))

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
