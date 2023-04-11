#
# Train Perceptron to classify Iris data
#   using PyTorch
#
#  (Note: This example also demonstrate usage of
#                   sklearn & pytorch features together)
#

#
# 1. Extract only the first two linearly separable classes
#     from the dataset.  Use only Petal & Sepal Length features
#     for classification and plotting, Scale the data using Standardization,
#     Create test & training datasets (test dataset 15% of total)
# 2. Create PyTorch model for Perceptron
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

#a bit of Tensor "impedance" matching for our model
target_train.shape=(-1,1)

#
#
############################################


############################################
# 2. Create model for Perceptron
#

#Build single Perceptron model using PyTorch
#
with torch.no_grad():
    #note: we don't want initialization calculations to be included in gradient tracking
    weight=(0.25*torch.randn(size=(2,1), dtype=torch.float32)).requires_grad_(True)
    bias=torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


#the basic perceptron cell
def perceptron(feature_input):
    feature_input=torch.tensor(feature_input,dtype=torch.float32)
    
    temp=torch.matmul(feature_input,weight)+bias
    output=torch.sigmoid(temp)
    
    return output

#function to make classification predictions with perceptron
def predict(feature_input):
    predict=torch.round(perceptron(feature_input))
    
    return predict.detach().numpy()

#the cost function, let's just use MSE to keep things simple
#  (note, we're implementing manually, but we could also use torch.nn.MSELoss here)
def cost(model_out, target_input):
    target_input=torch.tensor(target_input, dtype=torch.float32)
    cost=torch.sum(torch.pow((model_out - target_input),2))*(1.0/len(model_out))
    
    return cost

#
#
############################################


############################################
# 3. Train the model
#

#instantiate the optimizer
optim = torch.optim.SGD([weight,bias], lr=0.5)

# train the model for n_epochs
#
n_epochs=201
training_costs=[]
for e in range(n_epochs):
    optim.zero_grad()
    cost_tmp=cost(perceptron(data_train),target_train)
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
# 4. Plot metrics (generate confusion matrices
#     for training & testing datasets
#
 
#make classification predictions (training data)
# 
pred=predict(data_train)
 
#impedance matching...
target_train.shape=(-1,)
pred.shape=(-1,)
 
 
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

 
#...or, we can also generate metrics report using sklearn
#accuracy report & confusion matrix
print("Classification Report (testing data):")
print(classification_report(target_test,pred))
print("Confusion Matrix (testing data):")
print(confusion_matrix(target_test, pred))
 
#
#
############################################
