#
# Quadratic Regression PyTorch "skeleton" file for homework
#
# Your task: Complete the missing code (see comments below in the code):
#            1. Quadratic regression model (w2*x**2 + w1*x + b, 3 trainable parameters)
#            2. Cost function (should use MSE)
#            3. Training loop
#            4. Plot results
#

#Description of numbered code sections below:
# 1. Generate random dataset for training
# 2. Create PyTorch quadratic regression model
# 3. Train the model
# 4. Plot fitted curve vs. data
#


import torch
import numpy as np

torch.manual_seed(123) #let's make things repeatable!


############################################
# 1. Generate the dataset
#
## create a random toy dataset for regression 
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

def make_random_data():
    x = np.random.uniform(low=-2, high=4, size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0,
                            scale=(0.5 + t*t/3),
                            size=None)
        y.append(r)
    return  x, 2.3*x**2+1.726*x -0.84 + np.array(y)

x, y = make_random_data()

plt.plot(x, y, 'o')
plt.show()

#
#
############################################


############################################
# 2. Create the quadratic regression model
#

#quadratic regression model
#
with torch.no_grad():
    #note: we don't want initialization calculations to be included in gradient tracking
    weight2=(0.25*torch.randn(size=(1,),dtype=torch.float32)).requires_grad_(True)
    weight1=(0.25*torch.randn(size=(1,),dtype=torch.float32)).requires_grad_(True)
    bias=torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def qua_reg_model(feature_input):
    feature_input=torch.tensor(feature_input,dtype=torch.float32)
    
    output=weight2*feature_input**2+weight1*feature_input+bias
    
    return output

#Fill in the missing code here!


#the MSE cost function
#
def cost(model_out, target_input):
    target_input=torch.tensor(target_input, dtype=torch.float32)
    cost=torch.sum(torch.pow((model_out - target_input),2))*(1.0/len(model_out))
        
    return cost

#Fill in the missing code here!

#
#
############################################


############################################
# 3. Train the model
#

#Fill in the missing code here!

optim = torch.optim.SGD([weight2,weight1,bias], lr=0.01)
#
#
############################################


############################################
#

#plot fitted curve vs. data
#

#Fill in the missing code here!
n_epochs=401
training_costs=[]
for e in range(n_epochs):
    optim.zero_grad()                  #zero out gradient accumulation each epoch
    cost_tmp=cost(qua_reg_model(x),y)  #compute cost
    cost_tmp.backward()                #compute gradients on model graph
    optim.step()                       #move optimizer forward one step
    training_costs.append(float(cost_tmp))
    if not e % 50:
        print('Epoch %4d: %.4f' % (e, float(cost_tmp)))
 
#plot cost vs. epochs
plt.plot(training_costs)
plt.show()

# 4. Plot fitted curve vs. data

#plot fitted curve vs. data
x_fit=np.linspace(-2.0, 4.0, 10)
w2=weight2.detach().numpy()[0]
w1=weight1.detach().numpy()[0]
b=bias.detach().numpy()

#print the final estimated w & b model values
print()
print('w2-fit: {} w1-fit: {} b-fit: {}'.format(w2,w1,b))

#plot the results
y_fit=w2*x_fit**2+w1*x_fit+b
plt.plot(x_fit,y_fit)
plt.plot(x, y, 'o')
plt.show()

#
#
############################################

