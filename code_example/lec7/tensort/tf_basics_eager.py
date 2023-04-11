#
# Simple example to demonstrate TF variables,
#  constants, and expressions in Eager mode
#

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

a=tf.Variable(2.2)
b=tf.Variable(3.3)
c=tf.constant(4.4)

#@tf.function  #autograph
def compute(x,y,z):
    x=tf.convert_to_tensor(x)
    y=tf.convert_to_tensor(y)
    z=tf.convert_to_tensor(z)
    
    p=x*a + y*z*b
    out=tf.add(p,c)
        
    return out


out=compute(x=[1.0,2.0], y=[4.0,5.0], z=[6.0,7.0])

print(out)
print(out.numpy())
print(type(out))

a.assign(1.1)
out=compute(x=[1.0,2.0], y=[4.0,5.0], z=[6.0,7.0])

print(out)
print(out.numpy())
print(type(out))

