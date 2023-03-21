#
#First-visit Monte-Carlo control, optimal policy estimation demo
# using 1-D & 2-D Mouse-Cheese examples from lectures.
#
# MC Implementation: Exploring-starts first-visit Monte Carlo
#
# Note: 1-D Cat_and_Mouse environment uses labeling for left/right actions
#        that is opposite from my MDP lecture slide Mouse-Cheese examples (doh!)
#

from mcc_q import mcc_q
from cat_and_mouse import Cat_and_Mouse
import numpy as np
np.set_printoptions(precision=3)

#################################################################
#
# 1. Estimate optimal policy & q using Monte-Carlo method w/ exploring starts
#    for 1-D Mouse-Cheese example from lectures
#
#

cm=Cat_and_Mouse(rows=1,columns=7,mouseInitLoc=[0,3], cheeseLocs=[[0,0],[0,6]],stickyLocs=[[0,2]],slipperyLocs=[[0,4]])

#Compute pi(s) & q(s,a) using Monte-Carlo method
(policy,q)=mcc_q(cm,0.9,5000,100)

print('q-function:')
print(q)

print('\nPolicy:')
print(policy)

#let's walk the policy starting from a few different
# initial states (results are saved to animated .gif files)
#cm.policy2gif(policy,[0,3],'cm1d_1')
#cm.policy2gif(policy,[0,2],'cm1d_2')

#
# End
#
#################################################################

#################################################################
#
# 2. Compute & demo optimal policy for 2-D Mouse-and-cats example
#    (generates animated .gif of mouse following estimated optimal policy)
#
#

cm=Cat_and_Mouse(slipperyLocs=[[1,1],[2,1]],stickyLocs=[[2,4],[3,4]],catLocs=[[3,2],[3,3]])

#Compute pi(s) & q(s,a) using Monte-Carlo method
(policy,q)=mcc_q(cm,0.9,50000,100)

print('\nPolicy:')
print(policy)

#let's walk the policy starting from a few different
# initial states (results are saved to animated .gif files)
#cm.policy2gif(policy,[0,0],'cm2d_1')
#cm.policy2gif(policy,[0,3],'cm2d_2')
#cm.policy2gif(policy,[3,0],'cm2d_3')

#
# End
#
#################################################################