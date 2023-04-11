#
#Q-learning, optimal policy estimation demo
# using 1-D & 2-D Mouse-Cheese examples from lectures & "cliff" world
#
# Note: 1-D Cat_and_Mouse environment uses labeling for left/right actions
#        that is opposite from my MDP lecture slide Mouse-Cheese examples (doh!)
#

from Expect_Sarsa import Expect_Sarsa
from cat_and_mouse import Cat_and_Mouse
import numpy as np
np.set_printoptions(precision=3)
show_plots=True

#################################################################
#
# 1. Estimate optimal policy & q using Q-learning method
#    for 1-D Mouse-Cheese example from lectures
#
#

cm=Cat_and_Mouse(rows=1,columns=7,mouseInitLoc=[0,3], cheeseLocs=[[0,0],[0,6]],stickyLocs=[[0,2]],slipperyLocs=[[0,4]])

#Compute pi(s) & q(s,a) using Q-learning method
(policy,q)=Expect_Sarsa(cm,cm.currentState(),0.9,1.0,0.1,5,100,decayEpsilon=True,showPlots=show_plots)

print('q-function:')
print(q)

print('\nPolicy:')
print(policy)

#let's walk the policy starting from a few different
# initial states (results are saved to animated .gif files)
tot_steps=cm.policy2gif(policy,[0,3],'cm1d_ql1')
print('Total steps starting from {}: {}'.format([0,3],tot_steps))
tot_steps=cm.policy2gif(policy,[0,2],'cm1d_ql2')
print('Total steps starting from {}: {}'.format([0,2],tot_steps))

#
# End
#
#################################################################

#################################################################
#
# 2. Compute & demo optimal policy for stochastic 2-D Mouse-and-cats example
#    (generates animated .gif of mouse following estimated optimal policy)
#
#

cm=Cat_and_Mouse(slipperyLocs=[[1,1],[2,1]],stickyLocs=[[2,4],[3,4]],catLocs=[[3,2],[3,3]])

#Compute pi(s) & q(s,a) using Q-learning method
(policy,q)=Expect_Sarsa(cm,cm.currentState(),0.9,1.0,0.4,1500,250,decayEpsilon=True,showPlots=show_plots)

print('\nPolicy:')
print(policy)

#let's walk the policy starting from a few different
# initial states (results are saved to animated .gif files)
tot_steps=cm.policy2gif(policy,[0,0],'cm2d_ql1')
print('Total steps starting from {}: {}'.format([0,0],tot_steps))
tot_steps=cm.policy2gif(policy,[0,0],'cm2d_ql1a')
print('Total steps starting from {}: {}'.format([0,0],tot_steps))
tot_steps=cm.policy2gif(policy,[0,3],'cm2d_ql2')
print('Total steps starting from {}: {}'.format([0,3],tot_steps))
tot_steps=cm.policy2gif(policy,[3,0],'cm2d_ql3')
print('Total steps starting from {}: {}'.format([3,0],tot_steps))

#
# End
#
#################################################################

#################################################################
#
# 3. Compute & demo optimal policy for deterministic 2-D Mouse-and-cats example
#    (generates animated .gif of mouse following estimated optimal policy)
#
#

cm=Cat_and_Mouse(catLocs=[[2,2],[2,3]])

#Compute pi(s) & q(s,a) using Q-learning method
(policy,q)=Expect_Sarsa(cm,cm.currentState(),0.95,1.0,0.9,750,250,decayEpsilon=True,showPlots=show_plots)

print('\nPolicy:')
print(policy)

#let's walk the policy starting from a few different
# initial states (results are saved to animated .gif files)
tot_steps=cm.policy2gif(policy,[0,0],'cm2d_det_ql1')
print('Total steps starting from {}: {}'.format([0,0],tot_steps))
tot_steps=cm.policy2gif(policy,[0,3],'cm2d_det_ql2')
print('Total steps starting from {}: {}'.format([0,3],tot_steps))
tot_steps=cm.policy2gif(policy,[3,0],'cm2d_det_ql3')
print('Total steps starting from {}: {}'.format([3,0],tot_steps))

#
# End
#
#################################################################

#################################################################
#
# 4. An example inspired by Sutton's cliff-walking case (section 6.5, 2nd ed)
#    Here the "cliff" is replaced by a wall of cats, also use 8-way moves instead of 4
#
#    (generates animated .gif of mouse following estimated optimal policy)
#
#

cliff=Cat_and_Mouse(rows=4,columns=12,mouseInitLoc=[3,0],cheeseLocs=[[3,11]],catLocs=[[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10]])
#cm.render()


#Compute pi(s) & q(s,a) using Q-learning method
(policy,q)=Expect_Sarsa(cliff,cliff.currentState(),0.9,0.2,0.2,1500,150,decayEpsilon=False,showPlots=show_plots)

print('\nPolicy:')
print(policy)

#let's walk the policy starting from a few different
# initial states (results are saved to animated .gif files)
tot_steps=cliff.policy2gif(policy,[3,0],'cliff_ql',maxEpisodeLen=15)
print('Total steps starting from {}: {}'.format([3,0],tot_steps))

#
# End
#
#################################################################