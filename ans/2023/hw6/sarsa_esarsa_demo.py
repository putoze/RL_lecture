#
#Optimal policy estimation demo:
# 1. Sarsa
# 2. Expected Sarsa
#
# Demo'ed using 2-D Mouse-Cheese environment
#

from sarsa import sarsa
from esarsa import esarsa
from cat_and_mouse import Cat_and_Mouse
import numpy as np
np.set_printoptions(precision=3)
show_plots=True

#################################################################
#
# 1. Sarsa: Compute & demo optimal policy for 2-D Mouse-and-cats example
#    (generates animated .gif of mouse following estimated optimal policy)
#
#

cm=Cat_and_Mouse(slipperyLocs=[[1,1],[2,1]],stickyLocs=[[2,4],[3,4]],catLocs=[[3,2],[3,3]])

#Compute pi(s) & q(s,a) using SARSA method
(policy,q)=sarsa(cm,cm.currentState(),0.9,0.25,0.01,5000,150,decayEpsilon=True,showPlots=show_plots)

print('\nPolicy:')
print(policy)

#let's walk the policy starting from a few different
# initial states (results are saved to animated .gif files)
tot_steps=cm.policy2gif(policy,[0,0],'cm2d_sarsa1')
print('Total steps starting from {}: {}'.format([0,0],tot_steps))
tot_steps=cm.policy2gif(policy,[0,3],'cm2d_sarsa2')
print('Total steps starting from {}: {}'.format([0,3],tot_steps))
tot_steps=cm.policy2gif(policy,[3,0],'cm2d_sarsa3')
print('Total steps starting from {}: {}'.format([3,0],tot_steps))

#
# End
#
#################################################################

#################################################################
#
# 2. Expected SARSA: Compute & demo optimal policy for stochastic 2-D Mouse-and-cats example
#    (generates animated .gif of mouse following estimated optimal policy)
#
#

cm=Cat_and_Mouse(slipperyLocs=[[1,1],[2,1]],stickyLocs=[[2,4],[3,4]],catLocs=[[3,2],[3,3]])

#Compute pi(s) & q(s,a) using expected sarsa method
(policy,q)=esarsa(cm,cm.currentState(),0.9,0.25,0.4,1500,250,decayEpsilon=True,showPlots=show_plots)

print('\nPolicy:')
print(policy)

#let's walk the policy starting from a few different
# initial states (results are saved to animated .gif files)
tot_steps=cm.policy2gif(policy,[0,0],'cm2d_esarsa1')
print('Total steps starting from {}: {}'.format([0,0],tot_steps))
tot_steps=cm.policy2gif(policy,[0,3],'cm2d_esarsa2')
print('Total steps starting from {}: {}'.format([0,3],tot_steps))
tot_steps=cm.policy2gif(policy,[3,0],'cm2d_esarsa3')
print('Total steps starting from {}: {}'.format([3,0],tot_steps))

#
# End
#
#################################################################
