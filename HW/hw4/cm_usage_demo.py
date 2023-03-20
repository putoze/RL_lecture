#
# A brief demo & explanation of the Cat_and_Mouse simulation environment features
#

from cat_and_mouse import Cat_and_Mouse
import numpy as np

# 1. If you want to create a new simulation environment, just call the Cat_and_Mouse
#   constructor and set the various parameters to tell the environment
#   where to put cats, cheese, sticky or slippery tiles, etc.  For complete
#   information on the available parameters, please see the help information
#   in Cat_and_Mouse.py (source code for the Cat_and_Mouse environment)
#
# 2. You can create 1-D or 2-D environments using the rows and columns parameters
#


# Here's an example instance of the 1-D Mouse-Cheese case we've discussed in
# our lectures on MDPs and DP. In this particular case, I have chosen to
# set the mouse initial state to tile [0,3]
cm = Cat_and_Mouse(rows=1, columns=7, mouseInitLoc=[0, 3], cheeseLocs=[
                   [0, 0], [0, 6]], stickyLocs=[[0, 2]], slipperyLocs=[[0, 4]])


# if you want to know how many states and actions are available for this environment instance, you
# can query the numStates & numActions attributes
print('NumStates: {}'.format(cm.numStates))
print('NumActions: {}'.format(cm.numActions))


# If you ever want to visualize the current state of the environment, just call the render method, as follows.
# By default, if you don't specify any parameters, it will display to the screen
cm.render()


# If you would like to take an action in the environment and see the resulting change in state
# and receive the associated reward, just use the step method, as follows.  Note, actions are
# enumerated (integers). If you want to see the allowable action values, please see the comments
# in Cat_and_Mouse.py (For 1-D environments, there are 2 actions, for 2-D environments there are 8 actions)
action = 1  # move left (West)
(newState, reward, gameOver) = cm.step(action)

print('New state: {} Reward: {} GameOver: {}'.format(newState, reward, gameOver))
cm.render()

# If you need to reset the environment, call reset
#
# Note: reset() also allows you to specify a new initial state (Mouse location) or reset
#       the random number generator
#       Examples: cm.reset(initLoc=[1,1]) or cm.reset(initLoc=[1,1], resetPrng=True)
cm.reset()


# If you would like to move the Mouse to a new initial state, use the initState method.
# initState can either take an enumerated state value (int) or coordinate [int,int]
cm.initState([0, 5])


# If you'd like to save the images to .png or .jpg files, you can provide an output filename,
# and the images will be saved to disk rather than being displayed on the screen.  Let's take
# a look at our new initial state for the Mouse
cm.render(outfile='simenv_state.png')


# what happens if we move the mouse to the right one step from this new initial state?
#
action = 0  # move right (East)
(newState, reward, gameOver) = cm.step(action)

# ah ha! we got a reward this time, found the cheese, gameOver!
print('New state: {} Reward: {} GameOver: {}'.format(newState, reward, gameOver))
cm.render(outfile='simenv_newstate.png')


# if you ever need to get the current Mouse state back from the environment, you can
# use either currentState() or currentStateXY depending on whether you want the enumerated
# state value, or XY coordinate.  Here's an example:
print('Current state (enumerated): {}'.format(cm.currentState()))
print('Current state (XY coordinate): {}'.format(cm.currentStateXY()))


# Ok, we're almost done.  So, one final cool trick you can do with this environment.  Suppose you
# have calculated a policy for the mouse, possibly even an optimal policy.  You can use the
# step() method to "walk" your mouse through the environment while following your policy,
# you can then also use the render() and rendered2gif methods to create an animated gif of the results
# Here's an example, using an optimal policy.
cm.reset()

# first, let's create the policy
policy = np.zeros((cm.numStates, 1), dtype=np.int32)
policy[0] = policy[1] = policy[2] = 1
policy[3] = policy[4] = policy[5] = policy[6] = 0

# now let's walk the mouse through the environment by following the policy
# let's use an initial state of [0,3]
#
cm.initState([0, 3])

filebase = 'cm_demo'
cm.render(outfile=filebase+'0.png')

i = 1
gameOver = False
while not gameOver:
    # look up the policy action for current state
    action = policy[cm.currentState()][0]
    # take the action, move to next state, then move to next step
    (nextState, reward, gameOver) = cm.step(action)
    # render current state of the environment to .png file
    cm.render(outfile=filebase+str(i)+'.png')
    i += 1

# let's create an animated gif of the policy walk from the .png's
cm.rendered2gif(outfile=filebase+'.gif')

cm.reset()


# And finally, if you've got a basic deterministic policy and you're not trying to do anything too fancy,
# there is a built-in function to execute the policy walk and generate an animated gif in a single function call
# Let's give it a try...
cm.policy2gif(policy, [0, 3], 'cm_demo1')
cm.policy2gif(policy, [0, 3], 'cm_demo2')

#
# Ok that's it, you now know how to use the Cat_and_Mouse simulator!
#
