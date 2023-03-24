#
# Cat and Mouse grid world simulation
#  Mouse robot seeks cheese while hopefully avoiding cats
#
# A few notes about this environment:
#  - In 2-D environment instances, there are 8 possible actions: {0:'E',1:'W',2:'N',3:'S',4:'SE',5:'SW',6:'NE',7:'NW'}
#  - In "1-D" horizontal environment instances, there are 2 possible actions: {0:'E',1:'W'}
#  - In "1-D" vertical environment instances, there are 2 possible actions: {0:'N',1:'S'}
#  - Cheese states are terminal absorbing states, when entering reward=1, game over
#  - Cat states are terminal absorbing states,  when entering reward=-10, game over
#  - If attempt to move off of a sticky tile, there is some probability of getting stuck
#  - When entering a slippery tile, there is some probability of slipping forward (in the direction you were moving)
#
# A few notes on the Cat_and_Mouse API methods:
#  - To move the environment one step forward, given an action, use the step method, it will return
#      a tuple containing (next_state, reward, gameOverStatusFlag)
#  - To set a new initial starting state for the mouse, use the initState method
#  - To get a rendered image of the current environment state, use the render method. You can either
#      render to screen or to a file (.png, .jpg, etc.)
#  - To query the current mouse state, use currentState or currentStateXY, depending on whether you
#      want the enumerated state index or current [x,y] location
#

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from random import Random


class Cat_and_Mouse:
    """
    Cat and Mouse Grid World Environment simulator
    """

    def __init__(self, rows=5, columns=5, mouseInitLoc=None, cheeseLocs=None,
                 catLocs=[], stickyLocs=[], slipperyLocs=[], prngSeed=456):
        """
        Input parameters:
            rows:         Number of grid rows (int), default 5
            columns:      Number of grid columns (int), default 5
            mouseInitLoc: Mouse's initial location/coordinate [(int},(int)], default [0,0]
            cheeseLocs:   List of cheese locations/coordinates, default single cheese at [rows-1,columns-1]
            catLocs:      List of cat locations/coordinates, default no cats
            stickyLocs:   List of sticky tile locations/coordinates, default none
            slipperyLocs: List of slippery tile locations/coordinates, default none
            prngSeed:     Random number generator seed, default 456

        Example environment instances:
            cm=Cat_and_Mouse(catLocs=[[1,1],[2,3]])
            cm.render() #render to screen

            cm=Cat_and_Mouse(rows=5,columns=5,mouseInitLoc=[4,4], stickyLocs=[[2,3]],
                            slipperyLocs=[[1,1],[2,2],[3,3]],cheeseLocs=[[0,4]],catLocs=[[0,0]])
            cm.render(outfile='cmsim.png') #render to .png file
        """
        # env config params
        self.rows = rows
        self.columns = columns
        if mouseInitLoc is None:
            self.mouseInitLoc = [0, 0]
        else:
            self.mouseInitLoc = list(mouseInitLoc)
        if cheeseLocs is None:
            self.cheeseLocs = [[rows-1, columns-1]]
        else:
            self.cheeseLocs = list(cheeseLocs)
        self.catLocs = list(catLocs)
        self.stickyLocs = list(stickyLocs)
        self.slipperyLocs = list(slipperyLocs)
        self.prngSeed = prngSeed

        # a few adjustable constants
        self.slipProb = 0.5  # slip probability on slippery tiles
        self.stickProb = 0.5  # stick probability on sticky tiles

        # a bit of simple consistency checking first
        if self.rows < 1 or self.columns < 1:
            raise Exception("Error: rows, columns must be >= 1")
        if self.offOfMap(self.mouseInitLoc):
            raise Exception(
                "Error: mouse initial location out of bounds: {}".format(self.mouseInitLoc))
        for cheese in self.cheeseLocs:
            if cheese in self.catLocs:
                raise Exception(
                    "Error: Cat and cheese locations cannot overlap: {}".format(cheese))
        for cheese in self.cheeseLocs:
            if self.offOfMap(cheese):
                raise Exception(
                    "Error: cheese location out of bounds: {}".format(cheese))
        for cat in self.catLocs:
            if self.offOfMap(cat):
                raise Exception(
                    "Error: cat location out of bounds: {}".format(cat))
        for sticky in self.stickyLocs:
            if sticky in self.slipperyLocs:
                raise Exception(
                    "Error: Sticky and slippery locations cannot overlap: {}".format(sticky))
        for sticky in self.stickyLocs:
            if self.offOfMap(sticky):
                raise Exception(
                    "Error: sticky location out of bounds: {}".format(sticky))
        for slippery in self.slipperyLocs:
            if self.offOfMap(slippery):
                raise Exception(
                    "Error: slippery location out of bounds: {}".format(slippery))

        # computed env params
        self.numStates = rows*columns

        if self.rows == 1:
            self.numActions = 2
            self.actions = {0: 'E', 1: 'W'}
            self.moves = {0: [0, 1], 1: [0, -1]}
        elif self.columns == 1:
            self.numActions = 2
            self.actions = {0: 'N', 1: 'S'}
            self.moves = {0: [-1, 0], 1: [1, 0]}
        else:
            self.numActions = 8
            self.actions = {0: 'E', 1: 'W', 2: 'N',
                            3: 'S', 4: 'SE', 5: 'SW', 6: 'NE', 7: 'NW'}
            self.moves = {0: [0, 1], 1: [0, -1], 2: [-1, 0], 3: [1,
                                                                 0], 4: [1, 1], 5: [1, -1], 6: [-1, 1], 7: [-1, -1]}

        # init prng (random number gen)
        self.prng = Random()
        self.prng.seed(self.prngSeed)

        # initialize mouse robot state (at mouseInitLoc)
        self.mouseLoc = list(self.mouseInitLoc)

        # list of rendered filenames (e.g., .png, .jpg)
        self.renderedFiles = []

    def offOfMap(self, coord):
        """
        Returns True if coordinate is off of grid-world map (out of bounds), else False
        """
        if coord[0] >= self.rows or coord[0] < 0:
            return True
        if coord[1] >= self.columns or coord[1] < 0:
            return True
        return False

    def currentState(self):
        """
        Returns current mouse state as enumerated state (integer index)
        """
        return self.mouseLoc[0]*self.columns+self.mouseLoc[1]

    def currentStateXY(self):
        """
        Returns current mouse state in x,y coord form
        """
        return self.mouseLoc

    def initState(self, newState=None):
        """
        Reset mouse init state. If newState is None, return to mouseInitLoc (set in ctor)

        If newState is type int, assume it's an enumerated state value, otherwise
        assume the state is specified using x,y coordinates [int, int]
        """
        # if newState==None, return to mouseInitLoc
        if newState == None:
            self.mouseLoc = list(self.mouseInitLoc)
        elif type(newState) == int:
            tmp = [newState//self.columns, newState % self.columns]
            if self.offOfMap(tmp):
                raise Exception(
                    "Error: initState: mouse location out of bounds: {}".format(tmp))
            self.mouseLoc = tmp
        else:
            if self.offOfMap(newState):
                raise Exception(
                    "Error: initState: mouse location out of bounds: {}".format(newState))
            self.mouseLoc = list(newState)

    def reset(self, initLoc=None, resetPrng=False):
        """
        Reset entire environment: 1.Mouse state back to mouseInitLoc, unless initLoc is specified
                                     (mouse initLoc uses same format as initState() method (see above))
                                  2.Only if requested, also reset random number generator using seed,
                                  3.Rendered file list reset to []
        """
        if initLoc is None:
            self.mouseLoc = list(self.mouseInitLoc)
        else:
            self.initState(initLoc)
        if resetPrng:
            self.prng.seed(self.prngSeed)
        self.renderedFiles = []

    def step(self, action):
        """
        Increment mouse state using specified action

        Returns: (newState, reward, gameOverFlag)
        """
        # if starting location is at cheese or cat location (infinite absorbing final states),
        # game is already over. Return reward 0, game over
        if (self.mouseLoc in self.cheeseLocs) or (self.mouseLoc in self.catLocs):
            return self.currentState(), 0.0, True

        newMouseLoc = [0, 0]
        # move to next state based on current action
        #

        slipped = False
        slipStart = None
        # if starting location is sticky tile, need to determine whether
        # we get stuck or not
        if (self.mouseLoc in self.stickyLocs) and self.prng.random() < self.stickProb:
            # got stuck on sticky tile (self.stickProb chance of getting stuck)
            newMouseLoc = self.mouseLoc
        else:
            # move to next state (mouse location) based on action
            newMouseLoc[0] = self.mouseLoc[0]+self.moves[action][0]
            newMouseLoc[1] = self.mouseLoc[1]+self.moves[action][1]

            # if new location is slippery tile, need to determine whether
            # we slip forward or not... (self.slipProb chance of slipping forward)
            # May happen multiple times if have adjacent slippery tiles.
            slipped = False
            while (newMouseLoc in self.slipperyLocs) and self.prng.random() < self.slipProb:
                # slipped forward
                slipped = True
                # need to remember this in case we slip off edge of map
                slipStart = list(newMouseLoc)

                # slip forward to new location
                newMouseLoc[0] = newMouseLoc[0]+self.moves[action][0]
                newMouseLoc[1] = newMouseLoc[1]+self.moves[action][1]

        # If new location is off of map, return reward -1, keep current location on map edge,
        # Else, update to new location (result of action)
        if self.offOfMap(newMouseLoc):
            if slipped:
                self.mouseLoc = slipStart
            return self.currentState(), -1.0, False
        else:
            self.mouseLoc = newMouseLoc

        # after checking boundary conditions above, compute and return rewards
        #
        if self.mouseLoc in self.cheeseLocs:
            # if mouse found cheese, return reward 1, game over
            return self.currentState(), 1.0, True
        elif self.mouseLoc in self.catLocs:
            # if mouse encountered cat, return reward -10, game over
            return self.currentState(), -10.0, True
        else:
            # all other cases return reward 0
            return self.currentState(), 0.0, False

    def render(self, outfile=None):
        """
        Render and plot current state of the grid world

        if outfile is None: render to screen
        else: render to file

        if keepFiles is True: keep list of rendered filenames
        """

        # plot current map state
        mapstate = np.zeros((self.rows+1, self.columns+1))
        plt.matshow(mapstate)

        for cheese in self.cheeseLocs:
            plt.text(cheese[1]+0.5, cheese[0]+0.5, 'Cheese', va='center', ha='center',
                     bbox=dict(boxstyle='round', facecolor='gold', edgecolor='0.3'))
        for cat in self.catLocs:
            plt.text(cat[1]+0.5, cat[0]+0.5, 'Cat', va='center', ha='center',
                     bbox=dict(boxstyle='round', facecolor='red', edgecolor='0.3'))
        for sticky in self.stickyLocs:
            plt.text(sticky[1]+0.5, sticky[0]+0.5, 'Stick', va='center', ha='center',
                     bbox=dict(boxstyle='round', facecolor='grey', edgecolor='0.3'))
        for slippery in self.slipperyLocs:
            plt.text(slippery[1]+0.5, slippery[0]+0.5, 'Slip', va='center', ha='center',
                     bbox=dict(boxstyle='round', facecolor='blue', edgecolor='0.3'))

        mouse = plt.text(self.mouseLoc[1]+0.5, self.mouseLoc[0]+0.5, 'Mouse', va='center', ha='center',
                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        ax = plt.gca()
        ax.grid()

        if outfile is None:
            # plot to screen
            plt.show()
        else:
            # plot to file
            plt.savefig(outfile)

            # keep a list of rendered files
            self.renderedFiles.append(outfile)

        # cleanup plots
        plt.cla()
        plt.close('all')

    def rendered2gif(self, outfile, framePeriod=500):
        """
        Convert sequence of static rendered frames to animated gif,
        with animation period of framePeriod (msec)
        """
        if len(self.renderedFiles) == 0:
            # nothing to do!
            print('rendered2gif: empty filelist, nothing to do!')
            return

        # read in the image files
        imageFrames = []
        for imageFile in self.renderedFiles:
            imageFrames.append(Image.open(imageFile))

        # save image sequence to animated gif
        imageFrames[0].save(outfile, format='gif', append_images=imageFrames[1:],
                            save_all=True, duration=framePeriod)

    def policy2gif(self, policy, mouseInitLoc, filebaseName, maxEpisodeLen=None, resetPrng=False):
        """
        From specified initial state, this method will "walk" the mouse through the
        environment using the given policy. Result will be saved to an animated .gif file.

        Input Parameters:
            policy:  Currently this function only supports deterministic policies in 1-D column vector (numStates x 1) format
                     Example: numpy.zeros((simenv.numStates,1),dtype=np.int32)
                              (fill each state entry with an action index (the deterministic policy))
            mouseInitLoc: The initial starting location of the mouse agent.  moustInitLoc format is same
                          as self.initState() method above.
            filebaseName: (string) The prefix used for all intermediate .png files and also the final .gif file.
            maxEpisodeLen: Maximum limit on episode length, in case policy results in infinite sequence (loops, for example)
            resetPrng: Reset the prng using initial seed before performing policy-walk (use this if you want exactly 
                         repeatable results in stochastic environments)

        Returns:
            Total number of steps taken during this policy-walk episode
        """
        # reset the environment
        self.reset(mouseInitLoc, resetPrng)

        # walk the policy and generate animated .gif
        self.render(outfile=filebaseName+'0.png')
        done = False
        i = 0
        while not done:
            action = policy[self.currentState()][0]
            (state, reward, done) = self.step(action)
            i += 1
            self.render(outfile=filebaseName+str(i)+'.png')
            if (maxEpisodeLen is not None) and (i >= maxEpisodeLen):
                done = True
        self.rendered2gif(outfile=filebaseName+'.gif')

        # reset the environment before we exit...
        #  Doing this to cleanup .png filelist & put mouse back at init state
        self.reset(mouseInitLoc, resetPrng)

        # return total number of steps taken during this policy walk
        return i
