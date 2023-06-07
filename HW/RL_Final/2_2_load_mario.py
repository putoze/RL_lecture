

from mario import *


simenv = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
simenv = JoypadSpace(env, [["right"], ["right", "A"]])
simenv.reset()
simenv = SkipFrame(env, skip=4)
simenv = GrayScaleObservation(env)
simenv = ResizeObservation(env, shape=84)
simenv = FrameStack(env, num_stack=4, new_step_api=True)


#simenv = RecordVideo(env,"super_agent")
# IF using RecordVideo to modify the envs, note that only 4 items returned

save_dir = Path("checkpoints") / "after_train"
mario = Mario(state_dim=(4, 84, 84),
              action_dim=simenv.action_space.n, save_dir=save_dir)
# print(mario.net.state_dict())
# input()

# chkpt means checkpoint
# If training more chkpt, choose different to testing the results
checkpoint = torch.load("./mario_net_15.chkpt")
mario.net.load_state_dict(checkpoint['model'])
# print(mario.net.state_dict())
# input()


# Reset the env state
state = env.reset()


while True:
    action = mario.act(state)
    state, reward, terminate, truc, info = env.step(action)
    if terminate:
        state = env.reset()
    env.render()


'''
### For using RecordVideo to Play
# Play the game!
while True:
    ### Run agent on the state
    action = mario.act(state)
    ### Agent performs action
    next_state, reward, done, info = simenv.step(action)
    ### Remember
    mario.cache(state, next_state, action, reward, done)

    ### Next_State
    state = next_state
    # Check if end of game
    if done:
        break
'''
