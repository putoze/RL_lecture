from mario import *


#### #### #### #### ####
#### Let's Training ####
#### #### #### #### ####

use_cuda = torch.cuda.is_available()  # trying to compute faster
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("chckp") / \
    datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
mario = Mario(state_dim=(4, 84, 84),
              action_dim=env.action_space.n, save_dir=save_dir)


logger = MetricLogger(save_dir)

episodes = 10000000  # last-time 10 thousand
for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:

        # Run agent on the state
        action = mario.act(state)

        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)

        # Remember
        mario.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = mario.learn()

        # Logging
        # plot the curve of rewards/lengths/avg_loss/avg_q's
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate,
                      step=mario.curr_step)
