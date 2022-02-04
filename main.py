import gym
import highway_env
import datetime
from pathlib import Path
from metrics import MetricLogger
from agent import DeepCar
from action_coder import ActionCoder

env = gym.make("racetrack-v0")
config = {
       "observation": {
           "type": "GrayscaleObservation",
           "observation_shape": (128, 128),
           "stack_size": 4,
           "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
           "scaling": 1.75,
       },
       "duration": 300,
       "action_reward": -0.3,
       "policy_frequency": 4
   }
env.configure(config)
obs = env.reset()

save_dir = Path('results') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

action_resolution = 0.05
agent = DeepCar(state_dim=(4, 128, 128), action_dim=int((2 / action_resolution)**2))
print('Action Size (Discrete):\t',int((2 / action_resolution)**2))

logger = MetricLogger(save_dir)
episodes = int(1e4)
action_coder = ActionCoder(action_resolution)

for e in range(episodes):
    state = env.reset()
    while True:
        if e % 200 == 0:
            env.render()
        action = agent.act(state)
        action_coded = action_coder.convert(action)
        next_state, reward, done, info = env.step(action_coded)
        agent.cache(state, next_state, action, reward, done)
        q, loss = agent.learn()
        logger.log_step(reward, loss, q)
        state = next_state
        if done:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )
