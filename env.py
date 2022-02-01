import gym
import highway_env
from matplotlib import pyplot as plt

env = gym.make("racetrack-v0")
env.reset()
for _ in range(100):
    action = [0.1,0.1]
    obs, reward, done, info = env.step(action)
    env.render()

plt.imshow(env.render(mode="rgb_array"))
plt.show()