import gym, numpy

env = gym.make('CartPole-v0')

def go():
    for _ in range(1000):
        env.render()
        observation, reward, done, info = \
            env.step(env.action_space.sample()) # Random action
        print(observation, reward)
        if done:
            break

while True:
    observation = env.reset()
    print(observation)
    go()
