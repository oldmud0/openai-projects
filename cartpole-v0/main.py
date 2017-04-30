# Submitted to AP Computer Science Principles exam
# Based on "Deep Q Learning with Keras and Gym":
#    https://keon.io/deep-q-learning/
#
# This original code can also be found in my personal GitHub repository:
# https://github.com/oldmud0/openai-projects/blob/master/cartpole-v0/main.py
#
# I grant permission for College Board to use this code for sample purposes
# and to remove this comment block if it is necessary.

import gym, numpy, collections, copy
import h5py
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

# Use the CartPole environment from OpenAI Gym
env = gym.make('CartPole-v0')

# Small hack to let the AI run for an "unlimited" amount of time
# per episode
env._max_episode_steps = 4000

class Model:
    def __init__(self):
        # Ring buffer of experiences.
        # Old memories get deleted automatically.
        self.memory = collections.deque(maxlen = 100000)

        self.model = Sequential([
            Dense(20, input_dim = 4, activation = "tanh"),
            Dense(20, activation = "tanh", kernel_initializer = "uniform"),
            Dense(2,  activation = "linear")
        ])
        self.model.compile(loss = "mse",
            optimizer = RMSprop(lr = .0001)
        )

        # Also known as the "discount rate."
        # This allows us to predict future rewards based on current state.
        self.gamma = .9

        # The percent at which the model should take random actions
        # and see what comes out of them.
        self.explore_rate = 1.0
        self.explore_decay = 0.99
        self.explore_min = 0.05

        # The rate at which the actual model learns.
        self.learn_rate = .0001

        # Filenames for model loading/saving
        self.filename = "cartpole-ai.h5"
        self.memory_filename = "cartpole-ai-memory.npy"

    def remember(self, data):
        self.memory.append(data)

    def action(self, curr_state):
        if numpy.random.rand() <= self.explore_rate:
            return env.action_space.sample()
        action_values = self.model.predict(curr_state)
        return numpy.argmax(action_values[0])

    def replay(self, size):
        if size > len(self.memory):
            size = len(self.memory)
            print("Size of requested replay is bigger than the model's current memory! Clamped to", size)

        # Sample random experiences from our memories
        batch = numpy.random.choice(len(self.memory), size, replace = False)

        independent = numpy.zeros((size, 4))
        dependent   = numpy.zeros((size, 2))
        i = 0
        for experience in batch:
            observation, action, reward, next_state, done = self.memory[experience]

            if done:
                # This action resulted in the end of the simulation.
                # Give us the reward!
                target = reward
            else:
                # Otherwise, figure out what it is or will be in the future
                target = reward + self.gamma * numpy.amax(self.model.predict(next_state)[0])

            # Now use the AI model to determine our target
            approx_target = self.model.predict(observation)
            approx_target[0][action] = target
            
            independent[i], dependent[i] = observation, approx_target
            i += 1
        # Tune the model by correlating actions with results (experiences)
        self.model.fit(independent, dependent, batch_size = size, epochs = 1, verbose = False)

        self.explore_rate = max(self.explore_min, self.explore_rate * self.explore_decay)

    def load_model(self):
        try:
            self.model.load_weights(self.filename)

            # Load using .npy format, which in turn invokes Pickle.
            self.memory = collections.deque(numpy.load(self.memory_filename), maxlen = 100000)

            # Load using hdf5storage, but the library seems to be broken.
            #self.memory = collections.deque(
            #    hdf5storage.read(filename = "cartpole-ai-memory.h5"),
            #    maxlen = 100000
            #)

            # Load using h5py, but h5py does not support O-type (generic) numpy objects.
            #with h5py.File('cartpole-ai-memory.h5', 'r') as hf:
            #    self.memory = collections.deque(hf['cartpole-ai-memory'][:], maxlen = 100000)

            print("Loaded", len(self.memory), "memories from", self.memory_filename)

            # Number of memories at which the explore rate should be at its minimum
            memory_threshold = 30000
            min_threshold = self.explore_min / \
                (self.explore_decay * self.explore_rate ** memory_threshold)
            self.explore_rate = max(self.explore_min, \
                min_threshold * self.explore_decay * self.explore_rate ** len(self.memory))
        except (IOError, KeyError) as e:
            print(e)
            print("Model not found. Ignoring.")

    def save_model(self):
        self.model.save_weights(self.filename)

        # Save using .npy format, which in turn invokes Pickle.
        numpy.save(self.memory_filename, self.memory)

        # Save using hdf5storage, but the library seems to be broken.
        #hdf5storage.write(data = np.array(list(self.memory)), filename = "cartpole-ai-memory.h5")

        # Save using h5py, but h5py does not support O-type (generic) numpy objects.
        #with h5py.File('cartpole-ai-memory.h5', 'w') as hf:
        #        hf.create_dataset("cartpole-ai-memory",  data = self.memory)

        print("Saved", len(self.memory), "memories to", self.memory_filename)

episode_num = 1
graphical = True
def run_episode():
    global episode_num, graphical

    observation = env.reset()
    observation = numpy.reshape(observation, [1, 4])

    success = True
    new_memories = 0

    for _ in range(1000):
        new_memories += 1

        if graphical:
            try:
                env.render()
            except AttributeError:
                print("The window was closed.")
                graphical = False

        action = ai.action(observation)
        next_state, reward, done, info = \
            env.step(action)
        next_state = numpy.reshape(next_state, [1, 4])
        if done:
            reward = -10
        ai.remember((observation, action, reward, next_state, done))
        observation = copy.deepcopy(next_state)
        #print(observation, reward)
        if done:
            success = False
            break

    print("Episode:", episode_num, \
        "| Memories:", len(ai.memory), \
        "| New memories:", new_memories, \
        "| Learn rate:", "{:.4f}".format(ai.explore_rate)
    )
    ai.replay(32)
    episode_num += 1

#
# Client code
#

ai = Model()
ai.load_model()
for episode in range(500):
        run_episode()
ai.save_model()