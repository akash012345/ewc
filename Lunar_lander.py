import tensorflow as tf
import numpy as np
import gym
import os
import random
from collections import deque



# variable initialization functions
def weight_variable(shape, name):

    initial = tf.truncated_normal(shape, stddev=0.1, name= name)
    return tf.Variable(initial)

def bias_variable(shape, name):
    initial = tf.truncated_normal(shape, mean=0, stddev=0.1, name=name)
    return tf.Variable(initial)

class Agent():
    def __init__(self, state_size, action_size):
        self.save_model_path   = "./my-model.ckpt123"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.00005
        self.gamma              = 0.95
        self.exploration_rate   = 0.001
        self.exploration_min    = 0.001
        self.exploration_decay  = 0.995
        self.n_hidden1          = 50
        self.n_hidden2          = 40
        self.lam                = 580
        self.sess               = tf.InteractiveSession()


        self.x = tf.placeholder(tf.float32, [None, state_size], name='features')
        self.target = tf.placeholder(tf.float32, [None, action_size], name='output')
        
        w1 = weight_variable([state_size, self.n_hidden1], 'weight1')
        b1 = bias_variable([self.n_hidden1], 'bias1')

        h1 = tf.nn.relu(tf.matmul(self.x, w1) + b1, name = "hidden1")

        w2 = weight_variable([self.n_hidden1, self.n_hidden2], "weight2")
        b2 = bias_variable([self.n_hidden2], "bias2")

        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2, name = "hidden2")

        w3 = weight_variable([self.n_hidden2, self.action_size], "weight3")
        b3 = bias_variable([self.action_size], "bias3")

        self.y = tf.matmul(h2, w3) + b3

        self.var_list = [w1, b1, w2, b2, w3, b3]

        self.loss = tf.losses.mean_squared_error(self.target, self.y)
        
        self.train_step = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
        
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        self.saver.restore(self.sess,self.save_model_path)

    def predict(self, state):
        return self.sess.run(self.y, feed_dict={self.x : state})

    def train(self, state, target, lam):
        # self.lam = lam
        self.sess.run(self.train_step, feed_dict={self.x : state, self.target : target})

    def save_model(self,filename):
        print("model saved "+filename)
        self.saver.save(self.sess, self.save_model_path+filename)


    def act(self, state, action_size):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(action_size)
        act_values = self.predict(state)
        return np.argmax(act_values[0][0:action_size])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size, lam = 0):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.predict(next_state)[0])
            target_f = self.predict(state)
            target_f[0][action] = target
            self.train(state, target_f, lam)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


class Game:
    def __init__(self):
        self.sample_batch_size = 100
        self.episodes          = 10000
        #enviornment 2 runs first
        self.env2              = gym.make('LunarLander-v2')
        self.env1              = gym.make('CartPole-v0')
        self.env1_input        = self.env1.observation_space.shape[0]
        self.env2_input        = self.env2.observation_space.shape[0]
        self.input_size        = max(self.env2_input,self.env1_input)
        self.env1_output       = self.env1.action_space.n
        self.env2_output       = self.env2.action_space.n
        self.output_size       = max(self.env1_output,self.env2_output)
        self.agent             = Agent(self.input_size, self.output_size)
        self.render_activate    = True


    def run(self):
        try:

            for index_episode in range(self.episodes):
                state = self.env2.reset()
                N=self.input_size-self.env2.observation_space.shape[0]
                state = np.pad(state, (0, N), 'constant')
                state = np.reshape(state, [1, self.input_size])

                done = False
                index = 0
                rew = 0
                while not done:
                    if self.render_activate:
                        self.env2.render()

                    action = self.agent.act(state, self.env2.action_space.n)

                    next_state, reward, done, _ = self.env2.step(action)
                    next_state = np.pad(next_state, (0, N), 'constant')
                    next_state = np.reshape(next_state, [1, self.input_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                    rew += reward
                print("Episode {}# Score: {}".format(index_episode, rew))
               # self.agent.replay(self.sample_batch_size, 0)      

        finally:
            x=int(input("do u want to save the model (press 1 to save) : "))
            if x==1:
                name=input("Enter the file no:")
                self.agent.save_model(name)

if __name__ == "__main__":
    game = Game()
    game.run()
