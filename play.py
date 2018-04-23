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
        self.state_size         = state_size
        self.action_size        = action_size
        self.n_hidden1 			= 50
        self.n_hidden2			= 40
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
        
        self.sess.run(tf.global_variables_initializer())

        self.saver=tf.train.Saver()
        self.saver.restore(self.sess,'./my-model.ckpt123')

    def predict(self, state):
        return self.sess.run(self.y, feed_dict={self.x : state})

    def act(self, state, action_size):
        act_values = self.predict(state)
        return np.argmax(act_values[0][0:action_size])

class Game:
    def __init__(self):
        self.episodes          = 3
        self.env1              = gym.make('LunarLander-v2')
        self.env2              = gym.make('CartPole-v0')
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
            while True:
                reward1=0
                for index_episode in range(self.episodes):
                    state = self.env1.reset()
                    N=self.input_size-self.env1.observation_space.shape[0]
                    state = np.pad(state, (0, N), 'constant')
                    state = np.reshape(state, [1, self.input_size])

                    done = False
                    index = 0
                    rew = 0
                    while not done:
                        if self.render_activate:
                            self.env1.render()

                        action = self.agent.act(state, self.env1.action_space.n)

                        next_state, reward, done, _ = self.env1.step(action)
                        next_state = np.pad(next_state, (0, N), 'constant')
                        next_state = np.reshape(next_state, [1, self.input_size])
                        state = next_state
                        index += 1
                        rew += reward
                    reward1 +=rew
                    print("Episode {}# Score: {}".format(index_episode, rew))
                self.avgreward1 = reward1/self.episodes

                print(self.avgreward1)


                print("running the second game")
                reward2=0
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
                        state = next_state
                        index += 1
                        rew += reward
                    reward2+=rew
                    print("Episode {}# Score: {}".format(index_episode, rew))
                self.avgreward2 = reward2/self.episodes

                print(self.avgreward2)
        finally:
            print('finished')
            #self.agent.save_model()
            
if __name__ == "__main__":
    game = Game()
    game.run()
