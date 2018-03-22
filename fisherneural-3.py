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
        self.save_model_path   = "saved.cpkt"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.001
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.001
        self.exploration_decay  = 0.995
        self.n_hidden1 			= 100
        self.n_hidden2			= 90
        self.lam				= 30
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

        # self.intialize()

        # self.sess.run(tf.global_variables_initializer())


        # self.saver = tf.train.Saver()

    # def intialize(self):
    #     self.F_accum = []
    #     for v in range(len(self.var_list)):
    #         self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))
    #     self.star_vars = []
    #     for v in range(len(self.var_list)):
    #         self.star_vars.append(self.var_list[v].eval())
    #     ewc_penalty = 0
    #     for v in range(len(self.var_list)):
    #         ewc_penalty += (self.lam/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
    #     self.loss = tf.losses.mean_squared_error(self.target, self.y) + ewc_penalty
    #     self.train_step = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)



    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))


    def predict(self, state):
        return self.sess.run(self.y, feed_dict={self.x : state})

    def train(self, state, target, lam):
        # self.lam = lam
        self.sess.run(self.train_step, feed_dict={self.x : state, self.target : target})
    	
    def update_ewc_penalty(self):
        self.ewc_loss = self.loss
        for v in range(len(self.var_list)):
            self.ewc_loss+= (self.lam/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
        # self.loss = tf.losses.mean_squared_error(self.target, self.y) + ewc_penalty
        self.train_step = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.ewc_loss)
        self.sess.run(tf.global_variables_initializer())

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []
        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())

    def save_model(self):
    	print("model saved")
    	# self.saver.save(self.sess, self.save_model_path)


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

    def compute_fisher(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))
            print(self.var_list[v])
        probs = tf.nn.softmax(self.y)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
        print('a')
        for i in range(sample_batch_size):
            sample_batch = random.sample(self.memory, sample_batch_size)
            for state, action, reward, next_state, done in sample_batch:
                # compute first-order derivatives
                ders = self.sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: state})           
                # square the derivatives and add to total
                for v in range(len(self.F_accum)):
                    self.F_accum[v] += np.square(ders[v])
            # divide totals by number of sampless 
            for v in range(len(self.F_accum)):
                self.F_accum[v] /= sample_batch_size




class CartPole:
    def __init__(self):
        self.sample_batch_size = 100
        self.episodes          = 200
        self.testno			   = 10
        self.fisher_sample_size = 20
        #enviornment 2 runs first
        self.env1              = gym.make('Acrobot-v1')
        self.env2              = gym.make('CartPole-v0')
        self.env1_input        = self.env1.observation_space.shape[0]
        self.env2_input        = self.env2.observation_space.shape[0]
        self.input_size        = max(self.env2_input,self.env1_input)
        self.env1_output       = self.env1.action_space.n
        self.env2_output       = self.env2.action_space.n
        self.output_size       = max(self.env1_output,self.env2_output)
        self.agent             = Agent(self.input_size, self.output_size)
        

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
                self.agent.replay(self.sample_batch_size, 0)
            

            # calculate fisher information
            self.agent.star()
            self.agent.compute_fisher(self.fisher_sample_size)
            self.agent.update_ewc_penalty()  
            self.agent.restore(self.agent.sess)

            print("Testing...")
            self.agent.exploration_rate = 0
            reward1 = 0
            for index_episode in range(self.testno):
                state = self.env2.reset()
                N=self.input_size-self.env2.observation_space.shape[0]
                state = np.pad(np.array(state), (0, N), 'constant')
                state = np.reshape(state, [1, self.input_size])

                done = False
                index = 0
                rew = 0
                while not done:
                    self.env2.render()
                    action = self.agent.act(state, self.env2.action_space.n)
                    next_state, reward, done, _ = self.env2.step(action)
                    next_state = np.pad(np.array(next_state), (0, N), 'constant')
                    next_state = np.reshape(next_state, [1, self.input_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                    rew += reward
                reward1 += rew
                print("Episode {}# Score: {}".format(index_episode, rew))

            self.avgreward1 = reward1/self.testno

            # train second task

            # self.agent.restore()

            print("Train 2...")

            self.agent.exploration_rate = 1
                        
            for index_episode in range(self.episodes):
                state = self.env1.reset()
                N=self.input_size-self.env1.observation_space.shape[0]
                state = np.pad(np.array(state), (0, N), 'constant')
                state = np.reshape(state, [1, self.input_size])
                done = False
                index = 0
                rew = 0
                while not done:
                    self.env1.render()

                    action = self.agent.act(state, self.env1.action_space.n)

                    next_state, reward, done, _ = self.env1.step(action)

                    next_state = np.pad(np.array(next_state), (0, N), 'constant')
                    next_state = np.reshape(next_state, [1, self.input_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                    rew += reward
                print("Episode {}# Score: {}".format(index_episode, rew))
                self.agent.replay(self.sample_batch_size, self.fisher_sample_size)

            # play first task again

            reward1 = 0

            self.agent.exploration_rate = 0

            for index_episode in range(self.testno):
                state = self.env2.reset()
                N=self.input_size-self.env2.observation_space.shape[0]
                state = np.pad(np.array(state), (0, N), 'constant')
                state = np.reshape(state, [1, self.input_size])

                done = False
                index = 0
                rew = 0
                while not done:
                    self.env2.render()
                    action = self.agent.act(state, self.env2.action_space.n)
                    next_state, reward, done, _ = self.env2.step(action)
                    next_state = np.pad(np.array(next_state), (0, N), 'constant')
                    next_state = np.reshape(next_state, [1, self.input_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                    rew += reward
                reward1 += rew
                print("Episode {}# Score: {}".format(index_episode, rew))

            self.avgreward2 = reward1/self.testno

            print(self.avgreward1)
            print(self.avgreward2)
                # self.agent.replay(self.sample_batch_size, True)
        finally:
            self.agent.save_model()

if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.run()
