import math
import random
import numpy as np 
import theano
from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, batch_norm, DropoutLayer, Deconv2DLayer, BatchNormLayer, NonlinearityLayer, ElemwiseSumLayer, ConcatLayer, FlattenLayer, Pool2DLayer, Upscale2DLayer
from lasagne.nonlinearities import sigmoid, LeakyRectify, sigmoid, tanh, softmax, elu
from lasagne.nonlinearities import rectify as relu
from lasagne.init import Normal, HeNormal
from lasagne.layers import get_output,get_all_param_values,get_all_params
from lasagne.updates import adam
from theano import tensor as T

class UCB1():

    def __init__(self, counts = None, 
                       values = None,
                       n_arms = None,
                       reward_interpolation = None):
        if n_arms is not None:
            self.initialize(n_arms)
        else:
            self.counts = counts # Count represent counts of pulls for each arm. For multiple arms, this will be a list of counts.
            self.values = values # Value represent average reward for specific arm. For multiple arms, this will be a list of values.
        #how work in the time
        if reward_interpolation is None:
            self.interpolation = lambda value,reward,n : ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        else:
            self.interpolation = reward_interpolation
    
    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return
    
    def get_prefered_arm(self):
        return self.values.index(max(self.values))
    
    def get_arms_values(self):
        return self.values

    # UCB arm selection based on max of UCB reward of each arm
    def select_arm(self):
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm
    
        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)

        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus

        return ucb_values.index(max(ucb_values))
    
    # Choose to update chosen arm and reward
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        # Update average/mean value/reward for chosen arm
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = self.interpolation(value, reward, n)


class QTable():

    def __init__(self, nactions):
        self.table = {}
        self.nactions = nactions

    def __getitem__(self, state):
        if state in self.table:
            return self.table[state]
        else: 
            self.table[state] = [0.0 for a in range(self.nactions)]
            return self.table[state]
    
    def action(self,state):
        actions = self.__getitem__(state)
        if sum(actions) == 0.0:
            return random.randint(0, self.nactions-1)
        else:
            return np.argmax(actions)

class QFunction():

    def __init__(self, nactions, discount=1.0, learning_rate=0.1):
        self.nactions = nactions
        self.discount = discount
        self.learning_rate = learning_rate
        self.table = QTable(nactions)

    #TIME-DIFFERENCE SUPERVISED (or SARSA)
    def update_sarsa(self, state, action, next_state, next_action, reward):
        qvalue = self.table[state][action]
        next_qvalue = self.table[next_state][next_action]
        new_qvalue = qvalue + self.learning_rate * ( reward + self.discount * next_qvalue - qvalue)
        self.table[state][action] = new_qvalue

    def update(self, state, action, next_state, reward):
        qvalue = self.table[state][action]
        next_qvalue = np.max(self.table[next_state])
        new_qvalue = qvalue + self.learning_rate * ( reward + self.discount * next_qvalue - qvalue)
        self.table[state][action] = new_qvalue

    def action(self, state):
        return self.table.action(state)

class DeepQFunction():

    @staticmethod
    def build_network(size, actions, nd = 30):
        InputNoise = InputLayer(shape=(None, size))
        gnet0 = DenseLayer(InputNoise, nd, W=Normal(0.02), nonlinearity=relu)
        gnet1 = DenseLayer(gnet0, nd, W=Normal(0.02), nonlinearity=relu)
        gnet2 = DenseLayer(gnet1, nd, W=Normal(0.02), nonlinearity=relu)
        gnetout = DenseLayer(gnet2, actions, W=Normal(0.02), nonlinearity=softmax)
        return gnetout 

    def get_loss_function(self):
        #args
        self.states = T.matrix('state')
        self.rewards = T.col('reward')
        self.next_states = T.matrix('next_state')        
        self.actions = T.icol('action')        
        #q(s,a)
        actionmask = T.eq(T.arange(self.nactions).reshape((1, -1)), self.actions.reshape((-1, 1)))
        actionmask = actionmask.astype(theano.config.floatX)
        q_action = (get_output(self.network, self.states) * actionmask).sum(axis=1).reshape((-1, 1))
        #max(q(s_next))
        next_q_action = T.max(get_output(self.network, self.next_states), axis=1, keepdims=True)
        #loss = target - qvalue
        loss = (self.rewards + self.discount * next_q_action - q_action)
        #mse
        mse = 0.5 * loss ** 2
        #sum loss
        return T.sum(mse)

    def get_loss_sarsa_function(self):
        #args
        self.states = T.matrix('state')
        self.actions = T.icol('action')
        self.next_states = T.matrix('next_state')  
        self.next_actions = T.icol('next_action')
        self.rewards = T.col('reward')
        #q(s,a)
        actionmask = T.eq(T.arange(self.nactions).reshape((1, -1)),
                          self.actions.reshape((-1, 1))).astype(theano.config.floatX)
        q_action = (get_output(self.network, self.states) * actionmask).sum(axis=1).reshape((-1, 1))
        #q(s_next,a_next)
        next_actionmask = T.eq(T.arange(self.nactions).reshape((1, -1)),
                               self.next_actions.reshape((-1, 1))).astype(theano.config.floatX)
        next_q_action = (get_output(self.network, self.next_states) * next_actionmask).sum(axis=1).reshape((-1, 1))
        #loss = target - qvalue
        loss = (self.rewards + self.discount * next_q_action - q_action)
        #mse
        mse = 0.5 * loss ** 2
        #sum loss
        return T.sum(mse)

    def build_training_function(self):
        self._loss = self.get_loss_function()
        self._params = get_all_params(self.network, trainable=True)
        self._updates_net = adam(self._loss, self._params, 
                                 learning_rate=self.learning_rate,
                                 beta1=0.)
        return theano.function([self.states, self.actions, 
                                self.next_states, 
                                self.rewards], 
                                self._loss, 
                                updates=self._updates_net)

    def build_training_sarsa_function(self):
        self._loss = self.get_loss_sarsa_function()
        self._params = get_all_params(self.network, trainable=True)
        self._updates_net = adam(self._loss, self._params, 
                                 learning_rate=self.learning_rate,
                                 beta1=0.)
        return theano.function([self.states, self.actions, 
                                self.next_states, self.next_actions, 
                                self.rewards], 
                                self._loss, 
                                updates=self._updates_net)


    def __init__(self, size, nactions, discount=1.0, learning_rate=0.1, use_sarsa=False):
        self.nactions = nactions
        self.discount = discount
        self.learning_rate = learning_rate
        self.input = T.matrix("input")
        self.size = size
        self.network = DeepQFunction.build_network(size, nactions)
        self.qfuction = theano.function([self.input], get_output(self.network, self.input, deterministic=True))
        if use_sarsa:
            self.qtrain = self.build_training_sarsa_function()
        else:
            self.qtrain = self.build_training_function()

    def update(self, states, actions, next_states, rewards):        
        states        = np.array(states).astype(theano.config.floatX).reshape((len(states), self.size))           #as f32 matrix
        actions       = np.array(actions).astype(np.int32).reshape((-1,1))                                        #as column
        next_states   = np.array(next_states).astype(theano.config.floatX).reshape((len(next_states), self.size)) #as f32 matrix
        rewards       = np.array(rewards).astype(theano.config.floatX).reshape((-1,1))                            #as column
        self.qtrain(states, actions, next_states, rewards)

    def update_sarsa(self, states, actions, next_states, next_actions, rewards):        
        states        = np.array(states).astype(theano.config.floatX).reshape((len(states), self.size))           #as f32 matrix
        actions       = np.array(actions).astype(np.int32).reshape((-1,1))                                        #as column
        next_states   = np.array(next_states).astype(theano.config.floatX).reshape((len(next_states), self.size)) #as f32 matrix
        next_actions  = np.array(next_actions).astype(np.int32).reshape((-1,1))                                   #as column
        rewards       = np.array(rewards).astype(theano.config.floatX).reshape((-1,1))                            #as column
        self.qtrain(states, actions, next_states, next_actions, rewards)

    def action(self, state):
        state = np.array(state).astype(theano.config.floatX).reshape((1, self.size))  #as f32 matrix
        return np.argmax(self.qfuction(state))
    
    def params(self):
        return get_all_param_values(self.network)
