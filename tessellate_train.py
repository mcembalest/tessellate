
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import Flatten, Dense, Dropout
import time
from tessellate_game import GameState, RED, BLUE, EMPTY, OUT_OF_PLAY



def create_model():
	model = models.Sequential()
	
	model.add(Flatten())
	model.add(Dense(150, activation='relu'))
	#model.add(Dropout(0.1))
	model.add(Dense(150, activation='relu'))
	#model.add(Dropout(0.1))
	model.add(Dense(150, activation='relu'))
	#model.add(Dropout(0.1))
	model.add(Dense(150, activation='relu'))
	# #model.add(Dropout(0.1))
	# model.add(Dense(50, activation='relu'))
	# model.add(Dense(50, activation='relu'))
	# model.add(Dense(50, activation='relu'))
	# model.add(Dense(50, activation='relu'))
	
	model.add(Dense(100))
	
	return model
   
def compute_loss(logits, actions, rewards): 
	#print('logits', logits)
	#print('actions', actions)
	#print('rewards', rewards)
	neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
	#print('neg log prob', neg_logprob)
	loss = tf.reduce_mean(neg_logprob * rewards)
	#print('loss', loss)
	return loss
  
def train_step(model, optimizer, observations, actions, rewards):
	with tf.GradientTape() as tape:
	  # Forward propagate through the agent network
		
		logits = model(observations)
		loss = compute_loss(logits, actions, rewards)
		#print(loss)
		grads = tape.gradient(loss, model.trainable_variables)
		
		optimizer.apply_gradients(zip(grads, model.trainable_variables))

def get_action(game_state, model, epsilon):
	#determine whether model action or random action based on epsilon
	#print('~~~~~~~~~~~~~~~~~~~~~~~~')

	

	#print(time.time() - t1)
	#t2 = time.time()
	t = time.time()
	logits = model.predict(game_state.board.reshape(1,10,10,1))
	print(time.time() - t)
	#print(time.time() - t2)
	#t3 = time.time()
	prob_weights = tf.nn.softmax(logits).numpy().flatten()

	playable_spots = np.array([game_state.legal_moves[(i//10, i%10)] for i in range(100)])
	#print('~~~~~~~~~~~~~~~~~~~~~~~~')
	#print(game_state.board)
	#print(playable_spots)
	#print('checking')
	# for row in range(10):
	# 	for col in range(10):
	# 		if not ((game_state.board[row][col] == 2 and playable_spots[10*row + col] == 1) or (game_state.board[row][col] in [0, 1, 3] and playable_spots[10*row + col] == 0)):
	# 			raise ValueError('mismatch at i = ', i, 'j = ', j)
	#print('~~~~~~~~~~~~~~~~~~~~~~~~')
	#print(prob_weights.shape)
	#print(time.time() - t3)
	#print('~~~~~~~~~~~~~~~~~~~~~~~~')

	act = np.random.choice(['model','random'], p=[1-epsilon, epsilon])
	if act == 'model':
		#print('$$$$$')
		#print(prob_weights * playable_spots)
		max_ = max(prob_weights * playable_spots)
		action = list(prob_weights).index(max_)
		assert max_ > 0
		#print(action)
		#print('$$$$$')
	if act == 'random':
		l = np.random.random(100)
		action = list(l).index(max(l * playable_spots))
		
	return action

def check_if_action_valid(obs,action):
	return obs.flatten()[action] == EMPTY

class Memory:
	def __init__(self): 
		self.clear()

	# Resets/restarts the memory buffer
	def clear(self): 
		self.observations = []
		self.actions = []
		self.rewards = []
		self.info = []
		
	def add_to_memory(self, new_observation, new_action, new_reward): 
		self.observations.append(new_observation)
		self.actions.append(new_action)
		self.rewards.append(float(new_reward))


def train_models(num_training_iterations, from_scratch = False):
	#train player 1 against random agent
	tf.keras.backend.set_floatx('float64')
	LEARNING_RATE = 1e-4
	red_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
	blue_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

	red_memory = Memory()
	blue_memory = Memory()
	epsilon = 0.2

	if from_scratch:
		player_1_model = create_model()
		player_2_model = create_model()

	else:
		player_1_model = keras.models.load_model('p1model', compile = False)
		player_2_model = keras.models.load_model('p2model', compile = False)

	for i_episode in range(num_training_iterations):

		print('episode', i_episode)

		game_state = GameState()

		red_memory.clear()
		blue_memory.clear()
		epsilon = epsilon * .99

		for t in range(1, 51):
			#t0 = time.time()

			if t % 2 == 1:
				player = player_1_model
			else:
				player = player_2_model

			#t4 = time.time()
			action = get_action(game_state, player, epsilon)
			#print('t4', time.time() - t4)
			row, col = int(action)//10, int(action)%10 # action is an integer from 0 to 99
			game_state.new_move(row, col)

			observation = game_state.board.astype(float).reshape(10, 10, 1)

			# weight the rewards later in the game to be exponentially more important for the learning algorithm
			# early weights are close to zero, later weights are close to one
			score_time_weight = 10*np.exp(0.1*(t - 50))
			islands_time_weight = 5*np.exp(0.1*t)

			if t % 2 == 1:
				reward = np.log(game_state.score[RED] / game_state.score[BLUE]) * score_time_weight + game_state.num_islands[RED] * islands_time_weight
				red_memory.add_to_memory(observation, action, reward)

			else:
				reward = np.log(game_state.score[BLUE] / game_state.score[RED]) * score_time_weight + game_state.num_islands[BLUE] * islands_time_weight
				blue_memory.add_to_memory(observation, action, reward)	

			if t % 10 == 0:


				#print('t0', time.time() - t0)

				#t1 = time.time()
				
				#print('doing a red train step')
				train_step(player_1_model, red_optimizer,
						 observations=np.array(red_memory.observations),
						 actions=np.array(red_memory.actions),
						 rewards = red_memory.rewards)

				#print('t1', time.time() - t1)

				#t2 = time.time()

				#print('doing a blue train step')
				train_step(player_2_model, blue_optimizer,
						 observations=np.array(blue_memory.observations),
						 actions=np.array(blue_memory.actions),
						 rewards = blue_memory.rewards)

				#print('t2', time.time() - t2)

	player_1_model.save('p1model')
	player_2_model.save('p2model')


num_training_iterations = 100
print('starting training')
train_models(num_training_iterations)

