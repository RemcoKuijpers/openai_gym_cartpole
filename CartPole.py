import gym
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
import time

'''
Observation CartPole: [position of cart, velocity of cart, angle of pole, rotation rate of pole]
0   Cart Position           -2.4/2.4
1   Cart Velocity           -Inf/Inf
2   Pole Angle              -41.8/41.8 degree
3   Pole Velocity at tip    -Inf/Inf

Action CartPole: [direction]
direction = 0 --> Force cart to left
direction = 1 --> Force cart to right
'''

class NN():
	def __init__(self, env):
		#self.create_model()
		self.use_best_model()
		#self.gather_data(env)
		#self.gather_data_best_model(env)
		#self.train_model()

	def create_model(self):
		self.model = Sequential()
		self.model.add(Dense(128, input_shape=(4,), activation="relu"))
		self.model.add(Dropout(0.6))

		self.model.add(Dense(256, activation="relu"))
		self.model.add(Dropout(0.6))

		self.model.add(Dense(512, activation="relu"))
		self.model.add(Dropout(0.6))

		self.model.add(Dense(256, activation="relu"))
		self.model.add(Dropout(0.6))

		self.model.add(Dense(128, activation="relu"))
		self.model.add(Dropout(0.6))
		self.model.add(Dense(2, activation="softmax"))

		self.model.compile(
			loss="categorical_crossentropy",
			optimizer="adam",
			metrics=["accuracy"])

	def gather_data(self, env):
		num_trials = 20000
		min_score = 55
		sim_steps = 500
		self.trainingX, self.trainingY = [], []

		scores = []
		for _ in range(num_trials):
			observation = env.reset()
			score = 0
			training_sampleX, training_sampleY = [], []
			for step in range(sim_steps):
				action = np.random.randint(0, 2)
				one_hot_action = np.zeros(2)
				one_hot_action[action] = 1
				training_sampleX.append(observation)
				training_sampleY.append(one_hot_action)

				observation, reward, done, _ = env.step(action)
				score += reward
				if done:
					break
			if score > min_score:
				scores.append(score)
				self.trainingX += training_sampleX
				self.trainingY += training_sampleY

		self.trainingX, self.trainingY = np.array(self.trainingX), np.array(self.trainingY)

	def train_model(self):
		self.model.fit(self.trainingX, self.trainingY, epochs=2)

	def use_best_model(self):
		self.model = load_model("keras_model")

	def gather_data_best_model(self, env):
		num_trials = 20
		min_score = 400
		sim_steps = 500
		self.trainingX, self.trainingY = [], []

		scores = []
		for _ in range(num_trials):
			observation = env.reset()
			score = 0
			training_sampleX, training_sampleY = [], []
			for step in range(sim_steps):
				action = np.argmax(self.model.predict(observation.reshape(1,4)))
				one_hot_action = np.zeros(2)
				one_hot_action[action] = 1
				training_sampleX.append(observation)
				training_sampleY.append(one_hot_action)

				observation, reward, done, _ = env.step(action)
				score += reward
				if done:
					break
			if score > min_score:
				scores.append(score)
				self.trainingX += training_sampleX
				self.trainingY += training_sampleY

		self.trainingX, self.trainingY = np.array(self.trainingX), np.array(self.trainingY)

def main():
	env = gym.make("CartPole-v1")
	network = NN(env)
	
	scores = []
	num_trials = 10
	sim_steps = 500
	for _ in range(num_trials):
		observation = env.reset()
		score = 0
		for step in range(sim_steps):
			env.render()
			action = np.argmax(network.model.predict(observation.reshape(1,4)))
			observation, reward, done, _ = env.step(action)
			score += reward
			if done:
				break
			time.sleep(0.01)
		scores.append(score)

	average_score = np.mean(scores)
	print("Average score: {}".format(average_score))

	if average_score > 500:
		network.model.save('keras_model')
		print("Model saved, achieved highest score")

if __name__ == "__main__":
	main()
