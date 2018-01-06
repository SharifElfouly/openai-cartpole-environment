import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from statistics import mean, median
from collections import Counter

LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000


def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break


def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = 1
                else:
                    output = 0

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))
    print('Number of accepted games', len(accepted_scores))

    return training_data


def create_model(input_nodes, output_nodes):
    model = Sequential()

    # hidden layer
    model.add(Dense(units=500, activation='sigmoid', input_dim=input_nodes))

    # output layer
    model.add(Dense(units=output_nodes, activation='sigmoid'))

    return model


def train_model(model, x, y):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(np.array(x), np.array(y), epochs=3, batch_size=128)

    return model


def create_one_hot(y):
    labels = []
    for label in y:
        if label == 0:
            labels.append([1, 0])
        else:
            labels.append([0, 1])
    return labels


def play(model):
    scores = []
    for episode in range(100):
        env.reset()
        action = env.action_space.sample()
        score = 0
        for t in range(goal_steps):
            env.render()
            observation, reward, done, info = env.step(action)
            observation = np.array(observation).reshape(1, 4)
            action = np.argmax(model.predict(observation))
            score += reward
            if done:
                break
        scores.append(score)
    print('Average score: {}'.format(sum(scores)/len(scores)))


model = create_model(4, 2)
data = initial_population()
x = np.array([i[0] for i in data])
y = np.array([i[1] for i in data])
labels = create_one_hot(y)
model = train_model(model, x, labels)
play(model)





