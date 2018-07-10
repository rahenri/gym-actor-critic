#!/usr/bin/env python3

import random
import gym
import numpy as np

import time

import tensorflow as tf


def weight_variable(shape, name=None):
    return tf.get_variable(name, shape)


def bias_variable(shape, name=None):
    return tf.get_variable(
        name, shape, initializer=tf.constant_initializer(0))


def make_stack(layers, inputs):

    prev_size = inputs.get_shape().as_list()[1]
    prev_layer = inputs

    weights = []

    for i, layer in enumerate(layers):
        w = weight_variable([prev_size, layer], name='{}/w'.format(i))
        b = bias_variable([layer], name='{}/b'.format(i))

        weights.append(w)

        next_layer = tf.matmul(prev_layer, w) + b
        if i < len(layers) - 1:
            next_layer = tf.nn.relu(next_layer)

        prev_layer = next_layer
        prev_size = layer

    if layers[-1] == 1:
        prev_layer = tf.reshape(prev_layer, [-1])

    reg = sum(tf.reduce_mean(w*w) for w in weights) / len(weights)

    return prev_layer, reg


def make_q(states, actions):
    with tf.variable_scope('q', reuse=tf.AUTO_REUSE):
        i = tf.concat([states, actions], axis=1)
        net, reg = make_stack([256]*1 + [1], i)
        return tf.tanh(net) * 1000, reg


def make_p(states, action_size):
    with tf.variable_scope('p', reuse=tf.AUTO_REUSE):
        s, reg = make_stack([256]*1 + [action_size], states)
        return tf.nn.softmax(s), reg


def get_vars(scope):
    out = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    assert len(out) > 0
    print(out)
    return out


class NNAgent():
    def __init__(self, state_size, action_size, maxr):
        self.learning_rate = 0.0001
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.999
        self.e = 0.001
        self.maxr = maxr
        self.sess = tf.Session()

        self.rewards = tf.placeholder('float', shape=[None], name='rewards')

        self.states = tf.placeholder(
            'float', shape=[None, state_size], name='states')

        self.next_states = tf.placeholder(
            'float', shape=[None, state_size], name='next_states')

        self.actions = tf.placeholder(
            'float', shape=[None, action_size], name='actions')

        self.gamma_ph = tf.placeholder(
            'float', shape=[None], name='gamma')

        self.q, q_reg = make_q(self.states, self.actions)

        self.p_next, p_reg = make_p(self.next_states, action_size)

        self.q_p_next, _ = make_q(self.next_states, self.p_next)

        self.pred_q = tf.stop_gradient(tf.clip_by_value(
            self.q_p_next * self.gamma_ph + self.rewards, -1000, 1000))

        self.q_loss = tf.reduce_mean(
            tf.square(self.q - self.pred_q)) + 0.1 * q_reg

        self.p_next_loss = -tf.reduce_mean(self.q_p_next) + 0.1 * p_reg
        print(self.p_next_loss)

        self.train_q = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.q_loss, var_list=get_vars('q/'))

        self.train_p = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.p_next_loss, var_list=get_vars('p/'))

        self.init = tf.global_variables_initializer()

        self.sess.run(self.init)

        self.history = []

    def reset(self):
        self.sess.run(self.init)

    def _train(self, states, actions, rewards, gamma, next_states):
        feed = {
            self.states: states,
            self.actions: actions,
            self.rewards: rewards,
            self.next_states: next_states,
            self.gamma_ph: gamma,
        }
        result = self.sess.run(
            [self.train_q, self.train_p, self.q_loss, self.p_next_loss], feed_dict=feed)
        return result[2], result[3]

    def remember(self, state, next_state, move, reward):
        self.history.append((state, next_state, move, reward))
        if len(self.history) > 1024*1024:
            self.history.pop(0)

    def move(self, state, randomize=False):
        if randomize and random.random() < self.e:
            return random.randint(0, self.action_size-1), [], 0
        feed = {
            self.next_states: [state],
        }
        out = self.sess.run([self.p_next, self.q_p_next], feed_dict=feed)
        probs = out[0][0]
        score = out[1][0]
        return np.argmax(probs), probs, score

    def replay(self, sample_size, batch_size):
        if not self.history:
            return
        batch = random.sample(self.history, min(
            len(self.history), sample_size))
        states = np.array([state for state, _, _, _ in batch])
        states_next = np.array([
            next_state if next_state is not None else np.zeros(
                self.state_size) for _, next_state, _, _ in batch])

        actions = np.array([move for _, _, move, _ in batch])
        actions_one_hot = np.zeros([len(actions), self.action_size])
        actions_one_hot[np.arange(len(actions)), actions] = 1

        gamma = np.array(
            [self.gamma if ns is not None else 0 for _, ns, _, _ in batch])

        rewards = np.array([reward for _, _, _, reward, in batch])

        out = []
        for i in range(0, len(batch), batch_size):
            end = min(i + batch_size, len(batch))
            out.append(self._train(
                states[i:end], actions_one_hot[i:end], rewards[i:end], gamma[i:end], states_next[i:end]))

        return out


def product(it):
    out = 1
    for v in it:
        out *= v
    return out


def main():

    # env = gym.make('CartPole-v1')
    # env = gym.make('MountainCar-v0')
    # env = gym.make('Acrobot-v1')
    env = gym.make('LunarLander-v2')
    # env = gym.make('FetchReach-v1')

    state_size = product(env.observation_space.shape)
    action_size = env.action_space.n

    agent = NNAgent(state_size, action_size, 100)

    window = []

    last_summary = time.time()
    last_demo = time.time()

    loss_q_window = []
    loss_p_window = []

    window_size = 50

    best_window = -1.0e10
    best_window_episode = window_size

    try:
        for i_episode in range(200000000):
            state = env.reset()
            # state = np.append(state, 0)
            total_reward = 0
            now = time.time()
            demo = (now - last_demo) > 30
            # demo = True
            reward = 0

            for step in range(10000000):
                move, dist, score = agent.move(state, not demo)

                if demo:
                    env.render()
                next_state, reward, done, _ = env.step(move)

                # reward = max(0.1, reward - abs(next_state[0]))
                # discount reward for off center position
                # reward -= abs(next_state[0])

                total_reward += reward

                if done:
                    next_state = None

                agent.remember(state, next_state, move, reward)

                state = next_state

                if done:
                    break

            window.append(total_reward)

            window_mean = np.mean(window)

            if demo:
                last_demo = time.time()

            if (i_episode + 1) % 128 == 0:
                loss_q_window = []
                loss_p_window = []
                for i in range(8):
                    losses = agent.replay(1024*128, 32)
                    for q, p in losses:
                        loss_q_window.append(q)
                        loss_p_window.append(p)
                window_mean = np.mean(window)
                if window_mean > best_window:
                    best_window = window_mean
                    best_window_episode = i_episode + window_size
                print("{}: moving window: {:.02f}, max window: {:.02f}, best window: {:.02f}, training loss: {:.02f} {:.02f}".format(
                    i_episode, window_mean, np.max(window), best_window, np.mean(loss_q_window), np.mean(loss_p_window)))
                window = []

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
