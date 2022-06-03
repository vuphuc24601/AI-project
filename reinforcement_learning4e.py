"""Reinforcement Learning (Chapter 21)"""

import random
from collections import defaultdict,namedtuple, deque

from mdp4e import MDP, policy_evaluation
import numpy as np

class Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""
    
    def __init__(self, buffer_size, batch_size, seed=42):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state"])
        random.seed(seed)

    def add_experience(self, states, actions, rewards, next_states):
        """Adds experience(s) into the replay buffer"""
        experience = self.experience(states, actions, rewards, next_states)
        self.memory.append(experience)

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None: batch_size = num_experiences
        else: batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)

class QLearningAgent:

    def __init__(self, mdp, epsilon_greedy=False, Ne=None, Rplus=None, alpha=None, no_of_episodes=10000, threshold=500, B = 30, K = 100, M = 5000):

        self.gamma = mdp.gamma
        self.terminals = mdp.terminals
        self.all_act = mdp.actlist
        self.Ne = Ne  # iteration limit in exploration function
        self.Rplus = Rplus  # large value to assign before iteration limit
        self.epsilon_greedy = epsilon_greedy
        self.threshold = threshold
        self.B = B
        self.K = K
        self.M = M
        self.episode_cnt = 0
        self.threshold_cnt = 0
        self.replay_buffer = Replay_Buffer(M, B)

        self.Q = defaultdict(float)
        self.Nsa = defaultdict(float)
        self.s = None
        self.a = None
        self.r = None
        self.eps = None

        if self.epsilon_greedy:
            self.EPSILON_DELTA = 1 / no_of_episodes

        if alpha:
            self.alpha = alpha
        else:
            self.alpha = lambda n: 1. / (1 + n)  # udacity video

    def f(self, u, n):
        """Exploration function. Returns fixed Rplus until
        agent has visited state, action a Ne number of times.
        Same as ADP agent in book."""
        if n < self.Ne:
            return self.Rplus
        else:
            return u

    def actions_in_state(self, state):
        """Return actions possible in given state.
        Useful for max and argmax."""
        if state in self.terminals:
            return [None]
        else:
            return self.all_act

    def init_new_episode(self):
        self.s = self.r = None
        self.threshold_cnt = 0
        if self.epsilon_greedy:
            self.eps = max(0, 1 - self.EPSILON_DELTA)
        self.episode_cnt += 1

    def __call__(self, percept):
        s1, r1 = self.update_state(percept)
        Q, Nsa, s, a, r = self.Q, self.Nsa, self.s, self.a, self.r
        alpha, gamma, terminals = self.alpha, self.gamma, self.terminals,
        actions_in_state = self.actions_in_state
        
        self.threshold_cnt += 1
        
        if s is not None:
            Nsa[s, a] += 1
            if self.episode_cnt <= self.K:
                Q[s, a] += alpha(Nsa[s, a]) * (r1 + gamma * max(Q[s1, a1] for a1 in actions_in_state(s1)) - Q[s, a])
            self.replay_buffer.add_experience(s, a, r1, s1)
        if self.episode_cnt > self.K:
            for (_s,_a,_r1,_s1) in self.replay_buffer.pick_experiences(min(len(self.replay_buffer), self.B)):
                Nsa[_s, _a] += 1
                Q[_s, _a] += alpha(Nsa[_s, _a]) * (_r1 + gamma * max(Q[_s1, a1] for a1 in actions_in_state(_s1)) - Q[_s, _a])

        if s in terminals:
            self.s = self.a = self.r = None
        if s1 in terminals:
            return None
        else:
            self.s, self.r = s1, r1
            if self.eps != None:
                if self.eps > np.random.uniform(0.0, 1.0):
                    self.a = actions_in_state(s1)[np.random.choice(4)]
                else:
                    self.a = max(actions_in_state(s1), key= lambda a1: Q[s1, a1])
            elif self.Ne != None and self.Rplus != None:
                self.a = max(actions_in_state(s1), key=lambda a1: self.f(Q[s1, a1], Nsa[s1, a1]))

        return self.a

    def update_state(self, percept):
        """To be overridden in most cases. The default case
        assumes the percept to be of type (state, reward)."""
        return percept

    def continue_current_episode(self):
        return self.threshold_cnt < self.threshold


def run_single_trial(agent_program, mdp):
    """Execute trial for given agent_program
    and mdp. mdp should be an instance of subclass
    of mdp.MDP """

    def take_single_action(mdp, s, a):
        """
        Select outcome of taking action a
        in state s. Weighted Sampling.
        """
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for probability_state in mdp.T(s, a):
            probability, state = probability_state
            cumulative_probability += probability
            if x < cumulative_probability:
                break
        return state
    agent_program.init_new_episode()
    current_state = mdp.init
    while agent_program.continue_current_episode():
        current_reward = mdp.R(current_state)
        percept = (current_state, current_reward)
        next_action = agent_program(percept)
        if next_action is None:
            break
        current_state = take_single_action(mdp, current_state, next_action)
