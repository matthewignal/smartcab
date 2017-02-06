import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

n_trials = 200  # Sets number of trials

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env,
        # state = None, next_waypoint = None,
        # and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env,self)  # simple route planner to
        #  get next waypoint
        # TODO: Initialize any additional variables here
        self.Q_table = {}
        self.alpha = 0.2  # Tune alpha for use in Bellman Equation
        self.gamma = 0.2  # Tune gamma for use in Bellman Equation
        self.failures = 0
        self.illegal_actions = 0
        self.trial = 0
        self.reward = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.trial += 1
        self.reward = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner,
        #  also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        if self.trial > (float(n_trials / 2)):
            if deadline == 0:  # add up failures for performance metric
                self.failures += 1

        epsilon = float(1) / self.trial  # epsilon decays over time so
        # it is more exploratory at the beginning and is less so later on
        print "Epsilon:", epsilon

        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'],
                      inputs['right'], self.next_waypoint)
        print "State:", self.state

        encountered_states = set()
        encountered_states.add(self.state)

        # TODO: Select action according to your policy
        if self.state in encountered_states:
            if random.random() >= epsilon or self.trial > (float(n_trials) / 2):
                # choose best action if random number is greater than
                # epsilon, unless after trial 100
                options = {action: self.Q_table.get((self.state, action), 0) for
                           action in self.env.valid_actions}  # get the
                # options for each action for the given state from the
                # Q_table or impute 0
                print "Possible Actions:", options
                max_action = [action for action in self.env.valid_actions if
                              options[action] == max(options.values())]
                print "Best Action(s):", max_action
                # At start, there may be several options with equal scores.
                # Random option is then selected.
                action = max_action[0]  # If a tie, an action of 'None' will get
                # the first choice as we are prioritizing safety
            else:  # choose random option
                action = random.choice(self.env.valid_actions)
                print "Random Option Chosen!"
        else:
            action = random.choice(self.env.valid_actions)
            print "Random Option Chosen!"

        # Execute action and get reward
        reward = self.env.act(self, action)
        if self.trial >= float(n_trials) / 2:
            if reward == -1.0 or reward == 11.0:
                # illegal moves can return only those scores
                self.illegal_actions += 1

        # TODO: Learn policy based on state, action, reward
        # Bellman Equation: Q(s, a) = Q(s, a) + alpha(reward + gamma * maxQ(
        # s+1, a) - Q(s, a))

        # Learning Policy
        old_value = self.Q_table.get((self.state, action), 0)
        next_state = self.env.sense(self)
        next_waypoint = self.planner.next_waypoint()
        state_prime = (next_state['light'], next_state['oncoming'], next_state[
            'right'], next_state['left'], next_waypoint)
        utility_of_next_state = self.Q_table.get((state_prime, action), 0)

        utility_of_state = reward + self.gamma * utility_of_next_state - \
                           old_value

        # Q-table Gets Updated
        self.Q_table[(self.state, action)] = self.Q_table.get((self.state,
                                                               action),
                                                              0) + self.alpha\
                                            * utility_of_state


        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {" \
              "}, reward = {}".format(deadline, inputs, action, reward)
        print

    def performance(self):
        print "Destination Reached Rate = {}".format(float((n_trials / 2) -
                                                           self.failures) /
                                                    (n_trials / 2))
        print "Illegal Moves per Trial = {}".format(float(
            self.illegal_actions) / (n_trials / 2))

def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow
    # longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (
    # uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=n_trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C
    # on the command-line
    a.performance()

if __name__ == '__main__':
    run()