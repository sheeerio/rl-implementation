from agents import epsilon_greedy_agent, softmax_agent
from bandits import BernoulliStationaryBandit, GaussianStationaryBandit
from utils import plot_armcount, plot_regret
import numpy as np

num_iters = 5000
k = 10
mu = np.array([0.1,0.5,0.7,0.73,0.756,0.789,0.81,0.83,0.855,0.865])
# mu = np.arange(10)*0.1
bernoulli_bandit = BernoulliStationaryBandit(k , mu)

# player initialization
epsilon_greedy_player_bernoulli = epsilon_greedy_agent(bernoulli_bandit, 1, num_iters, 50)
softmax_player_bernoulli = softmax_agent(bernoulli_bandit, 0.1, num_iters)

def play_epsilon_greedy():
    data["bernoulli_bandit"]["epsilon"] = epsilon_greedy_player_bernoulli.play()
    plot_regret(data, num_iters, "bernoulli_bandit", "epsilon")
    plot_armcount(data, "bernoulli_bandit", k)

def play_softmax():
    data["bernoulli_bandit"]["softmax"] = softmax_player_bernoulli.play()
    plot_regret(data, num_iters, "bernoulli_bandit", "softmax")
    plot_armcount(data, "bernoulli_bandit", k)

if __name__ == "__main__" :
    data = {"bernoulli_bandit":{},"gauss_bandit":{}}
    data["bernoulli_bandit"]["epsilon"] = epsilon_greedy_player_bernoulli.play()
    data["bernoulli_bandit"]["softmax"] = softmax_player_bernoulli.play()
    plot_armcount(data, "bernoulli_bandit", k)