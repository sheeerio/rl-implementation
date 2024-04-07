from agents import epsilon_greedy_agent, softmax_agent, UCB_agent, Median_Elimination_Agent
from bandits import BernoulliStationaryBandit, GaussianStationaryBandit
from utils import plot_armcount, plot_regret
import numpy as np

num_iters = 5000
k = 10
mu = np.array([0.1,0.5,0.7,0.73,0.756,0.789,0.81,0.83,0.855,0.860])
sigma = np.arange(10)*0.01
# mu = np.arange(10)*0.1
bernoulli_bandit = BernoulliStationaryBandit(k , mu)
gauss_bandit = GaussianStationaryBandit(k, mu, sigma)

# player initialization
epsilon_greedy_player_bernoulli = epsilon_greedy_agent(bernoulli_bandit, 1, num_iters, 50)
softmax_player_bernoulli = softmax_agent(bernoulli_bandit, 0.1, num_iters)
ucb_player_bernoulli = UCB_agent(bernoulli_bandit, num_iters)
mea_player_bernoulli = Median_Elimination_Agent(bernoulli_bandit, epsilon=0.1, delta=0.1)

epsilon_greedy_player_gauss = epsilon_greedy_agent(gauss_bandit, 1, num_iters, 50)
softmax_player_gauss = softmax_agent(gauss_bandit, 0.1, num_iters)
ucb_player_gauss = UCB_agent(gauss_bandit, num_iters)
mea_player_gauss = Median_Elimination_Agent(gauss_bandit, epsilon=0.1, delta=0.1)

def play_epsilon_greedy():
    data["bernoulli_bandit"]["epsilon"] = epsilon_greedy_player_bernoulli.play()
    data["gauss_bandit"]["epsilon"] = epsilon_greedy_player_gauss.play()
    plot_regret(data, num_iters, "bernoulli_bandit", "epsilon")
    plot_regret(data, num_iters, "gauss_bandit", "epsilon")
    plot_armcount(data, "bernoulli_bandit", k)
    plot_armcount(data, "gauss_bandit")

def play_softmax():
    data["bernoulli_bandit"]["softmax"] = softmax_player_bernoulli.play()
    data["gauss_bandit"]["softmax"] = softmax_player_gauss.play()
    plot_armcount(data, "bernoulli_bandit", k)
    plot_armcount(data, "gauss_bandit", k)
    plot_regret(data, num_iters, "bernoulli_bandit", "softmax")
    plot_regret(data, num_iters, "gauss_bandit", "softmax")

def play_ucb():
    data["bernoulli_bandit"]["ucb"] = ucb_player_bernoulli.play()
    data["gauss_bandit"]["ucb"] = ucb_player_gauss.play()
    plot_armcount(data, "bernoulli_bandit", k)
    plot_armcount(data, "gauss_bandit", k)
    plot_regret(data, num_iters, "bernoulli_bandit", "ucb")
    plot_regret(data, num_iters, "gauss_bandit", "ucb")

def play_mea():
    data["bernoulli_bandit"]["mea"] = mea_player_bernoulli.play()

if __name__ == "__main__" :
    data = {"bernoulli_bandit":{},"gauss_bandit":{}}
    data["bernoulli_bandit"]["epsilon"] = epsilon_greedy_player_bernoulli.play()
    data["bernoulli_bandit"]["softmax"] = softmax_player_bernoulli.play()
    data["bernoulli_bandit"]["ucb"] = ucb_player_bernoulli.play()
    play_mea()
    # data["gauss_bandit"]["epsilon"] = epsilon_greedy_player_gauss.play()
    # data["gauss_bandit"]["softmax"] = softmax_player_gauss.play()
    # data["gauss_bandit"]["ucb"] = ucb_player_gauss.play()

    plot_armcount(data, "bernoulli_bandit", k)
    # plot_armcount(data, "gauss_bandit", k)
    # plot_regret(data, num_iters, "bernoulli_bandit")