import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    n_games = 100

    for i in range(n_games):
        obs, _ = env.reset()
        score = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs_, reward, done, *_ = env.step(action)
            score += reward
        print("episode ", i, "score %1.f" % score)
