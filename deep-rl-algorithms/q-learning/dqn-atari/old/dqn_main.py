import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env

if __name__ == "__main__":
    env = make_env("PongNoFrameskip-v4", render_mode="human")
    best_score = -np.inf
    load_checkpoint = True
    n_games = 20
    agent = DQNAgent(
        gamma=0.99,
        epsilon=0.1,  # change to 1.0 if !load_checkpoint
        lr=1e-4,
        input_dims=(env.observation_space.shape),
        n_actions=env.action_space.n,
        memory_size=15000,
        eps_min=0.1,
        batch_size=32,
        replace=1000,
        eps_dec=1e-5,
        chkpt_dir="models/",
        algo="DQNAgent",
        env_name="PongNoFrameskip-v4",
    )

    if load_checkpoint:
        agent.load_models()

    fname = f"{agent.algo}_{agent.env_name}_lr{agent.lr}_{n_games}_games"
    figure_file = "plots/" + fname + ".png"

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        score = 0
        obs, _ = env.reset()

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, *_ = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(obs, action, reward, int(done), obs_)
                agent.learn()
            obs = obs_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print(
            "episode ",
            i,
            "score: ",
            score,
            "average score %.1f best score %.1f epsilon %.2f"
            % (avg_score, best_score, agent.epsilon),
            "steps ",
            n_steps,
        )

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    plot_learning_curve(steps_array, scores, eps_history, figure_file)
