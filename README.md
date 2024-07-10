# rl-implementation

## Deep Q Networks

Command:
```
usage: main.py [-h] [-n_games N_GAMES] [-lr LR] [-eps_min EPS_MIN]
               [-gamma GAMMA] [-eps_dec EPS_DEC] [-eps EPS]
               [-max_mem MAX_MEM] [-repeat REPEAT] [-bs BS]
               [-replace REPLACE] [-env ENV] [-device DEVICE]
               [-load_checkpoint LOAD_CHECKPOINT] [-path PATH]
               [-algo ALGO] [-clip_rewards CLIP_REWARDS] [-no_ops NO_OPS]
               [-fire_first FIRE_FIRST]

Deep Q Learning

options:
  -h, --help            show this help message and exit
  -n_games N_GAMES      number of games to play
  -lr LR                learning rate for optimizer
  -eps_min EPS_MIN      minimum value for epsilon in epsilon-greedy action
                        selection
  -gamma GAMMA          discount factor for bellman update
  -eps_dec EPS_DEC      linear factor for decreasing epsilon
  -eps EPS              starting value for epsilon in epsilon-greedy
                        action selection
  -max_mem MAX_MEM      maximum size for memory replay buffer
  -repeat REPEAT        number of frames to stack for environment
  -bs BS                batch size for replay memory sampling
  -replace REPLACE      interval for replacing target network
  -env ENV              atari environment. PongNoFrameskip-v4
                        BreakoutNoFrameskip-v4 SpaceInvadersNoFrameskip-v4
                        EnduroNoFrameskip-v4 AtlantisNoFrameskip-v4
  -device DEVICE        GPU: mps or cuda:0 or cuda:1
  -load_checkpoint LOAD_CHECKPOINT
                        load model checkpoint
  -path PATH            path for model saving/loading
  -algo ALGO            DQNAgent/DDQNAgent/DuelingDQNAgent/DuelingDDQNAgen
                        t
  -clip_rewards CLIP_REWARDS
                        clip rewards to range -1 to 1
  -no_ops NO_OPS        max number of no ops for testing
  -fire_first FIRE_FIRST
                        set first action of episode to fire
```