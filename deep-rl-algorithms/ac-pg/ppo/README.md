# PPO

## Usage
```
sage: ppo.py [-h] [--exp_name EXP_NAME] [--gym-id GYM_ID]
              [--learning-rate LEARNING_RATE]
              [--seed SEED]
              [--total-timesteps TOTAL_TIMESTEPS]
              [--torch-deterministic [TORCH_DETERMINISTIC]]
              [--mps [MPS]] [--track [TRACK]]
              [--wandb-project-name WANDB_PROJECT_NAME]
              [--wandb-entity WANDB_ENTITY]
              [--capture-video [CAPTURE_VIDEO]]
              [--num-envs NUM_ENVS]
              [--num-steps NUM_STEPS]
              [--anneal-lr [ANNEAL_LR]] [--gae [GAE]]
              [--gamma GAMMA] [--gae-lambda GAE_LAMBDA]
              [--num-minibatches NUM_MINIBATCHES]
              [--update-epochs UPDATE_EPOCHS]
              [--norm-adv [NORM_ADV]]
              [--clip-coef CLIP_COEF]
              [--clip-vloss [CLIP_VLOSS]]
              [--ent-coef ENT_COEF] [--vf-coef VF_COEF]
              [--max-grad-norm MAX_GRAD_NORM]
              [--target-kl TARGET_KL]

optional arguments:
  -h, --help            show this help message and exit
  --exp_name EXP_NAME   the name of this environment
  --gym-id GYM_ID       the id of the gym environment
  --learning-rate LEARNING_RATE
                        the learning rate of the optimizer
  --seed SEED           seed of the experiment
  --total-timesteps TOTAL_TIMESTEPS
                        tota timesteps of the experiment
  --torch-deterministic [TORCH_DETERMINISTIC]
                        if toggled, `torch.backends.cudnn.
                        deterministic=False`
  --mps [MPS]           if toggled, mps will not be
                        enabled by default
  --track [TRACK]       if toggled, this experiment will
                        be tracked on weights and biases.
  --wandb-project-name WANDB_PROJECT_NAME
                        the wandb's project name
  --wandb-entity WANDB_ENTITY
                        the entity (team) of wandb's
                        project
  --capture-video [CAPTURE_VIDEO]
                        whether to capture videos of the
                        agent performances (checkout
                        videos/ folder)
  --num-envs NUM_ENVS   the number of parallel game
                        environments
  --num-steps NUM_STEPS
                        the number of steps to run in each
                        environment per policy rollout
  --anneal-lr [ANNEAL_LR]
                        if toggled, learning rate for
                        policy and value networks is
                        annealed
  --gae [GAE]           use GAE for advantage estimation
  --gamma GAMMA         the discount factor gamma
  --gae-lambda GAE_LAMBDA
                        the lambda for GAE
  --num-minibatches NUM_MINIBATCHES
                        the number of mini-batches
  --update-epochs UPDATE_EPOCHS
                        the K epochs to update the policy
  --norm-adv [NORM_ADV]
                        if toggled, normalizes advantages
                        per update
  --clip-coef CLIP_COEF
                        the surrogate clipping coefficient
  --clip-vloss [CLIP_VLOSS]
                        if toggled, clipped loss for value
                        function is used
  --ent-coef ENT_COEF   coefficient of the entropy
  --vf-coef VF_COEF     coefficient of value function
  --max-grad-norm MAX_GRAD_NORM
                        the maximum norm for gradient
                        clipping
  --target-kl TARGET_KL
                        the target KL divergence threshold
```