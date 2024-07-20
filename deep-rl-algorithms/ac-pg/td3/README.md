## abstract - high level summary (what, why, outcome)

They implement a pair of critics and take the minimum value to prevent overestimation in actor-critic methods created due to accumulated approximation errors. Additionally, policy updates are delayed to reduce per-update error. *Essentially Double Q-learning but actor critic and delayed and minimum of critics.*

## intro - high level overview (focus on the why)

DDQN does not work in actor-critic methods due to slow-changing policy which result in insignificant  difference between target and current networks to avoid maximization bias. 

The work is similar to Double Q-learning, which use a pair of independently trained value networks which allows for a less biased value estimation. This isn't perfect as an unbiased estimate with high variance can still lead to future overestimations in local regions of state space, which hurt learning a global policy. Subsequently, they clip the estimations that makes the network favor underestimations.

To reduce variance, they use target networks (as in deep Q-learning) for reducing the accumulation of errors. They also address the coupling of value and policy by proposing a delay in policy updates until the value estimate converges. Then, they employ a SARSA-style bootstrap update for regularizing action estimates.

## background - mathematical details; pay close attention

Nothing new introduced here. Still, the key equations:
- DDPG target: $y = r+\gamma Q_{\theta'}(s', a'), a'\sim\pi_{\phi'}(s')$

## methods - how experiments were performed



## results - what are we shooting for (ballpark)



## architecture & algos - what are we implementing
Questions to answer - 
- what algorithm?
- what data structures?
- what model architecture?
- what hyperparameters?
```
```
- what results to expect?