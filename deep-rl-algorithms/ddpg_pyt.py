# Need a replay buffer class
# Need a class for a target Q network (function of s, a)
# we will use batch norm
# the policy is deterministic, how to handle explore-exploit?
# deterministic policy means outputs the actual action instead of a probability
# will need a way to bound the actions to the environment limits
# We have two actor and two critic networks, a target for each.
# Updates are soft, according to theta_prime = tau*theta + (1-tau)*theta_prime, with tau << 1
# the target actor is just the evaluation actor plus some noise process -> will need a class for noise

