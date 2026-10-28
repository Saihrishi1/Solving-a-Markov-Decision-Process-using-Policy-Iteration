# Solving a Markov Decision Process using Policy Iteration

## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, by repeatedly performing policy evaluation and policy improvement to obtain the optimal value function and optimal policy.

---

## Problem Statement

In this experiment, the `FrozenLake-v1` environment is solved using the **Policy Iteration** algorithm.

The agent starts from the start state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process. Policy Iteration is used to repeatedly evaluate the current policy and improve it until the policy becomes stable.

The objective is to find:

1. The optimal state-value function $V^*(s)$
2. The optimal policy $pi^*(s)$

---

## Software Requirements

```bash
pip install gymnasium numpy
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---

## Theory

Policy Iteration is a Dynamic Programming method used to find the optimal policy of a Markov Decision Process.

It consists of two major steps:

1. **Policy Evaluation**
2. **Policy Improvement**

These two steps are repeated until the policy becomes stable.

---

## Policy Evaluation

Policy evaluation estimates the value function for the current policy.

The Bellman expectation equation is:

$$
V^\pi(s) =
\sum_a \pi(a \mid s)
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $pi(a \mid s)$ | Probability of taking action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $pi$ |

---

## Policy Improvement

Policy improvement updates the policy greedily with respect to the current value function.

The improved policy is obtained as:

$$
\pi'(s) =
\arg\max_a
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

If the improved policy is the same as the old policy, the policy is considered stable.

---

## Algorithm

1. Create the Gymnasium `FrozenLake-v1` environment.
2. Initialize a random policy.
3. Repeat until the policy becomes stable:
   - Evaluate the current policy using iterative policy evaluation.
   - Improve the policy greedily using the current value function.
   - Compare the old policy and the new policy.
4. Stop when the policy does not change.
5. Display the optimal value function and optimal policy.

---

## Python Program

```python

# -------------------------------------------------
# Policy Evaluation
# -------------------------------------------------
def policy_evaluation(env, policy, gamma, theta):

    V = np.zeros(env.observation_space.n)

    while True:
        delta = 0
        for state in range(n_states):
            v = V[state]
            new_v = 0
            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    new_v += action_prob * prob * (reward + gamma * V[next_state] * (not done))
            V[state] = new_v
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V


# -------------------------------------------------
# Policy Improvement
# -------------------------------------------------

def policy_improvement(env, V, policy, gamma):
    policy_stable = True
    for state in range(n_states):
        old_action = np.argmax(policy[state])
        action_values = np.zeros(n_actions)
        for action in range(n_actions):
            for prob, next_state, reward, done in env.P[state][action]:
                action_values[action] += prob * (reward + gamma * V[next_state] * (not done))
        best_action = np.argmax(action_values)
        policy[state] = np.eye(n_actions)[best_action]
        if old_action != best_action:
            policy_stable = False
    return policy, policy_stable


# -------------------------------------------------
# Policy Iteration
# -------------------------------------------------

def policy_iteration(env, gamma, theta):
    
    iteration = 0

    policy = np.ones((n_states, n_actions)) / n_actions
    
    V = np.zeros(n_states)
    
    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        policy, policy_stable = policy_improvement(env, V, policy, gamma)

        iteration += 1

        if policy_stable:
            break
    
    return policy, V, iteration

```

## Output

```text

Total Policy Iteration : 3

Optimal State-Value Function:
[[0.542  0.4988 0.4707 0.4569]
 [0.5585 0.     0.3583 0.    ]
 [0.5918 0.6431 0.6152 0.    ]
 [0.     0.7417 0.8628 0.    ]]

Optimal Policy:
[['←' '↑' '↑' '↑']
 ['←' '←' '←' '←']
 ['↑' '↓' '←' '←']
 ['←' '→' '↓' '←']]

```

<img width="496" height="361" alt="image" src="https://github.com/user-attachments/assets/99624291-0d46-436f-88ba-95088bd91b07" />

---

## Result

Thus, we have created a program for Policy Iteration and by repeatedly performing policy evaluation and policy improvement, have successfully obtained the optimal value function and optimal policy.

---

## Inference
```text


```
---

