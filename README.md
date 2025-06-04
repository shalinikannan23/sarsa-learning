# EX_06 - SARSA Learning Algorithm

## AIM

To implement and analyze the SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm for learning optimal policies in a given environment using on-policy temporal difference control.

## PROBLEM STATEMENT

In many real-world and simulated environments, an agent must learn an optimal sequence of actions to maximize cumulative rewards without prior knowledge of the environment's dynamics. The challenge lies in balancing exploration of new actions with exploitation of known rewarding strategies.

This experiment aims to solve the problem of learning an optimal policy for decision-making under uncertainty using the SARSA algorithm. SARSA is an on-policy Temporal Difference (TD) learning algorithm where the agent updates its Q-values based on the action actually taken in the next state. This approach allows the agent to improve its behavior gradually while still exploring the environment.

The goal is to:

- Implement the SARSA algorithm.

- Train the agent in a grid world or similar environment.

- Evaluate how effectively the agent learns an optimal policy over episodes.

- Compare performance metrics such as total reward per episode and convergence behavior.

## SARSA LEARNING ALGORITHM

![image](https://github.com/user-attachments/assets/59a1e43c-abc5-4440-a4fd-79eaf6e13b60)


## SARSA LEARNING FUNCTION
```py
DEVELOPED BY : SHALINI K
REGISTER NO. : 212222240095
```
```py
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    # Decay schedules for alpha and epsilon
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    # Policy based on epsilon-greedy
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(nA)

    for e in tqdm(range(n_episodes), leave=False):
        state = env.reset()
        action = select_action(state, Q, epsilons[e])
        done = False
        
        while not done:
            next_state, reward, done, _ = env.step(action)
            
            # Choose next action using epsilon-greedy policy
            next_action = select_action(next_state, Q, epsilons[e])
            
            # SARSA update
            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            td_delta = td_target - Q[state][action]
            Q[state][action] += alphas[e] * td_delta
            
            state = next_state
            action = next_action

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    # Calculate V and pi from the final Q
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:

### Mention the optimal policy, optimal value function , success rate for the optimal policy.
![image](https://github.com/user-attachments/assets/4abbf4ad-61f3-4f50-a58b-6e0a842e7334)
![image](https://github.com/user-attachments/assets/eae52f52-5187-4e64-a1c1-c9c38a0c4b0e)

![image](https://github.com/user-attachments/assets/1d20bed0-33bd-415d-82fc-28dff01c0605)

![image](https://github.com/user-attachments/assets/bd9609bd-741f-4648-9423-5abcbd932cd7)


### Include plot comparing the state value functions of Monte Carlo method and SARSA learning.

![image](https://github.com/user-attachments/assets/162a69ce-6688-4873-9660-489b622754fa)

![image](https://github.com/user-attachments/assets/434ba31e-530a-48a4-8d71-1600c6cdfa09)




## RESULT:

Thus, to implement and analyze the SARSA (State-Action-Reward-State-Action) reinforcement learning algorithm for learning optimal policies in a given environment using on-policy temporal difference control is successfully implemented. 
