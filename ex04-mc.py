import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('Blackjack-v0')


def single_run_20():
    """
    run the policy that sticks for >= 20
    """
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    # It can be used for the subtasks
    # Use a comment for the print outputs to increase performance (only there as example)
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    states = []
    ret = 0.
    while not done:
        print("observation:", obs)
        states.append(obs)
        if obs[0] >= 20:
            print("stick")
            obs, reward, done, _ = env.step(0)  # step=0 for stick
        else:
            print("hit")
            obs, reward, done, _ = env.step(1)  # step=1 for hit
        print("reward:", reward, "\n")
        ret += reward  # Note that gamma = 1. in this exercise
    print("final observation:", obs)
    return states, ret


def get_index(state):
    player_sum, dealer_card, useable_ace = state
    return player_sum-12, dealer_card-1, 1 if useable_ace else 0


def get_s_a_pair_index(s_a_pair):
    state, action = s_a_pair
    player_sum, dealer_card, useable_ace = state
    return player_sum-12, dealer_card-1, 1 if useable_ace else 0, action


def generate_episode_ver1():
    # Initialize lists to store sequence of states, actions and rewards
    states, actions, rewards = [], [], []

    # Reset the gym environment
    observation = env.reset()

    while True:
        # Append observations to the states array
        states.append(observation)

        # Select an action using the policy (see Slide Lec3-15) and append it to actions array
        score, dealer_score, usable_ace = observation
        action = 0 if score >= 20 else 1
        actions.append(action)

        # Perform action in current state, move to next state and receive reward
        observation, reward, done, info = env.step(action)
        rewards.append(reward)

        # If state is terminal, break loop
        if done:
            break

    return states, actions, rewards


def generate_episode_ver2(policy, epsilon=0.1):
    # Initialize lists to store sequence of states, actions and rewards
    # Use randomness by including epsilon (exploration)

    state_action_pairs, rewards = [], []

    observation = env.reset()

    while True:
        rV = np.random.uniform(0., 1.)
        if rV < epsilon:
            action = np.random.randint(0, 1)
        else:
            index = get_index(observation)
            action = policy[index]  # Choose next action from policy

        state_action_pairs.append((observation,action))

        observation, reward, done, info = env.step(action)
        rewards.append(reward)

        if done:
            break

    return state_action_pairs, rewards


def choose_initial_state_action():
    """
    Choose initial state-action pair
    """
    sums = np.arange(12, 22, 1)
    dealer = np.arange(1, 11, 1)
    ace = np.array([True, False])
    actions = np.array([0, 1])

    initial_state = (np.random.choice(sums), np.random.choice(dealer), np.random.choice(ace))
    initial_action = np.random.choice(actions)

    return initial_state, initial_action


def policy_evaluation(maxiter=10000):
    """
    Implementation of first-visit Monte Carlo prediction
    """
    # Dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # Initialize some important variables for computation
    V = np.zeros((10, 10, 2))
    returns = np.zeros((10, 10, 2))
    visits = np.zeros((10, 10, 2))

    for i in range(maxiter):
        # Generate an Episode following Policy
        states, actions, rewards = generate_episode_ver1()

        # Initialize G
        G = 0

        # Loop for each step of the episode, t=T-1,T-2,...,0
        for t in range (len(states) - 1, - 1, -1):
            Rtplus1 = rewards[t]
            St = states[t]

            G += Rtplus1

            # First-Visit: Check if the state was already visited in earlier steps of episode
            if St not in states[:t]:
                ind = get_index(St)
                visits[ind] += 1
                returns[ind] += G
                V[ind] = returns[ind]/visits[ind]

    return V


def monte_carlo_es(maxiter=2000000):
    """
    Implementation of Monte Carlo ES
    """
    # Dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # Initialize some important variables for computation
    pi = np.zeros((10, 10, 2), dtype=int)
    V = np.zeros((10, 10, 2))
    # Q = np.zeros((10, 10, 2, 2))
    Q = np.ones((10, 10, 2, 2)) * 100  # recommended: optimistic initialization of Q
    returns = np.zeros((10, 10, 2, 2))
    visits = np.zeros((10, 10, 2, 2))

    for i in range(maxiter):
        state_action_pairs, rewards = generate_episode_ver2(pi, epsilon=0.5)

        G = 0

        # Loop for each step of the episode, t=T-1,T-2,...,0
        for t in range(len(state_action_pairs) - 1, - 1, -1):
            Rtplus1 = rewards[t]
            S_A_Pair = state_action_pairs[t]

            G += Rtplus1

            # First-Visit: Check if the state_action_pair was already visited in earlier steps of episode
            if S_A_Pair not in state_action_pairs[:t]:
                state, _ = S_A_Pair
                S_ind = get_index(state)
                SA_ind = get_s_a_pair_index(S_A_Pair)
                visits[SA_ind] += 1
                returns[SA_ind] += G
                Q[SA_ind] = (returns[SA_ind]) / visits[SA_ind]
                pi[S_ind] = np.argmax(Q[S_ind])
                V[S_ind] = np.max(Q[S_ind])

        if i % 100000 == 0:
            print("Policy after Iteration: " + str(i))
            print(pi[:, :, 0])
            #print(pi[:, :, 1])
            print("Value Function after Iteration: " + str(i))
            print(V[:, :, 0])
            #print(pi[:, :, 1])

    return pi, V


def plot_value_function(val_func):
    # Set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax1 = plt.subplot(1, 2, 1, projection='3d')
    sums = np.arange(12, 22, 1)
    dealer = np.arange(1, 11, 1)
    S, D = np.meshgrid(sums, dealer)
    ax1.plot_surface(S, D, val_func[:, :, 0])
    ax1.set_title('Without Useable Ace')
    ax1.set_xlabel("Player's Sum")
    ax1.set_ylabel('Dealer Card')
    ax1.set_zlabel('Value')

    ax2 = plt.subplot(1, 2, 2, projection='3d')
    sums = np.arange(12, 22, 1)
    dealer = np.arange(1, 11, 1)
    S, D = np.meshgrid(sums, dealer)
    ax2.plot_surface(S, D, val_func[:, :, 1])
    ax2.set_title('With Useable Ace')
    ax2.set_xlabel("Player's Sum")
    ax2.set_ylabel('Dealer Card')
    ax2.set_zlabel('Value')

    plt.show()


def main():
    #single_run_20()

    ## 10,000 Iterations
    # V10k = policy_evaluation(maxiter=10000)
    # plot_value_function(V10k)
    ## 500,000 Iterations
    # V500k = policy_evaluation(maxiter=500000)
    # plot_value_function(V500k)
    #print(env.reset())

    # print(np.zeros((10,10), dtype=np.int8))
    # print(generate_episode_ver2(policy=np.zeros((10, 10, 2), dtype=np.int8), epsilon=0.9))
    # print(choose_initial_state_action())
    pol, v_func = monte_carlo_es()

    plot_value_function(v_func)


if __name__ == "__main__":
    main()
