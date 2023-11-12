from izombie_dqn.dqn_agent import DQNAgent

if __name__ == "__main__":
    agent = DQNAgent(device="cpu")
    agent.train(episodes=2000)

