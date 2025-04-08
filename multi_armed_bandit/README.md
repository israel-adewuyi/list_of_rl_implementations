## Multi-Armed Bandit

Implementation of the Multi-arm bandit from Chapter 2 of the [Sutton book](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf).
Library - [Gymnasium](https://gymnasium.farama.org/)
Main reference code : [ARENA RL chapter](https://github.com/callummcdougall/ARENA_3.0/tree/main/chapter2_rl)
Assistant: [Qwen](chat.qwen.ai) 

`agent.py` implements the Epsilon-Greedy, Upper Confidence Bound action selection methods as well a random action selection method (for reference)
`multi_armed_bandit_env.py` implements the environment for the multi armed bandit.
`main.py` and `utils.py` - code to run the agents, collect reward stats, plot culmulative rewards and write plots to file.

### To run
`python main.py`
