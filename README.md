# DQN Agent

## About
This particular Agent class is adopted from a Keras example. My goal was to generalize it and make it more production-ready. This particular rendition has the ability to save weights, train on any arbitrary Atari game, and also run through a single episode once training is complete for demo purposes.

## Setup
The first step is propping up a Virtual Environment that has all of the requirements you need. Before running these steps, make sure you have `python3` and `virtualenv` installed. To get everything you need to run the DQN Agent, run through the following steps on a Mac:
```
$ brew install cmake openmpi
$ cd dqn_agent
$ virtualenv ./breakout_env --python=python3
$ source ./breakout_env/bin/activate
$ pip install -r requirements.txt
$ git clone https://github.com/openai/baselines.git
$ cd baselines
$ pip install -e .
$ cd ..
```

## Examples
An example of both training and running this agent is provided in `breakout.py`.