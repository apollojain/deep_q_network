import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
import os.path

MODEL_INPUT_SHAPE = (84, 84, 4,)

class Agent(object):
    def __init__(self, env, gamma=0.99, 
                epsilon_min=0.1, batch_size=32, max_steps_per_episode=10000,
                reward_threshold=40, model_filepath="model.json",
                model_weights_filepath="model.h5"):
        # Create environment
        self.env = env

        # Initialize constants
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_max = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_interval = (self.epsilon_max - self.epsilon_min)
        self.batch_size = batch_size
        self.max_steps_per_episode = max_steps_per_episode
        self.num_actions = self.env.action_space.n
        self.reward_threshold = reward_threshold

        # Model parameters
        self.model_filepath = model_filepath
        self.model_weights_filepath = model_weights_filepath
        assert(self.env.observation_space.shape == MODEL_INPUT_SHAPE)
        self.input_shape = MODEL_INPUT_SHAPE
        self.model = self.get_model()
        self.model_target = self.get_model()
        self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        self.loss_function = keras.losses.Huber()

    def get_model(self):
        # check if model exists
        if os.path.exists(self.model_filepath) and os.path.exists(self.model_weights_filepath):
            with open(self.model_filepath) as json_file:
                loaded_model_json = json_file.read()
                loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(self.model_weights_filepath)
            return loaded_model
        else:
            return self.create_model()

    def save_model(self):
        model_json = self.model.to_json()
        with open(self.model_filepath, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.model_weights_filepath)
    
    def create_model(self):
        # Network defined by the Deepmind paper
        inputs = layers.Input(shape=self.input_shape)

        # Convolutions on the frames on the screen
        layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = layers.Flatten()(layer3)

        layer5 = layers.Dense(512, activation="relu")(layer4)
        action = layers.Dense(self.num_actions, activation="linear")(layer5)

        return keras.Model(inputs=inputs, outputs=action)

    def train_step(self):
        self.state = np.array(self.env.reset())
        episode_reward = 0

        for timestep in range(1, self.max_steps_per_episode):
            self.frame_count += 1

            if self.frame_count < self.epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
                action = np.random.choice(self.num_actions)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(self.state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = self.model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()
            
            # Decay probability of taking random action
            self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
            self.epsilon = max(self.epsilon, self.epsilon_min)

            # Apply the sampled action in our environment
            state_next, reward, done, _ = self.env.step(action)
            state_next = np.array(state_next)

            episode_reward += reward

            # Save actions and states in replay buffer
            self.action_history.append(action)
            self.state_history.append(self.state)
            self.state_next_history.append(state_next)
            self.done_history.append(done)
            self.rewards_history.append(reward)
            self.state = state_next

            if self.frame_count % self.update_after_actions == 0 and len(self.done_history) > self.batch_size:

                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)

                # use list comprehension to sample from replay buffer
                state_sample = np.array([self.state_history[i] for i in indices])
                state_next_sample = np.array([self.state_next_history[i] for i in indices])
                rewards_sample = [self.rewards_history[i] for i in indices]
                action_sample = [self.action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(self.done_history[i]) for i in indices]
                )

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = self.model_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + self.gamma * tf.reduce_max(
                    future_rewards, axis=1
                )

                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, self.num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = self.model(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for the action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # calculate the loss
                    loss = self.loss_function(updated_q_values, q_action)
                
                # Backpropogation
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            if self.frame_count % self.update_target_network == 0:
                # update the target network with new weights
                self.model_target.set_weights(self.model.get_weights())
                # Log details
                template = "Running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(self.running_reward, self.episode_count, self.frame_count))
                # save weights from model
                self.save_model()

            # Limit the state and reward
            if len(self.rewards_history) > self.max_memory_length:
                del self.rewards_history[:1]
                del self.state_history[:1]
                del self.state_next_history[:1]
                del self.action_history[:1]
                del self.done_history[:1]

            if done:
                break

        return episode_reward

    def train(self, epsilon_random_frames=50000, epsilon_greedy_frames=1000000.0,
            max_memory_length=100000, update_after_actions=4,
            update_target_network=10000):
        # Experience replay buffers
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = []
        self.running_reward = 0
        self.episode_count = 0
        self.frame_count = 0

        # Other relevant variables
        self.epsilon_random_frames = 50000
        self.epsilon_greedy_frames = 1000000.0
        self.max_memory_length = 100000
        self.update_after_actions = 4
        self.update_target_network = 10000

        while True:
            episode_reward = self.train_step()
            
            self.episode_reward_history.append(episode_reward)
            if len(self.episode_reward_history) > 100:
                del self.episode_reward_history[:1]
            self.running_reward = np.mean(self.episode_reward_history)

            self.episode_count += 1

            if self.running_reward > self.reward_threshold:
                print("Solved at episode {}!".format(self.episode_count))
                break

        self.save_model()

    def run(self):
        done = False
        state = self.get_state(self.env.reset())
        while not done:
            self.env.render()
            # Take best action
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()
            state_next, reward, done, _ = self.env.step(action)
            # get new state
            state_next = np.array(state_next)
            state = state_next