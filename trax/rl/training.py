# coding=utf-8
# Copyright 2020 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Classes for RL training in Trax."""

import gym
import numpy as np

from trax import layers as tl
from trax import supervised


class TimeStep(object):
  """A single step of a play in RL.

  TimeStep stores a state and potentially an action taken from it, a reward
  that was gained when taking the action, and potentially the discounted return
  gained from that state on (which also includes the reward from this step).
  """

  def __init__(self, state, action=None, reward=None, discounted_return=None):
    self.state = state
    self.action = action
    self.reward = reward
    self.discounted_return = discounted_return


def calculate_returns(trajectory, gamma):
  """Calculate discounted returns in a trajectory."""
  ret = 0.0
  for timestep in reversed(trajectory):
    cur_ret = timestep.reward or 0.0
    ret = gamma * ret + cur_ret
    timestep.discounted_return = ret


def _default_timestep_to_np(ts):
  """Default way to convert timestep to numpy."""
  return (np.array(ts.state, dtype=np.float32),
          np.array(ts.action + 1, dtype=np.int32),  # Shift by 1 for padding.
          np.array(ts.reward, dtype=np.float32),
          np.array(ts.discounted_return, dtype=np.float32))


class RLTask:
  """A RL task: environment and a collection of trajectories."""

  def __init__(self, env, initial_trajectories=1, gamma=0.99, max_steps=None,
               timestep_to_np=None):
    r"""Configures a RL task.

    Args:
      env: Environment confirming to the gym.Env interface or a string,
        in which case `gym.make` will be called on this string to create an env.
      initial_trajectories: either a list of trajectories (lists of TimeSteps)
        to use as at start or an int, in which case that many trajectories are
        collected using a random policy to play in env.
      gamma: float: discount factor for calculating returns.
      max_steps: Optional int: stop all trajectories at that many steps.
      timestep_to_np: a function that turns a timestep into a numpy array
        (ie., a tensor); if None, we just use the state of the timestep to
        represent it, but other representations (such as embeddings that include
        actions or serialized representations) can be passed here.

    """
    if isinstance(env, str):
      env = gym.make(env)
    self._env = env
    self._max_steps = max_steps
    self._gamma = gamma
    if isinstance(initial_trajectories, int):
      random_policy = lambda _: np.random.randint(self.n_actions)
      initial_trajectories = [self.play(random_policy)
                              for _ in range(initial_trajectories)]
    self._timestep_to_np = timestep_to_np or _default_timestep_to_np
    # Stored trajectories are indexed by epoch and within each epoch they
    # are stored in the order of generation so we can implement replay buffers.
    # TODO(lukaszkaiser): use dump_trajectories from BaseTrainer to allow
    # saving and reading trajectories from disk.
    self._trajectories = {0: initial_trajectories}

  @property
  def max_steps(self):
    return self._max_steps

  @property
  def n_actions(self):
    return self._env.action_space.n

  @property
  def state_shape(self):
    return self._env.observation_space.shape

  def play(self, policy):
    """Play an episode in env taking actions according to the given policy.

    Environment is first reset and an from then on, a game proceeds. At each
    step, the policy is asked to choose an action and the environment moves
    forward. A Trajectory is created in that way and returns when the episode
    finished, which is either when env returns `done` or max_steps is reached.

    Args:
      policy: a function taking a Trajectory and returning an action (int).

    Returns:
      a completed trajectory that was just played.
    """
    terminal = False
    cur_step = 0
    cur_timestep = TimeStep(self._env.reset())
    cur_trajectory = [cur_timestep]
    while not terminal and cur_step < self.max_steps:
      action = policy(cur_trajectory)
      cur_timestep.action = action
      state, reward, terminal, _ = self._env.step(action)
      cur_timestep.reward = reward
      cur_timestep = TimeStep(state)
      cur_trajectory.append(cur_timestep)
      cur_step += 1
    calculate_returns(cur_trajectory, self._gamma)
    return cur_trajectory

  def collect_trajectories(self, policy, n, epoch_id=1):
    """Collect n trajectories in env playing the given policy."""
    new_trajectories = [self.play(policy) for _ in range(n)]
    if epoch_id not in self._trajectories:
      self._trajectories[epoch_id] = []
    self._trajectories[epoch_id].extend(new_trajectories)
    returns = [sum([ts.reward for ts in tr[:-1]]) for tr in new_trajectories]
    return sum(returns) / float(len(returns))

  def trajectory_to_np(self, trajectory):
    """Create a tuple of numpy arrays from a given trajectory."""
    observations, actions, rewards, returns = [], [], [], []
    for timestep in trajectory:
      # TODO(lukaszkaiser): We do not include the end state. Should we?
      if timestep.action is not None:
        (obs, act, rew, ret) = self._timestep_to_np(timestep)
        observations.append(obs[None, ...])
        actions.append(act[None, ...])
        rewards.append(rew[None, ...])
        returns.append(ret[None, ...])
    return (np.concatenate(observations, axis=0),
            np.concatenate(actions, axis=0),
            np.concatenate(rewards, axis=0),
            np.concatenate(returns, axis=0))

  def trajectory_stream(self, epochs=None, max_slice_length=None):
    """Return a stream of random trajectory slices from the specified epochs.

    Args:
      epochs: a list of epochs to use; we use all epochs if None
      max_slice_length: maximum length of the slices of trajectories to return

    Yields:
      random trajectory slices sampled uniformly from all slices of length
      upto max_slice_length in all specified epochs
    """
    # TODO(lukaszkaiser): add option to sample from n last trajectories.
    def n_slices(t):
      """How many slices of length upto max_slice_length in a trajectory."""
      if not max_slice_length:
        return 1
      # A trajectory [a, b, c, end_state] will have 2 proper slices of length 2:
      # the slice [a, b] and the one [b, c].
      return max(1, len(t) - max_slice_length)

    # TODO(lukaszkaiser): the code below is slow, make it fast.
    while True:
      epochs = epochs or list(self._trajectories.keys())
      slices = [[n_slices(t) for t in self._trajectories[ep]] for ep in epochs]
      slices_per_epoch = [sum(s) for s in slices]
      slice_id = np.random.randint(sum(slices_per_epoch))  # Which slice?
      # We picked a trajectory slice, which epoch and trajectory is it in?
      slices_per_epoch_sums = np.array(slices_per_epoch).cumsum()
      epoch_id = min([i for i in range(len(epochs))
                      if slices_per_epoch_sums[i] >= slice_id])
      slice_in_epoch_id = slices_per_epoch_sums[epoch_id] - slice_id
      slices_in_epoch_sums = np.array(slices[epoch_id]).cumsum()
      trajectory_id = min([i for i in range(len(self._trajectories[epoch_id]))
                           if slices_in_epoch_sums[i] >= slice_in_epoch_id])
      trajectory = self._trajectories[epoch_id][trajectory_id]
      slice_start = slices_in_epoch_sums[trajectory_id] - slice_in_epoch_id
      slice_end = slice_start + (max_slice_length or len(trajectory))
      slice_end = min(slice_end, len(trajectory) - 1)
      yield trajectory[slice_start:slice_end]

  def batches_stream(self, batch_size, epochs=None, max_slice_length=None):
    """Return a stream of trajectory batches from the specified epochs.

    This function returns a stream of tuples of numpy arrays (tensors).
    If tensors have different lengths, they will be padded by 0.

    Args:
      batch_size: the size of the batches to return
      epochs: a list of epochs to use; we use all epochs if None
      max_slice_length: maximum length of the slices of trajectories to return

    Yields:
      batches of trajectory slices sampled uniformly from all slices of length
      upto max_slice_length in all specified epochs
    """
    def pad(tensor_list):
      max_len = max([t.shape[0] for t in tensor_list])
      pad_len = 2**int(np.ceil(np.log2(max_len)))
      return np.array([tl.zero_pad(t, (0, pad_len - t.shape[0]), axis=0)
                       for t in tensor_list])
    cur_batch = []
    for t in self.trajectory_stream(epochs, max_slice_length):
      cur_batch.append(t)
      if len(cur_batch) == batch_size:
        obs, act, rew, ret = zip(*[self.trajectory_to_np(t) for t in cur_batch])
        yield pad(obs), pad(act), pad(rew), pad(ret)
        cur_batch = []


class PolicyTrainer:
  """Trains a policy model using Reinforce on the given RLTask.

  This is meant as an example for RL loops for other RL algorithms.
  """

  def __init__(self, model, optimizer, lr_schedule, task, batch_size=16,
               train_steps_per_epoch=100, collect_per_epoch=100,
               max_slice_length=1, output_dir=None):
    """Configures the Reinforce loop.

    Args:
      model: Trax layer, representing the policy model.
          functions and eval functions (a.k.a. metrics) are considered to be
          outside the core model, taking core model output and data labels as
          their two inputs.
      optimizer: the optimizer to use to train the model.
      lr_schedule: learning rate schedule to use to train the model.
      task: RLTask instance, which defines the environment to train on.
      batch_size: batch size used to train the model.
      train_steps_per_epoch: how long to train in each RL epoch.
      collect_per_epoch: how many trajectories to collect per epoch.
      max_slice_length: the maximum length of trajectory slices to use.
      output_dir: Path telling where to save outputs (evals and checkpoints).
          Can be None if both `eval_task` and `checkpoint_at` are None.
    """
    self._task = task
    self._batch_size = batch_size
    self._train_steps_per_epoch = train_steps_per_epoch
    self._collect_per_epoch = collect_per_epoch
    self._max_slice_length = max_slice_length
    self._output_dir = output_dir
    self._epoch = 0
    self._eval_model = model(mode='eval')
    example_batch = next(self._batches_stream())
    self._eval_model.init(example_batch)
    self._inputs = supervised.Inputs(
        train_stream=lambda _: self._batches_stream())
    self._trainer = supervised.Trainer(
        model=model, optimizer=optimizer, lr_schedule=lr_schedule,
        loss_fn=tl.CrossEntropyLoss, inputs=self._inputs, output_dir=output_dir,
        has_weights=True, id_to_mask=0)

  def _batches_stream(self):
    for (obs, act, _, ret) in self._task.batches_stream(
        self._batch_size, max_slice_length=self._max_slice_length):
      norm_ret = ret / np.std(ret)
      yield obs, act, norm_ret

  @property
  def current_epoch(self):
    """Returns current step number in this training session."""
    return self._epoch

  def policy(self, trajectory):
    model = self._eval_model
    model.weights = self._trainer.model_weights
    pred = model(trajectory[-1].state[None, ...])
    return tl.gumbel_sample(pred[0, 1:])

  def run(self, n_epochs=1):
    """Runs this loop for n epochs.

    Args:
      n_epochs: Stop training after completing n steps.
    """
    for _ in range(n_epochs):
      self._epoch += 1
      self._trainer.train_epoch(self._train_steps_per_epoch, 10)
      avg_return = self._task.collect_trajectories(
          self.policy, self._collect_per_epoch, self._epoch)
      print('Average return in epoch %d was %.2f.' % (self._epoch, avg_return))
