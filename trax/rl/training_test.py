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
"""Tests for RL training."""

from absl.testing import absltest

from trax import layers as tl
from trax import lr_schedules
from trax import optimizers as opt
from trax.rl import training


class TrainingTest(absltest.TestCase):

  def test_task_sampling(self):
    """Trains a policy on cartpole."""
    tr1 = [training.TimeStep(1, 0, 0), training.TimeStep(1, 0, 0),
           training.TimeStep(2000)]
    tr2 = [training.TimeStep(4, 0, 0), training.TimeStep(2000)]
    task = training.RLTask(
        'CartPole-v0', initial_trajectories=[tr1, tr2])
    stream = task.trajectory_stream(max_slice_length=1)
    slices = []
    for _ in range(100):
      next_slice = next(stream)
      assert len(next_slice) == 1
      slices.append(next_slice[0].state)
    # mean_obs = sum(slices) / float(len(slices))
    # print("AAAAAAAAAAAAAAAAAA")
    # print(mean_obs)
    # print("AAAAAAAAAAAAAAAAAA")
    # Average should be around 3 since we're sampling from {1, 1, 4} uniformly.
    # assert mean_obs < 3.2
    # assert mean_obs > 2.8
    self.assertLen(slices, 100)

  def test_policytrainer_cartpole(self):
    """Trains a policy on cartpole."""
    task = training.RLTask('CartPole-v0', initial_trajectories=200,
                           max_steps=200)
    model = lambda mode: tl.Serial(  # pylint: disable=g-long-lambda
        tl.Dense(32), tl.Relu(), tl.Dense(3), tl.LogSoftmax())
    lr = lambda h: lr_schedules.MultifactorSchedule(  # pylint: disable=g-long-lambda
        h, constant=1e-4, warmup_steps=100, factors='constant * linear_warmup')
    trainer = training.PolicyTrainer(model, opt.Adam, lr, task)
    trainer.run(2)
    self.assertEqual(2, trainer.current_epoch)


if __name__ == '__main__':
  absltest.main()
