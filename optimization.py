# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import chainer


class AdamRule(chainer.optimizers.adam.AdamRule):
    def init_state(self, param):
        super(AdamRule, self).init_state(param)
        # Cancel weight decay in `bias` or `layer normalization` paramters
        if param.ndim <= 1:
            self.hyperparam.weight_decay_rate = 0.


class WeightDecayForMatrixAdam(chainer.optimizers.Adam):
    def create_update_rule(self):
        return AdamRule(self.hyperparam)
