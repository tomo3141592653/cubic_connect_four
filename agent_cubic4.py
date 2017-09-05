#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import copy

import numpy as np
np.random.seed(0)
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import optimizers
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3


class QNet(chainer.Chain):

    #n_outは4*4 n_inは4*4*4
    def __init__(self, n_in, n_units, n_out):
        super(QNet, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )

    def value(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.l3(h)

    def __call__(self, s_data, a_data, y_data):
        self.loss = None

        s = chainer.Variable(self.xp.asarray(s_data))
        Q = self.value(s)

        Q_data = copy.deepcopy(Q.data)

        if type(Q_data).__module__ != np.__name__:
            Q_data = self.xp.asnumpy(Q_data)

        t_data = copy.deepcopy(Q_data)
        for i in range(len(y_data)):
            t_data[i, a_data[i]] = y_data[i]

        t = chainer.Variable(self.xp.asarray(t_data))
        self.loss = F.mean_squared_error(Q, t)

        print('Loss:', self.loss.data)

        return self.loss


# エージェントクラス
class MarubatsuAgent(Agent):

    # エージェントの初期化
    # 学習の内容を定義する
    def __init__(self, gpu):
        # 盤の情報
        self.n_rows = 4
        self.n_cols = self.n_rows
        #高さを加える。
        self.n_hights = self.n_rows


        # 学習のInputサイズ
        self.dim = self.n_rows * self.n_cols * self.n_hights
        self.bdim = self.dim * 2 #白と黒の2

        self.gpu = gpu

        # 学習を開始させるステップ数
        self.learn_start = 5 * 10**3

        # 保持するデータ数
        self.capacity = 1 * 10**4

        # eps = ランダムに○を決定する確率
        self.eps_start = 1.0
        self.eps_end = 0.001
        self.eps = self.eps_start

        # 学習時にさかのぼるAction数
        #1でもいいのでは
        self.n_frames = 3

        # 一度の学習で使用するデータサイズ
        self.batch_size = 32

        self.replay_mem = []
        self.last_state = None
        self.last_action = None
        self.reward = None
        self.state = np.zeros((1, self.n_frames, self.bdim)).astype(np.float32) #これは何　盤の状態？

        self.step_counter = 0

        self.update_freq = 1 * 10**4

        self.r_win = 1.0
        self.r_draw = -0.5
        self.r_lose = -1.0

        self.frozen = False

        self.win_or_draw = 0
        self.stop_learning = 200

    # ゲーム情報の初期化
    def agent_init(self, task_spec_str):
        task_spec = TaskSpecVRLGLUE3.TaskSpecParser(task_spec_str)

        if not task_spec.valid:
            raise ValueError(
                'Task spec could not be parsed: {}'.format(task_spec_str))

        self.gamma = task_spec.getDiscountFactor()

        #n_framesかけるのはなぜ。(過去を遡る必要はないのではないだろうか)
        self.Q = QNet(self.bdim*self.n_frames, 30, self.n_rows * self.n_cols)

        if self.gpu >= 0:
            cuda.get_device(self.gpu).use()
            self.Q.to_gpu()
        self.xp = np if self.gpu < 0 else cuda.cupy

        self.targetQ = copy.deepcopy(self.Q)

        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95,
                                                  momentum=0.0)
        self.optimizer.setup(self.Q)

    # environment.py env_startの次に呼び出される。
    # 1手目の○を決定し、返す
    def agent_start(self, observation):
        # stepを1増やす
        self.step_counter += 1

        # observationを[0-2]の9ユニットから[0-1]の18ユニットに変換する
        self.update_state(observation)

        self.update_targetQ()

        # ○の場所を決定する
        int_action = self.select_int_action()
        action = Action()
        action.intArray = [int_action]

        # eps を更新する。epsはランダムに○を打つ確率
        self.update_eps()

        # state = 盤の状態 と action = ○を打つ場所 を退避する
        self.last_state = copy.deepcopy(self.state)
        self.last_action = copy.deepcopy(int_action)

        return action

    # エージェントの二手目以降、ゲームが終わるまで
    def agent_step(self, reward, observation):
        # ステップを1増加
        self.step_counter += 1

        self.update_state(observation)
        self.update_targetQ()

        # ○の場所を決定
        int_action = self.select_int_action()
        action = Action()
        action.intArray = [int_action]
        self.reward = reward

        # epsを更新
        self.update_eps()

        # データを保存 (状態、アクション、報酬、結果)
        self.store_transition(terminal=False)

        if not self.frozen:
            # 学習実行
            if self.step_counter > self.learn_start:
                self.replay_experience()

        self.last_state = copy.deepcopy(self.state)
        self.last_action = copy.deepcopy(int_action)

        # ○の位置をエージェントへ渡す
        return action

    # ゲームが終了した時点で呼ばれる
    def agent_end(self, reward):
        # 環境から受け取った報酬
        self.reward = reward

        if not self.frozen:
            if self.reward >= self.r_draw:
                self.win_or_draw += 1
            else:
                self.win_or_draw = 0

            if self.win_or_draw == self.stop_learning:
                self.frozen = True
                f = open('result.txt', 'a')
                f.writelines('Agent frozen\n')
                f.close()

        # データを保存 (状態、アクション、報酬、結果)
        self.store_transition(terminal=True)

        if not self.frozen:
            # 学習実行
            if self.step_counter > self.learn_start:
                self.replay_experience()

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass

    def update_state(self, observation=None):
        if observation is None:
            frame = np.zeros(1, 1, self.bdim).astype(np.float32)
        else:
            observation_binArray = []

            for int_observation in observation.intArray:
                bin_observation = '{0:02b}'.format(int_observation)
                observation_binArray.append(int(bin_observation[0]))
                observation_binArray.append(int(bin_observation[1]))

            frame = (np.asarray(observation_binArray).astype(np.float32)
                                                     .reshape(1, 1, -1))
        self.state = np.hstack((self.state[:, 1:], frame))

    def update_eps(self):
        if self.step_counter > self.learn_start:
            if len(self.replay_mem) < self.capacity:
                self.eps -= ((self.eps_start - self.eps_end) /
                             (self.capacity - self.learn_start + 1))

    def update_targetQ(self):
        if self.step_counter % self.update_freq == 0:
            self.targetQ = copy.deepcopy(self.Q)

    def select_int_action(self):
        free = [] #topが開いている場所 4*4
        bits = self.state[0, -1] #最新のデータ。

        for i in range(0,16):
            state = i + 16*3
            if bits[state*2] == 0 and bits[state*2 +1] == 0:
                free.append(i)

        def drop(i_top):
            for z in range(4):
                state = z*16 + i_top
                if bits[state*2] == 0 and bits[state*2 +1] == 0:
                    return state


        #一番上のやつにする。

        s = chainer.Variable(self.xp.asarray(self.state))
        Q = self.Q.value(s)

        # Follow the epsilon greedy strategy
        if np.random.rand() < self.eps:
            int_top = free[np.random.randint(len(free))]
            #int_action = drop(int_top)
            int_action = int_top
        else:
            Qdata = Q.data[0]

            if type(Qdata).__module__ != np.__name__:
                Qdata = self.xp.asnumpy(Qdata)

            for i in np.argsort(-Qdata):
                if i in free:
                    int_top = i
                    #int_action = drop(int_top)
                    int_action = int_top
                    break

        return int_action #まてよこれってint_topの方がいいのでは。

    def store_transition(self, terminal=False):
        if len(self.replay_mem) < self.capacity:
            self.replay_mem.append(
                (self.last_state, self.last_action, self.reward,
                 self.state, terminal))
        else:
            self.replay_mem = (self.replay_mem[1:] +
                [(self.last_state, self.last_action, self.reward, self.state,
                  terminal)])

    def replay_experience(self):
        indices = np.random.randint(0, len(self.replay_mem), self.batch_size)
        samples = np.asarray(self.replay_mem)[indices]

        s, a, r, s2, t = [], [], [], [], []

        for sample in samples:
            s.append(sample[0])
            a.append(sample[1])
            r.append(sample[2])
            s2.append(sample[3])
            t.append(sample[4])

        s = np.asarray(s).astype(np.float32)
        a = np.asarray(a).astype(np.int32)
        r = np.asarray(r).astype(np.float32)
        s2 = np.asarray(s2).astype(np.float32)
        t = np.asarray(t).astype(np.float32)

        s2 = chainer.Variable(self.xp.asarray(s2))
        Q = self.targetQ.value(s2)
        Q_data = Q.data

        if type(Q_data).__module__ == np.__name__:
            max_Q_data = np.max(Q_data, axis=1)
        else:
            max_Q_data = np.max(self.xp.asnumpy(Q_data).astype(np.float32), axis=1)

        t = np.sign(r) + (1 - t)*self.gamma*max_Q_data

        self.optimizer.update(self.Q, s, a, t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Q-Learning')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    AgentLoader.loadAgent(MarubatsuAgent(args.gpu))
