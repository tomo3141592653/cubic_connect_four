#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
np.random.seed(0)
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Reward_observation_terminal


class MarubatsuEnvironment(Environment):

    # 盤の状態 [空白, ○, ×]
    flg_free = 0
    flg_agent = 1
    flg_env = 2

    # 報酬
    r_win = 1.0
    r_draw = -0.5
    r_lose = -1.0

    # 敵プレイヤーが正常に打つ確率
    opp = 0.75

    def __init__(self):
        self.n_rows = 4
        self.n_cols = self.n_rows

        self.n_heights = self.n_rows


        #4つ並んだら勝敗が作くline
        lines = [self.xyzs2points(xyzs) for xyzs in self.make_line_xyz()]


        self.lines = lines

        self.history = []


    def xyzs2points(self,xyzs):
        #[[1,2,3],[2,3,4]] -> [*,*,*]
        res = sorted([xyz[0] + xyz[1]*4 + xyz[2] *16 for xyz in xyzs])
        return res


    def make_line_xyz(self):
        #レシピ
        #xy0平面を作成 -> xyi平面作成 -> yz平面 -> xz平面作成 ->最後に斜め２ついれて終わり。
        line_xyz = []
        for z in range(4):
            #xy平面作成
            for y in range(4):
                line_xyz.append([[i,y,z] for i in range(4)])
     
            for x in range(4):
                line_xyz.append([[x,i,z] for i in range(4)])

            line_xyz.append([[i,i,z] for i in range(4)])     
            line_xyz.append([[i,3-i,z] for i in range(4)])


        line_xy = line_xyz[:] #xy平面で揃っているやつ

        #yzのpermutation xzであっているやつ
        for xyzs in line_xy :
            #xyzs = [[0,0,0],[0,0,1],...]
            line_xyz.append([[p[1],p[2],p[0]] for p in xyzs])
            line_xyz.append([[p[2],p[0],p[1]] for p in xyzs])


        #斜め上方向
        line_xyz.append([[i,i,i] for i in range(4)])     
        line_xyz.append([[i,3-i,i] for i in range(4)])
        line_xyz.append([[i,i,3-i] for i in range(4)])     
        line_xyz.append([[3-i,i,i] for i in range(4)])

        return line_xyz


        #注意lineの方向は考慮していない。




    # RL_Glueの設定を行う。
    def env_init(self):
        # OBSERVATONS INTS = 盤の状態 (0 ~ 2 の値が 4*4*4次元)  ex) (0,0,0,1,2,1,2,0,0)
        # ACTIONS INTS = ○を打つ場所を指定 (0 ~ 4*4*4)
        # REWARDS = 報酬 (-1.0 ~ 1.0)   ex) 勝 1, 引分 -0.5, 負 -1
        return 'VERSION RL-Glue-3.0 PROBLEMTYPE episodic DISCOUNTFACTOR 0.99 OBSERVATIONS INTS (9 0 1) ACTIONS INTS (0 8) REWARDS (-1.0 1.0)'

    # Episodeの開始
    def env_start(self):
        # 盤面を初期化
        self.map = [0] * self.n_rows * self.n_cols * self.n_heights

        # 盤の状態を保持し、最後に確認するためのリスト
        self.history = []

        current_map = ''
        for i in range(0, len(self.map), self.n_cols):
            current_map += ' '.join(map(str, self.map[i:i+self.n_cols])) + '\n'
        self.history.append(current_map)

        # 盤の状態をRL_Glueを通してエージェントに渡す
        observation = Observation()
        observation.intArray = self.map

        return observation

    def get_free_top_of_map(self):
                #1番上が空いていたらfree
        free_top = []

        for i in range(16):
            i_top = i + 16*3
            if self.map[i_top] == self.flg_free:
                free_top.append(i)
        return(free_top)

    def get_drop_ball_point(self,i_top):
        for z in range(4):            
            if self.map[i_top+ z * 16] == self.flg_free:
               return(i_top+ z * 16)

    def env_step(self, action):

        #まずはエージェントさんに勝敗を告げる。
        # エージェントから受け取った○を打つ場所
      
        int_action_agent = self.get_drop_ball_point(action.intArray[0])

        # 盤に○を打ち、空白の個所を取得する
        self.map[int_action_agent] = self.flg_agent #これが盤面

        free_top = self.get_free_top_of_map()

        #free = [i for i, v in enumerate(self.map) if v == self.flg_free]
        n_free = len(free_top)
      
        rot = Reward_observation_terminal()
        rot.r = 0.0
        rot.terminal = False

        # ○を打った後の勝敗を確認する
        for line in self.lines:
            state = np.array(self.map)[line]

            point = sum(state == self.flg_agent)
  
            if point == self.n_rows:
                rot.r = self.r_win
                rot.terminal = True
                break

            point = sum(state == self.flg_env)

            if point == self.n_rows:
                rot.r = self.r_lose
                rot.terminal = True
                break

        # 勝敗がつかなければ、×を打つ位置を決める

        if not rot.terminal:
            # 空白がなければ引き分け
            if n_free == 0:
                rot.r = self.r_draw
                rot.terminal = True
            else:
                int_action_env = None

                # 空白が1個所ならばそこに×を打つ
                if n_free == 1:
                    int_action_env = self.get_drop_ball_point(free_top[0])
                    rot.terminal = True
                else:
                    # ×の位置を決定する 75%
                    if np.random.rand() < self.opp:

                        #勝てそうなら勝ちに行く。
                        #todo アルゴリズム変更。n_free回打ってみてチェック。

                        for line in self.lines:
                            state = np.array(self.map)[line]
                            point = sum(state == self.flg_env) #環境さん

                            if point == self.n_rows - 1:#環境さんが勝ちそう!

                                index = np.where(state == self.flg_free)[0]

                                if len(index) != 0:
                                    want_to_put = line[index[0]]
                                    i_top = want_to_put % 16 #上から落としてみて起きたい場所におけるか？
                                    if(want_to_put == self.get_drop_ball_point(i_top)):
                                       int_action_env = want_to_put
                                       break

                        #負けそうなら回避する。

                        #todo アルゴリズム変更。負ける箇所が複数なら負けを宣言。


                        if int_action_env is None:
                            for line in self.lines:
                                state = np.array(self.map)[line]
                                point = sum(state == self.flg_agent) #エージェントさん

                                if point == self.n_rows - 1:
                                    index = np.where(state == self.flg_free)[0]
                                    if len(index) != 0:
                                        want_to_put = line[index[0]]
                                        i_top = want_to_put % 16 #上から落としてみて起きたい場所におけるか？
                                        if(want_to_put == self.get_drop_ball_point(i_top)):
                                           int_action_env = want_to_put
                                           break


                                        int_action_env = line[index[0]]
                                        break

                    # ×の位置をランダムに決定する 25%
                    if int_action_env is None:
                        int_action_env = self.get_drop_ball_point(free_top[np.random.randint(n_free)])

                # 盤に×を打つ
                self.map[int_action_env] = self.flg_env #このままでいい。

                free_top = self.get_free_top_of_map() #0の箇所を探索している。
                n_free = len(free_top)

                # ×を打った後の勝敗を確認する
                for line in self.lines:
                    state = np.array(self.map)[line]

                    point = sum(state == self.flg_agent)

                    if point == self.n_rows:
                        rot.r = self.r_win
                        rot.terminal = True
                        break

                    point = sum(state == self.flg_env)

                    if point == self.n_rows:
                        rot.r = self.r_lose
                        rot.terminal = True
                        break

                if not rot.terminal and n_free == 0:
                    rot.r = self.r_draw
                    rot.terminal = True

        # 盤の状態と報酬、決着がついたかどうか をまとめて エージェントにおくる。
        observation = Observation()
        observation.intArray = self.map
        rot.o = observation

        current_map = 'map\n'
        for i in range(0, len(self.map), self.n_cols):
            current_map += ' '.join(map(str, self.map[i:i+self.n_cols])) + '\n'
            if(i % 16 == 0):
                current_map += "\n"

        self.history.append(current_map)

        if rot.r == -1:
            f = open('history.txt', 'a')
            history = '\n'.join(self.history)
            f.writelines('# START\n' + history + '# END\n\n')
            f.close()

        # 決着がついた場合は agentのagent_end
        # 決着がついていない場合は agentのagent_step に続く
        return rot

    def env_cleanup(self):
        pass

    def env_message(self, message):
        pass


if __name__ == '__main__':
    EnvironmentLoader.loadEnvironment(MarubatsuEnvironment())
