import numpy as np
import math
import copy

eps = 1e-8

class MCTS:
    def __init__(self, game, dy_net, pr_net, args):
        self.game=game
        self.dy_net=dy_net
        self.pr_net=pr_net
        self.args=args
        self.create_func()
        
        self.num_defender=game.num_defender
        self.Ns={}
       
        self.Ls = [{}]  # legal actions at s for defender and attacker
        self.Ps=[]
        self.Qsa=[]
        self.Nsa=[]
        
        for i in range(self.num_defender):
            self.Ls.append({})
            self.Ps.append({})
            self.Qsa.append({})
            self.Nsa.append({})

    def act_prob(self, obs, temp=1):
        for _ in range(self.args.num_sims):
            self.search(obs)
        str_obs=str(obs)
        attacker_his, defender_position = obs
        defender_legal_act, _ = self._legal_action(False, attacker_his, defender_position)
        counts = []
        probs = []
        for i in range(self.num_defender):
            counts.append([self.Nsa[i][(str_obs, str(a))] if (str_obs, str(
                a)) in self.Nsa[i] else 0 for a in defender_legal_act[i]])
            if temp==0:
                bestAs=np.array(np.argwhere(counts[i]==np.max(counts[i]))).flatten()
                bestA=np.random.choice(bestAs)
                p=[0]*len(counts[i])
                p[bestA]=1
                probs.append(p)
            else:
                counts[i] = [x ** (1. / temp) for x in counts[i]]
                counts_sum = float(sum(counts[i]))
                probs.append([x / counts_sum for x in counts[i]])
        return probs
    
    def act_prior_prob(self, obs):
        attacker_his, defender_position = obs
        defender_legal_act, _ = self._legal_action(
            False, attacker_his, defender_position)
        probs=[]
        for i in range(self.num_defender):
            probs.append(self.pr_net.prior_pol(obs, defender_legal_act[i]))
        return(probs)

    def search(self, obs):
        str_obs=str(obs)
        attacker_his, defender_position=obs

        if self._is_end(attacker_his, defender_position):
            return self._reward(attacker_his, defender_position)[0] # reward for defender

        if str_obs not in self.Ns: # leaf node
            self.Ns[str_obs]=0      
            v = self.pr_net.state_value(obs)    
            defender_legal_act, attacker_legal_act = self._legal_action(
                False, attacker_his, defender_position)
            self.Ls[-1][str_obs]=attacker_legal_act
            for i in range(self.num_defender):
                self.Ls[i][str_obs]=defender_legal_act[i]
                self.Ps[i][str_obs] = self.pr_net.prior_pol(
                    obs, defender_legal_act[i])
            return v

        best_act=[]
        for i in range(self.num_defender):
            best_value = -float('inf')
            cur_best_act=0
            defender_legal_act = self.Ls[i][str_obs]
            prior_pol=self.Ps[i][str_obs]
            for idx in range(len(defender_legal_act)):
                a=defender_legal_act[idx]
                str_a=str(a)
                if (str_obs, str_a) in self.Qsa[i]:
                    u = (self.Qsa[i][(str_obs, str_a)]-self.args.bias) + self.args.cpuct*prior_pol[idx] * math.sqrt(self.Ns[str_obs]) / (
                        1 + self.Nsa[i][(str_obs, str_a)])
                else:
                    u = self.args.cpuct*prior_pol[idx] * math.sqrt(self.Ns[str_obs]+eps)
                if u>best_value:
                    best_value=u
                    cur_best_act=a
            assert not cur_best_act==0
            best_act.append(cur_best_act)

        best_act=tuple(best_act)
        attacker_legal_act=self.Ls[-1][str_obs]
        attacker_act=self.dy_net.predict(obs, attacker_legal_act)

        next_attack_his=copy.deepcopy(attacker_his)
        next_attack_his.append(attacker_act)

        next_obs=(next_attack_his,best_act)
        v=self.search(next_obs)
        
        for i in range(self.num_defender):
            a = best_act[i]
            str_a=str(a)
            if (str_obs, str_a) in self.Qsa[i]:
                self.Qsa[i][(str_obs, str_a)] = (self.Nsa[i][(str_obs, str_a)]*self.Qsa[i][(str_obs, str_a)]+v)/(self.Nsa[i][(str_obs, str_a)]+1)
                self.Nsa[i][(str_obs, str_a)]+=1
            else:
                self.Qsa[i][(str_obs, str_a)]=v
                self.Nsa[i][(str_obs, str_a)]=1
        self.Ns[str_obs]+=1
        return v

    def create_func(self):
        self._is_end=self.game.current_state.is_end
        self._reward=self.game.current_state.reward
        self._legal_action=self.game.current_state.legal_action

