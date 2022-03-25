import random
import networkx as nx
from network import *
from mcts import MCTS
import numpy as np
import copy

eps = 1e-6
class RandomAttacker:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.nx_graph = game.nx_graph
        self.attacker_init = game.attacker_init
        self.exits = game.exits
        self.time_horizon = game.time_horizon
        self.paths = []
        self.path = None
        self.require_update = False

        for e in self.exits:
            self.paths += list(nx.all_simple_paths(self.nx_graph,
                                                   self.attacker_init[0], e, cutoff=self.time_horizon))

    def select_act(self, obs=None):
        assert self.path is not None, 'pls reset a random attacker.'
        act = self.path[self.t]
        self.t += 1
        return act

    def train_select_act(self, obs=None):
        assert self.path is not None, 'pls reset a random attacker.'
        legal_act = self.game.adjlist[self.path[self.t-1]]
        act = self.path[self.t]
        act_idx = legal_act.index(act)
        self.t += 1
        return act, act_idx

    def reset(self, train=None):
        self.t = 1
        self.path = random.choice(self.paths)
        

class NFSPAttacker:
    def __init__(self,game, args):
        self.game = game
        self.args = args
        self.nx_graph = game.nx_graph
        self.attacker_init = game.attacker_init
        self.exits = game.exits
        self.time_horizon = game.time_horizon
        self.paths = {}
        self.path = None
        self.require_update=True

        if args.graph_id >=2:
            for i,e in enumerate(self.exits):
                self.paths[e]=list(nx.all_simple_paths(self.nx_graph,
                         self.attacker_init[0], e, cutoff=self.game.graph.t_cutoff[i]-3))
        else:
            for e in self.exits:
                self.paths[e] = list(nx.all_simple_paths(self.nx_graph,
                                                    self.attacker_init[0], e, cutoff=self.time_horizon))

        self.ban_capacity=args.ban_capacity # bandit capacity
        self._next_idx=0
        self.acts=[]
        self.act_values=[]
        self.n_acts = dict.fromkeys(self.exits, 1) # track the number of actions being chosen recently
        self.act_est = dict.fromkeys(self.exits, 0)

        self.cache_capacity=args.cache_capacity
        self.N_acts=np.ones(len(self.exits))
        self.cache=np.zeros(len(self.exits))

        self.br_rate=args.br_rate

    def update(self, act, val):
        if len(self.acts) < self.ban_capacity:
            self.acts.append(act)
            self.act_values.append(val)
            self.act_est[act] += (val-self.act_est[act])/(self.n_acts[act]+1)
            self.n_acts[act]+=1
        else:
            assert len(self.acts) == self.ban_capacity
            self.act_est[act] += (val-self.act_est[act])/(self.n_acts[act]+1)

            del_act=self.acts[self._next_idx]
            del_val=self.act_values[self._next_idx]
            self.act_est[del_act] -= (del_val-self.act_est[del_act])/(self.n_acts[del_act]-1)
            self.n_acts[del_act]-=1

            self.acts[self._next_idx]=act
            self.act_values[self._next_idx]=val
            self.n_acts[act]+=1

            self._next_idx += 1
            self._next_idx %= self.ban_capacity
        
        if self.cache.sum() >= self.cache_capacity:
            self.N_acts+=self.cache
            self.cache = np.zeros(len(self.exits))

    def select_act(self, obs=None):
        assert self.path is not None, 'pls reset a random attacker.'
        act = self.path[self.t]
        self.t += 1
        return act

    def train_select_act(self, obs=None):
        assert self.path is not None, 'pls reset a random attacker.'
        legal_act = self.game.adjlist[self.path[self.t-1]]
        act = self.path[self.t]
        act_idx = legal_act.index(act)
        self.t += 1
        return act, act_idx

    def reset(self, train=True):
        if np.random.rand() < self.br_rate:
            self.is_br=True
            selected_exit = max(self.exits,
                                key=lambda x: self.act_est[x])
            self.path=random.choice(self.paths[selected_exit])
            if train:
                idx = self.exits.index(selected_exit)
                self.cache[idx]+=1
        else:
            self.is_br=False
            prob = self.N_acts/self.N_acts.sum()
            selected_exit=np.random.choice(self.exits, 1, p=prob).item()
            self.path = random.choice(self.paths[selected_exit])
        self.t=1
        self.selected_exit=selected_exit

    def synch(self, act_est, N_acts):
        self.act_est=act_est
        self.N_acts=N_acts

class SinglePathAttacker(object):
    def __init__(self, path):
        self.path = path
        self.t = None
        self.require_update = False

    def select_act(self, obs=None):
        act = self.path[self.t]
        self.t += 1
        return act

    def reset(self, train=None):
        self.t = 1


class DQNAttacker:
    def __init__(self, game, args):
        self.game=game
        self.args=args
        self.model = q_net(game, args)
        self.target_net = copy.deepcopy(self.model)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()
        from utils import ReplayBuffer
        self.buffer = ReplayBuffer(int(5e5))
        self.opt = torch.optim.RMSprop(
            self.model.parameters(), lr=0.0001)
        self.opt_scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, int(1e4), gamma=0.95)
        # self._legal_action = self.game.current_state.legal_action
        self._step_counter=0
        self._learn_step=0

    def select_act(self, obs, train=True):
        with torch.no_grad():
            legal_act=self._legal_action(obs)
            epsilon = self._get_epsilon(train)
            if np.random.rand() < epsilon:
                act = random.choice(legal_act)  # int
                #idx = legal_act.index(act)
            else:
                q_val = self.model([obs])
                legal_q_val = q_val[legal_act]
                idx = torch.argmax(legal_q_val)
                act = legal_act[idx]
            if train:
                self._step_counter += 1
            return act

    def _legal_action(self, obs):
        attacker_his, defender_position = obs
        legal_act = self.game.adjlist[attacker_his[-1]]
        return legal_act

    def _get_epsilon(self, train, power=1.0):
        if not train:
            return 0.0
        decay_steps = min(self._step_counter, int(1e6))
        decayed_epsilon = (
            0.05 + (0.95 - 0.05) *
            (1 - decay_steps / int(1e6))**power)
        return decayed_epsilon

    def learn(self, transitions):
        obs = [t.obs for t in transitions]
        action = [[t.action] for t in transitions]
        reward = [t.reward for t in transitions]
        next_obs = [t.next_obs for t in transitions]
        #next_legal_actions = [t.next_legal_action for t in transitions]
        next_legal_actions = [ self._legal_action(t.next_obs) for t in transitions]
        is_end = [t.is_end for t in transitions]

        q_vals = self.model(obs)  # shape: (batch, max_actions)
        action = torch.tensor(action, dtype=torch.long, device=self.args.device)
        a_values = torch.gather(
            q_vals, 1, action).flatten()  # shape(batch,)
        next_q_vals = self.target_net(next_obs)
        next_max_q_values = []
        for i in range(len(next_legal_actions)):
            next_max_q_values.append(
                torch.max(next_q_vals[i][next_legal_actions[i]]))
        next_max_q_values = torch.stack(next_max_q_values)
        target_values = torch.Tensor(
            reward).to(self.args.device) + (1 - torch.Tensor(is_end).to(self.args.device)) * 1 * next_max_q_values
        target_values = target_values.detach()
        target_values.requires_grad = False

        loss = F.smooth_l1_loss(a_values, target_values)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.opt.step()
        self.opt_scheduler.step()
        self._learn_step += 1

        if self._learn_step % 1000 == 0:
            self.target_net.load_state_dict(self.model.state_dict())
            self.target_net.eval()

        return loss

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), "{}/BrModel.th".format(save_path))

    def load_model(self, save_path):
        self.model.load_state_dict(torch.load(
            f"{save_path}/BrModel.th", map_location=torch.device(self.args.device)))

    def reset(self):
        pass

class MctsDefender:
    def __init__(self, game, args):
        self.game = game
        self.time_horizon = game.time_horizon
        self.args = args
        self.num_defender = game.num_defender
        self.dy_net = dy_net(game, args)
        self.pr_net = pr_net(game, args)
        self.mcts = MCTS(self.game, self.dy_net, self.pr_net, self.args)
        self._legal_action = self.mcts._legal_action

        self.buffer = []
        self.total_traj = 0
        self.buffer_size = args.buffer_size
        self.v_opt = torch.optim.Adam(
            self.pr_net.value_net.parameters(), lr=self.args.lr*10)
        self.def_pre_opt = torch.optim.Adam(
            self.pr_net.policy_net.parameters(), lr=self.args.lr)
        self.att_pre_opt = torch.optim.Adam(
            self.dy_net.policy_net.parameters(), lr=self.args.lr)
        #self.att_pre_opt = torch.optim.RMSprop(self.dy_net.policy_net.parameters(), lr=self.args.lr)

    def select_act(self, obs, prior=False, temp=1):
        self.mcts = MCTS(self.game, self.dy_net, self.pr_net, self.args)
        copy_obs = copy.deepcopy(obs)
        attacker_his, defender_position = copy_obs
        if prior:
            probs = self.act_prior_prob(copy_obs)
        else:
            probs = self.act_prob(copy_obs, temp)
        act = []
        defender_legal_act, _ = self._legal_action(
            False, attacker_his, defender_position)
        for i in range(self.num_defender):
            act.append(np.random.choice(defender_legal_act[i], p=probs[i]))
        act = tuple(act)
        return act

    def train_select_act(self, obs=None, prior=False):
        self.mcts = MCTS(self.game, self.dy_net, self.pr_net, self.args)
        copy_obs = copy.deepcopy(obs)
        attacker_his, defender_position = copy_obs
        if prior:
            probs = self.act_prior_prob(copy_obs)
        else:
            probs = self.act_prob(copy_obs, self.args.temp)
        act = []
        act_idx = []
        defender_legal_act, attacker_legal_act = self._legal_action(
            False, attacker_his, defender_position)
        for i in range(self.num_defender):
            index = np.arange(len(defender_legal_act[i]))
            index = np.random.choice(index, p=probs[i])
            act_idx.append(index)
            act.append(defender_legal_act[i][index])
        act = tuple(act)
        act_idx = tuple(act_idx)
        return act, act_idx, defender_legal_act, attacker_legal_act

    def act_prob(self, obs, temp=1):
        return self.mcts.act_prob(obs, temp)

    def act_prior_prob(self, obs):
        return self.mcts.act_prior_prob(obs)

    def learn(self, batch_size=64):
        assert len(self.buffer) >= batch_size
        data = np.random.choice(self.buffer, batch_size, replace=False)
        defender_his = []
        attacker_his = []
        defender_his_idx = []
        attacker_his_idx = []
        defender_legal_act = []
        attacker_legal_act = []
        ret = []
        mask = []
        for trajectory in data:
            defender_his.append(trajectory["defender_his"])
            attacker_his.append(trajectory["attacker_his"])
            defender_his_idx.append(trajectory["defender_his_idx"])
            attacker_his_idx.append(trajectory["attacker_his_idx"])
            defender_legal_act.append(trajectory["defender_legal_act"])
            attacker_legal_act.append(trajectory["attacker_legal_act"])
            ret.append(trajectory["return"])
            mask.append(trajectory["mask"])
        defender_his = torch.tensor(
            np.stack(defender_his)).to(self.args.device)
        attacker_his = torch.tensor(
            np.stack(attacker_his)).to(self.args.device)
        defender_his_idx = torch.tensor(
            np.stack(defender_his_idx)).to(self.args.device)
        attacker_his_idx = torch.tensor(
            np.stack(attacker_his_idx)).to(self.args.device)
        defender_legal_act = torch.tensor(
            np.stack(defender_legal_act)).to(self.args.device)
        attacker_legal_act = torch.tensor(
            np.stack(attacker_legal_act)).to(self.args.device)
        ret = torch.tensor(np.stack(ret)).to(self.args.device)
        mask = torch.tensor(np.stack(mask)).to(self.args.device)
        length = mask.sum(dim=1)
        
        v_loss = 0
        def_pre_loss = 0
        att_pre_loss = 0
        for t in range(0, self.time_horizon+1):
            def_pos = defender_his[:, t]  # [batch, num_defender]
            att_pos = attacker_his[:, t]  # [batch, 1]
            time = torch.tensor(t).to(self.args.device).repeat(
                batch_size).detach()  # [batch]
            pre_v = self.pr_net.value_net.batch_forward(def_pos, att_pos, time)
            lable_v = ret[:, t]*(self.args.gamma**(torch.maximum(length-t,torch.zeros_like(length))))
            #lable_v = ret[:, t]
            # v_loss += (mask[:, t] * ((pre_v-lable_v)**2))
            v_loss += mask[:, t]*(-lable_v*torch.log(torch.sigmoid(pre_v)+eps)-(1-lable_v)
                        * torch.log(1-torch.sigmoid(pre_v)+eps))
            # v_loss += (mask[:, t]*F.binary_cross_entropy_with_logits(pre_v, lable_v,  reduction="none"))
                                                                     #weight=torch.tensor([1-self.args.bias]).to(self.args.device), reduction="none"))
            if t==self.time_horizon:
                break

            def_pos = defender_his[:, t]  # [batch, num_defender]
            att_pos = attacker_his[:, t]  # [batch, 1]
            time = torch.tensor(t).to(self.args.device).repeat(
                batch_size).detach()  # [batch]
            def_leg_act = defender_legal_act[:, t+1]
            pre_def_act = self.pr_net.policy_net.batch_forward(
                def_pos, att_pos, time, def_leg_act)
            lable_def_act = defender_his_idx[:, t+1].flatten().long()
            def_mask = mask[:, t+1].repeat(1, self.num_defender).flatten()
            def_pre_loss += (def_mask*F.cross_entropy(pre_def_act,
                                                      lable_def_act, reduction="none"))

            att_leg_act = attacker_legal_act[:, t+1]
            pre_att_act = self.dy_net.policy_net.batch_forward(
                def_pos, att_pos, time, att_leg_act)
            lable_att_act = attacker_his_idx[:, t+1].flatten().long()
            att_pre_loss += (mask[:, t+1].flatten()*F.cross_entropy(
                pre_att_act, lable_att_act, reduction="none"))

        v_loss = (v_loss.mean()/length.mean())
        def_pre_loss = (def_pre_loss.mean() /
                        length.mean()/self.num_defender)
        att_pre_loss = (att_pre_loss.mean()/length.mean())

        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pr_net.value_net.parameters(), 1)
        self.v_opt.step()

        def_pre_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pr_net.policy_net.parameters(), 1)
        self.def_pre_opt.step()

        att_pre_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dy_net.policy_net.parameters(), 1)
        self.att_pre_opt.step()

        return v_loss, def_pre_loss, att_pre_loss

    def add_trajectory(self, trajectory):
        if self.total_traj < self.buffer_size:
            self.buffer.append(trajectory)
            self.total_traj += 1
        else:
            idx = self.total_traj % self.buffer_size
            self.buffer[idx] = trajectory
            self.total_traj += 1

    def save_models(self, save_path):
        torch.save(self.pr_net.state_dict(), "{}/pr_net.th".format(save_path))
        torch.save(self.dy_net.state_dict(), "{}/dy_net.th".format(save_path))

    def load_models(self, save_path):
        self.pr_net.load_state_dict(torch.load(f"{save_path}/pr_net.th", map_location=torch.device(self.args.device)))
        self.dy_net.load_state_dict(torch.load(f"{save_path}/dy_net.th", map_location=torch.device(self.args.device)))

    def reset(self):
        pass


class GreedyDefender:
    def __init__(self, game, args=None):
        self.game=game
        self.nx_graph = game.nx_graph
        self.num_defender = game.num_defender

    def select_act(self, obs):
        attacker_his, defender_position = obs
        attacker_position=attacker_his[-1]
        act=[]
        for i in range(self.num_defender):
            shortest_path = nx.shortest_path(
                self.nx_graph, defender_position[i], attacker_position)
            if len(shortest_path) >= 3:
                action = shortest_path[1]
            else:
                action = shortest_path[0]
            act.append(action)
        return tuple(act)

    def reset(self):
        pass

class UniformDefender:
    def __init__(self, game, args=None):
        self.game=game
        self.num_defender=game.num_defender
        self._legal_action = self.game.current_state.legal_action
    def select_act(self, obs):
        attacker_his, defender_position = obs
        legal_act, _ = self._legal_action(
            False, attacker_his, defender_position)
        act=[]
        for i in range(self.num_defender):
            act.append(np.random.choice(legal_act[i]))
        return tuple(act)
        
    def reset(self):
        pass
