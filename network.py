import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class policy_net(nn.Module):
    def __init__(self, game, args):
        super(policy_net, self).__init__()
        self.args=args
        self.game=game
        self.num_nodes=game.num_nodes
        self.time_horizon=game.time_horizon
        self.num_defender=game.num_defender
        self.obsn_embedding = nn.Embedding(
            self.num_nodes+1, args.embedding_dim, padding_idx=0)
        self.obs_f1 = nn.Linear((self.num_defender+1)*args.embedding_dim+self.time_horizon+1, args.hidden_dim)
        self.obs_f2=nn.Linear(args.hidden_dim, args.hidden_dim)
    
        self.actn_embedding = nn.Embedding(
            self.num_nodes+1, args.embedding_dim, padding_idx=0)
        self.act_f1=nn.Linear(args.embedding_dim, args.hidden_dim)
        self.act_f2 = nn.Linear(args.hidden_dim, args.hidden_dim)

        self.init_weights()
        self.to(args.device)

    def forward(self, obs, legal_act):
        with torch.no_grad():
            #copy_obs=copy.deepcopy(obs)
            attacker_his, defender_position = obs
            if isinstance(defender_position, tuple):
                defender_position=list(defender_position)
            attacker_position=attacker_his[-1]

            defender_position.append(attacker_position)
            obs = torch.LongTensor(defender_position).to(self.args.device)
            obs=self.obsn_embedding(obs).flatten()

            t=F.one_hot(torch.tensor(len(attacker_his)-1), num_classes=self.time_horizon+1).to(self.args.device).float()
            obs=torch.cat([obs,t])
            obs=self.obs_f1(obs)
            obs=self.obs_f2(F.relu(obs)) # [hidden_dim]

            act = torch.LongTensor(legal_act).to(self.args.device)
            act=self.actn_embedding(act) # [n_legal_act, embedding_dim]
            act=self.act_f1(act)
            act = self.act_f2(F.relu(act))  # [n_legal_act, hidden_dim]

            logits=torch.matmul(act, obs)
            return F.softmax(logits, dim=-1)

    def batch_forward(self, def_pos, att_pos, time, legal_act):
        batch_size = def_pos.shape[0]
        num_player = legal_act.shape[1]
        obs = torch.cat([def_pos, att_pos], dim=-1).long()
        obs = self.obsn_embedding(obs).view(batch_size, -1)
        t = F.one_hot(time,
                      num_classes=self.time_horizon+1).float()
        obs = torch.cat([obs, t], dim=-1)
        obs = self.obs_f1(obs)
        obs = self.obs_f2(F.relu(obs))  # [batch, hidden_dim]
        obs = obs.unsqueeze(1).repeat(1, num_player, 1)
    
        act=self.actn_embedding(legal_act.long()) # [batch, num_player, max_act, embedding_dim]
        act=self.act_f1(act)
        act = self.act_f2(F.relu(act))  # [batch, num_player, max_act, hidden_dim]
        
        logits = torch.bmm(act.view(batch_size*num_player, -1, self.args.hidden_dim),
                           obs.view(batch_size*num_player, self.args.hidden_dim).unsqueeze(-1))
        return logits.squeeze()  # [batch*num_player, max_act]

    def init_weights(self):
        for param in self.parameters():
            nn.init.normal_(param, 0.0, 0.1)

class value_net(nn.Module):
    def __init__(self, game, args):
        super(value_net, self).__init__()
        self.args=args
        self.game=game
        self.num_nodes = game.num_nodes
        self.time_horizon = game.time_horizon
        self.num_defender = game.num_defender
        self.obsn_embedding = nn.Embedding(
            self.num_nodes+1, args.embedding_dim, padding_idx=0)
        self.obs_f1 = nn.Linear(
            (self.num_defender+1)*args.embedding_dim+self.time_horizon+1, args.hidden_dim)
        self.obs_f2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.obs_f3=nn.Linear(args.hidden_dim, 1)

        self.to(args.device)

    def forward(self, obs):
        with torch.no_grad():
            attacker_his, defender_position = obs
            assert defender_position, tuple
            defender_position = list(defender_position)
            attacker_position = attacker_his[-1]

            defender_position.append(attacker_position)
            obs = torch.LongTensor(defender_position).to(self.args.device)
            obs = self.obsn_embedding(obs).flatten()

            t = F.one_hot(torch.tensor(len(attacker_his)-1),
                        num_classes=self.time_horizon+1).to(self.args.device).float()
            obs = torch.cat([obs, t])
            obs = self.obs_f1(obs)
            obs = self.obs_f2(F.relu(obs))  # [batch, hidden_dim]
            obs = self.obs_f3(F.relu(obs))
            return torch.sigmoid(obs)

    def batch_forward(self, def_pos, att_pos, time):
        batch_size=def_pos.shape[0]
        obs = torch.cat([def_pos, att_pos], dim=-1).long()
        obs = self.obsn_embedding(obs).view(batch_size,-1)
        t = F.one_hot(time,
                      num_classes=self.time_horizon+1).float()
        obs = torch.cat([obs, t], dim=-1)
        obs = self.obs_f1(obs)
        obs = self.obs_f2(F.relu(obs))  # [hidden_dim]
        logis = self.obs_f3(F.relu(obs))
        return logis # [batch, 1]

class q_net(nn.Module):
    def __init__(self,game, args):
        super(q_net, self).__init__()
        self.args = args
        self.game = game
        self.num_nodes = game.num_nodes
        self.time_horizon = game.time_horizon
        self.num_defender = game.num_defender
        
        self.embedding = nn.Embedding(
            self.num_nodes+1, args.embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(args.embedding_dim, args.hidden_dim)
        self.fc1=nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, self.num_nodes+1)
        self.to(args.device)
    
    def forward(self, obs):
        attacker_history, position = zip(*obs)
        feature = F.relu(self.encoder_forward(attacker_history))
        output = self.fc1(feature)
        output = self.fc2(output)
        return output

    def encoder_forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = list(inputs)
        if not hasattr(self, '_flattened'):
            self.encoder.flatten_parameters()
            setattr(self, '_flattened', True)
        lengths = torch.tensor([len(n) for n in inputs],
                               dtype=torch.long, device=self.args.device)

        inputs = [torch.LongTensor(k).to(self.args.device) for k in inputs]
        inputs = pad_sequence(inputs, batch_first=True,
                              padding_value=0).detach()
        inputs.requires_grad = False

        inputs = self.embedding(inputs).permute(1, 0, 2)
        packed = nn.utils.rnn.pack_padded_sequence(
            inputs, lengths, enforce_sorted=False)
        out, _ = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        idx = (lengths-1).view(-1, 1).expand(len(lengths),
                                             out.size(2)).unsqueeze(0)
        out = out.gather(0, idx).squeeze()
        return out

class pr_net(nn.Module):
    def __init__(self,game, args):
        super(pr_net, self).__init__()
        self.args=args
        self.game=game
        self.policy_net=policy_net(game, args)
        self.value_net=value_net(game, args)

    def state_value(self, obs):
        return self.value_net(obs).item()

    def prior_pol(self, obs, legal_act): # output should be a distribution over legal_act
        return self.policy_net(obs, legal_act).cpu().numpy()

class dy_net(nn.Module):
    def __init__(self,game, args):
        super(dy_net, self).__init__()
        self.args=args
        self.game=game
        self.policy_net=policy_net(game, args)

    def predict(self, obs, legal_act):
        probs = self.policy_net(obs, legal_act).cpu().numpy()
        return np.random.choice(legal_act, p=probs)    
