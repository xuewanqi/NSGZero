import copy
from itertools import product

class Game:
    def __init__(self, graph):
        self.graph=graph
        self.nx_graph=graph.graph
        self.adjlist=graph.adjlist
        self.time_horizon = graph.time_horizon
        self.defender_init = graph.defender_init
        self.attacker_init = graph.attacker_init
        self.exits = graph.exits
        self.num_defender=graph.num_defender
        self.num_nodes=graph.num_nodes
        self.reset()
    
    def reset(self):
        self.current_state = GameState(
            self, copy.deepcopy(self.defender_init), copy.deepcopy(self.attacker_init))
        return self.current_state

    def step(self, defender_act, attacker_act):
        assert isinstance(defender_act, tuple) and isinstance(attacker_act, int)
        # defender_legal_act, attacker_legal_act = self.current_state.legal_action(
        #     combinational=Trues)
        # assert defender_act in defender_legal_act
        # assert attacker_act in attacker_legal_act

        defender_his=copy.deepcopy(self.current_state.defender_his)
        defender_his.append(defender_act)

        attacker_his=copy.deepcopy((self.current_state.attacker_his))
        attacker_his.append(attacker_act)

        self.current_state=GameState(self, defender_his, attacker_his)
        return self.current_state

class GameState:
    def __init__(self, game, defender_his, attacker_his):
        self.defender_his=defender_his
        self.attacker_his=attacker_his

        self.adjlist=game.adjlist
        self.time_horizon = game.time_horizon
        self.defender_init = game.defender_init
        self.attacker_init = game.attacker_init
        self.exits = game.exits
        self.num_defender = game.num_defender

        assert len(defender_his)==len(attacker_his)
        assert len(defender_his) >= 1 and len(defender_his) <= self.time_horizon+1

    def is_end(self, attacker_his=None, defender_position=None):
        if attacker_his is None:
            attacker_his = self.attacker_his
        if defender_position is None:
            defender_position=self.defender_his[-1]
        assert len(self.defender_his)==len(self.attacker_his)
        assert len(attacker_his) <= self.time_horizon+1
        if (len(attacker_his)==self.time_horizon+1) or \
            (attacker_his[-1] in defender_position) or \
              (attacker_his[-1] in self.exits):
            return True
        else:
            return False

    def obs(self):
        defender_obs = (self.attacker_his, self.defender_his[-1])
        attacker_obs = (self.attacker_his, self.defender_his[0])
        return defender_obs, attacker_obs

    def reward(self, attacker_his=None, defender_position=None):
        if not self.is_end(attacker_his, defender_position):
            return 0, 0
        else:
            if attacker_his is None:
                attacker_his = self.attacker_his
            if defender_position is None:
                defender_position = self.defender_his[-1]

            if attacker_his[-1] in defender_position:
                defender_rwd=1
            elif attacker_his[-1] in self.exits:
                defender_rwd=0
            else:
                assert len(attacker_his)==self.time_horizon+1
                defender_rwd=1
            attacker_rwd=-defender_rwd
            return defender_rwd, attacker_rwd

    def legal_action(self, combinational=False, attacker_his=None, defender_position=None):
        if self.is_end(attacker_his, defender_position):
            attacker_legal_act=[0]
            if combinational:
                defender_legal_act=[(0,)*self.num_defender] #[(0, 0, ..)]
            else:
                defender_legal_act=[[0]]*self.num_defender #[[0],[0],..]
        else:
            if attacker_his is None:
                attacker_his = self.attacker_his
            if defender_position is None:
                defender_position = self.defender_his[-1]

            attacker_legal_act = self.adjlist[attacker_his[-1]]
            if combinational:
                defender_legal_act = self._query_legal_defender_actions(
                    defender_position)
            else:
                defender_legal_act=[]
                for i in range(self.num_defender):
                    defender_legal_act.append(self.adjlist[defender_position[i]])
        return defender_legal_act, attacker_legal_act

    def _query_legal_defender_actions(self, current_position):
        # current_position: (position0, position1,...)
        before_combination = []
        for i in range(len(current_position)):
            before_combination.append(self.adjlist[current_position[i]])
        return list(product(*before_combination))

if __name__=='__main__':
    from graph import Graph
    import random

    grid_7=Graph(0)
    game=Game(grid_7)
    combinational_act_spa=False

    print(f'defender obs : {game.current_state.obs()[0]}, defender rwd : {game.current_state.reward()[0]}')
    print(f'attacker obs : {game.current_state.obs()[1]}, attacker rwd : {game.current_state.reward()[1]}')
    while not game.current_state.is_end():
        defender_legal_act, attacker_legal_act = game.current_state.legal_action(
            combinational=combinational_act_spa)
        if combinational_act_spa:
            defender_act=random.choice(defender_legal_act)
        else:
            defender_act=[]
            for i in range(game.num_defender):
                defender_act.append(random.choice(defender_legal_act[i]))
            defender_act=tuple(defender_act)
        attacker_act=random.choice(attacker_legal_act)
        game.step(defender_act, attacker_act)
        print(f'defender obs : {game.current_state.obs()[0]}, defender rwd : {game.current_state.reward()[0]}')
        print(f'attacker obs : {game.current_state.obs()[1]}, attacker rwd : {game.current_state.reward()[1]}')
    print(f'defender his : {game.current_state.defender_his}')
    print(f'attakcer his : {game.current_state.attacker_his}')
