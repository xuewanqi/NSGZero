import agent
from graph import Graph
from game import Game
from execute import *
import os.path
import yaml
from types import SimpleNamespace as SN
import pprint
from utils import *
from multiprocessing import Pipe, Process, set_start_method
import time

def run(_run, _config, _log):
    args = SN(**_config)
    args.ex_results_path = os.path.join(args.ex_results_path, str(_run._id))
    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    if args.use_tensorboard:
        logger.setup_tb(args.ex_results_path)

    # sacred is on by default
    logger.setup_sacred(_run)

    graph = Graph(args.graph_id)
    game = Game(graph)

    defender = agent.MctsDefender(game, args)
    if args.att_type=="random":
        attacker = agent.RandomAttacker(game, args)
    elif args.att_type=="nfsp":
        attacker = agent.NFSPAttacker(game, args)

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    parent_conns, worker_conns = zip(
        *[Pipe() for _ in range(args.num_workers)])
    ps = [Process(target=worker, args=(worker_conn, parent_conn, train_execute_episode, test_execute_episode, game, defender, attacker))
          for (worker_conn, parent_conn) in zip(worker_conns, parent_conns)]
    for p in ps:
        p.daemon = True
        p.start()

    start_time = time.time()
    last_time = start_time
    logger.console_logger.info("Beginning training for {} episodes".format(args.max_episodes))
    
    last_train_e=0
    last_test_e=-args.test_every-1
    last_save_e=0
    last_log_e=0

    e=0
    while e < args.max_episodes:
        # collect trajectories
        for parent_conn in parent_conns:
            parent_conn.send("train_epi")
            if attacker.require_update:
                parent_conn.send((attacker.act_est, attacker.N_acts))
        for parent_conn in parent_conns:
            trajectory = parent_conn.recv()
            defender.add_trajectory(trajectory)
            if attacker.require_update:
                selected_exit, ret, is_br = parent_conn.recv()
                attacker.update(selected_exit, ret)
                if is_br:
                    idx=attacker.exits.index(selected_exit)
                    attacker.cache[idx]+=1
        e+=args.num_workers
        if len(defender.buffer) >= args.train_from and (e-last_train_e) / args.train_every >= 1.0:
            v_loss, def_pre_loss, att_pre_loss = defender.learn()
            logger.log_stat("v_loss", v_loss.item(), e)
            logger.log_stat("def_pre_loss", def_pre_loss.item(),
                            e)
            logger.log_stat("att_pre_loss", att_pre_loss.item(),
                            e)
            last_train_e = e
        if (e-last_test_e) / args.test_every >= 1.0:
            logger.console_logger.info(
                "episodes: {} / {}".format(e, args.max_episodes))
            last_test_e = e

            R = 0
            n = 0
            for _ in range(int(args.test_nepisodes//args.num_workers)):
                for parent_conn in parent_conns:
                    parent_conn.send("test_epi")
                for parent_conn in parent_conns:
                    reward = parent_conn.recv()
                    R += reward
                    n += 1
            R /= n
            logger.log_stat("test_return", R, e)
            
        if args.save_model and (e-last_save_e) / args.save_every >= 1.0:
            
            last_save_e = e
            save_path = os.path.join(
                args.ex_results_path, "models", str(e))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            defender.save_models(save_path)

        if (e-last_log_e) / args.log_every >= 1.0:
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_log_e, e, args.max_episodes), time_str(time.time() - start_time)))
            last_time = time.time()
            logger.log_stat("episodes", e, e)
            logger.print_recent_stats()
            last_log_e=e
            if args.att_type=="nfsp":
                prob = attacker.N_acts/attacker.N_acts.sum()
                prob = np.around(prob, decimals=4)
                logger.console_logger.info(f"Average Prob: {prob}")
                logger.console_logger.info(f"Action Value Est: {attacker.act_est}")
    for parent_conn in parent_conns:
        parent_conn.send("close")

    for p in ps:
        p.join()
