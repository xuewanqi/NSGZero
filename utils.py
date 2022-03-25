from collections import defaultdict
import logging
import numpy as np
import os
import json
import random 

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def print_recent_stats(self):
        log_str = "Recent Stats | Episode: {:>8}\n".format(
            self.stats["episodes"][-1][0])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episodes":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            item = "{:.4f}".format(
                np.mean([x[1] for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)

import time

def time_left(start_time, t_start, t_current, t_max):
    if t_current >= t_max:
        return "-"
    time_elapsed = time.time() - start_time
    t_current = max(1, t_current)
    time_left = time_elapsed * (t_max - t_current) / (t_current - t_start)
    # Just in case its over 100 days
    time_left = min(time_left, 60 * 60 * 24 * 100)
    return time_str(time_left)


def time_str(s):
    """
    Convert seconds to a nicer string showing days, hours, minutes and seconds
    """
    days, remainder = divmod(s, 60 * 60 * 24)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    string = ""
    if days > 0:
        string += "{:d} days, ".format(int(days))
    if hours > 0:
        string += "{:d} hours, ".format(int(hours))
    if minutes > 0:
        string += "{:d} minutes, ".format(int(minutes))
    string += "{:d} seconds".format(int(seconds))
    return string



def sliding_mean(data_array, window=5):
    """Sliding average"""
    new_list = []
    for i in range(len(data_array)):
        indices = range(max(i - window + 1, 0),
                        min(i + window + 1, len(data_array)))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return new_list

def get_file_path(data_path):
    files = os.listdir(data_path)
    file_path=[os.path.join(data_path,_dir) for _dir in files]
    return file_path


def raw_data_loader(file_path_list, clip, metric):
    data = []
    for file_path in file_path_list:
        try:
            with open(os.path.join(file_path, 'info.json')) as f:
                d = json.load(f)

                if metric not in d.keys():
                    continue

                # if len([_d['value'] if isinstance(_d, dict) else _d for _d in d[metric]]) < clip:
                #     continue

        except FileNotFoundError:
            print(f'FileNotFoundError: {file_path}')
            continue
        except json.JSONDecodeError:
            print(f'Error, path: {file_path}')
            continue

        data.append(d)
    if len(data) == 0:
        raise ValueError(
            f'data is empty, file_path_list: {file_path_list}, clip: {clip}, metric: {metric}')
    return data


def raw_data_processor(data_list, align=True, sliding_mean_config=None, metric='adv_eval_battle_won_mean'):
    _processed_data, processed_data = [], []
    min_len = 1000000000  # a very large int
    threshold = 1

    for data in data_list:
        if metric == 'test_return_mean':
            data[metric] = [d['value'] if isinstance(
                d, dict) else d for d in data[metric]]

        if len(data[metric]) < threshold:
            continue

        _processed_data.append(data[metric])
        if len(data[metric]) <= min_len:
            min_len = len(data[metric])

    if align:
        for data in _processed_data:
            processed_data.append(data[:min_len])
    else:
        processed_data = _processed_data

    if sliding_mean_config['sliding']:
        for i, data in enumerate(processed_data):
            processed_data[i] = sliding_mean(
                data, window=sliding_mean_config['window_size'])

    #print(f'# seeds: {len(processed_data)}')
    return processed_data


def get_mean_std_for_each_algo(data_path,
                               metric='test_return',
                               clip=0,
                               sliding_mean_config={
                                   'sliding': True, 'window_size': 5},
                               return_processed_data=False,
                               use_median=False):
    file_path = get_file_path(data_path)

    raw_data = raw_data_loader(
        file_path_list=file_path, clip=clip, metric=metric)
    processed_data = raw_data_processor(
        data_list=raw_data, align=True, metric=metric, sliding_mean_config=sliding_mean_config)

    algo_data = np.array(processed_data) * (100 if metric ==
                                            'test_battle_won_mean' else 1)  # percent %
    xticks = True if metric == 'adv_eval_battle_won_mean_T' else False
    if not xticks:
        if use_median:
            algo_data_mean = np.median(algo_data, axis=0)
            algo_data_q75, algo_data_q25 = np.percentile(
                algo_data, [75, 25], axis=0)
        else:
            algo_data_mean = np.mean(algo_data, axis=0)
            algo_data_std = np.std(algo_data, axis=0)

        if clip != 0:
            if use_median:
                algo_data_mean, algo_data_q75, algo_data_q25 = algo_data_mean[
                    :clip], algo_data_q75[:clip], algo_data_q25[:clip]
            else:
                algo_data_mean, algo_data_std = algo_data_mean[:clip], algo_data_std[:clip]

        if return_processed_data:
            if use_median:
                return algo_data_mean, algo_data_q75, algo_data_q25, processed_data
            return algo_data_mean, algo_data_std, processed_data

        return algo_data_mean, algo_data_std
    else:
        x_mean = np.mean(algo_data, axis=0)
        return x_mean


class ReplayBuffer(object):
    def __init__(self, replay_buffer_capacity):
        self._replay_buffer_capacity = replay_buffer_capacity
        self._data = []
        self._next_entry_index = 0

    def add(self, element):
        if len(self._data) < self._replay_buffer_capacity:
            self._data.append(element)
        else:
            self._data[self._next_entry_index] = element
            self._next_entry_index += 1
            self._next_entry_index %= self._replay_buffer_capacity

    def sample(self, num_samples):
        if len(self._data) < num_samples:
            raise ValueError("{} elements could not be sampled from size {}".format(
                num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)
