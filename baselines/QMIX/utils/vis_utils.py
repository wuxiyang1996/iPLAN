import os
import json
import struct
import io
import numpy as np

# to read rmappo files
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorboard.compat.proto.event_pb2 as event_pb2  # to read all other files


def newest(path):
    '''Returns full path of newest file in given path'''
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def read(data):
    header = struct.unpack('Q', data[:8])
    event_str = data[12:12 + int(header[0])]  # 8+4
    data = data[12 + int(header[0]) + 4:]
    return data, event_str


def get_tb_stats(logdir, desired_tag):
    """Gets newest events file from logdir and returns specified statistic and timesteps"""
    try:
        logpath = newest(logdir)
        with open(logpath, 'rb') as f:
                        data = f.read()
    except FileNotFoundError:
        print('Unable to find log file in ', logdir)
        return

    steps, tag_values = [], []
    while data:
        data, event_str = read(data)
        event = event_pb2.Event()

        event.ParseFromString(event_str)
        if event.HasField('summary'):
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    if value.tag == desired_tag:
                        steps.append(event.step)
                        tag_values.append(value.simple_value)
    return {"ts": steps,
            desired_tag: tag_values
            }


def get_mappo_tb_stats(logdir, desired_tag):
    it = EventAccumulator(logdir).Reload()
    events = it.Scalars(desired_tag)
    returns = np.array([e.value for e in events])
    ts = np.array([e.step for e in events])

    return {
            "ts": ts,
            desired_tag: returns
            }

def get_sacred_stats(logdir, desired_tag):
    f = open(os.path.join(logdir, 'info.json'))
    data = json.load(f)
    ts = data["test_ippo_battle_won_mean_T"]
    returns = data["test_ippo_battle_won_mean"]

    return {
            "ts": ts,
            desired_tag: returns
            }

def combine_tb_logs(logdirs,
                    stat_name="test_ippo_battle_won_mean",
                    ts_round_base=None,
                    results_for_sns=True):
    '''
    Reads tensorboard stats from multiple log files and averages them. 
    All log files should have same number of evaluations at approximately the same times

    logdirs: a list of string paths, WITHOUT the events file specified. Newest events file will be read from 
             specified path by default. 
    stat_name: name of statistic to be read from the events file
    ts_round_base: stats are logged after the episode ends, so the statistic logging timestep does not match the log_interval 
                   specifid in the .yaml, but is close. If not none, this function will round the ts to the nearest ts_round_base. 
    '''
    all_results_dict = {}
    ts, max_ts = [], 0
    for i, logdir in enumerate(logdirs):
        if "rmappo" in logdir:
            res_dict = get_mappo_tb_stats(logdir, desired_tag=stat_name)
        elif "sacred" in logdir:
            res_dict = get_sacred_stats(logdir, desired_tag=stat_name)
        else:
            res_dict = get_tb_stats(logdir, desired_tag=stat_name)
        all_results_dict[f"run_{i}"] = res_dict[stat_name]

        ts_list = res_dict["ts"]
        if len(ts_list) >= max_ts:
            ts = np.array(ts_list)
            max_ts = len(ts_list)

    if ts_round_base is not None:
        # round a_number to the nearest base
        def nearest_multiple(a_number, base): return base * \
            round(a_number / base)
        ts = np.array([nearest_multiple(step, ts_round_base) for step in ts])
    all_results_dict["ts"] = ts

    if not results_for_sns:
        results_all = np.array(
            [all_results_dict[f"run_{i}"] for i in range(len(logdirs))])
        mean_results, std_results = np.mean(
            results_all, axis=0), np.std(results_all, axis=0)
        return (all_results_dict["ts"], mean_results, std_results)

    return all_results_dict
