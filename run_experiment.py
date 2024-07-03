import os
from dataset import *
from model import CSD, get_stats
import numpy as np
import pandas as pd
import json
import time
from dataclasses import asdict
from tqdm import tqdm
from utils import Timer
import argparse

#### argparse ####
parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
  return v.lower() in ('true', '1')

def float_or_none(value):
    try:
        return float(value)
    except:
        return None

# Model parameters
model_arg = add_argument_group('Model')
model_arg.add_argument('--seed', type=int, default=123, help='random seed')
model_arg.add_argument('--root', type=str, default=None, help='root directory to save results')
model_arg.add_argument('--out_dir', type=str, default='test', help='output directory to be created')
model_arg.add_argument('--nExp', type=int, default=10, help='number of experiments with the same parameter')
model_arg.add_argument('--nOD', type=int, default=100, help='number of OD pairs')
model_arg.add_argument('--nRS', type=int, default=100, help='number of RS pairs')
model_arg.add_argument('--nDrv', type=int, default=50000, help='number of drivers')
model_arg.add_argument('--nTsk', type=int, default=100000, help='number of tasks')
model_arg.add_argument('--theta', type=float, default=1.0, help='theta')
model_arg.add_argument('--cbarR', type=float, default=3.0, help='ratio of operation cost with respect to sp cost')
model_arg.add_argument('--fp_tol', type=float, default=1e-5, help='tolerance for master problem')
model_arg.add_argument('--paramName', nargs='+', type=str, default=['nOD'], help='name of parameter to change in experiment')
model_arg.add_argument('--paramVals', nargs='+', type=float, default=[10], help='parameter values to change in experiment')
model_arg.add_argument('--od_weight', type=bool, default=False, help='if sample OD pairs with weights or not')
model_arg.add_argument('--rs_weight', type=bool, default=False, help='if sample RS pairs with weights or not')


def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed


if __name__ == '__main__':
    config, _ = get_config()
    timer = Timer()
    np.random.seed(config.seed)

    # output directory
    if config.root is not None:
        out_dir = os.path.join(config.root, "results", config.out_dir)
    else:
        out_dir = os.path.join("results", config.out_dir)
    
    try:
        os.makedirs(out_dir, exist_ok = False)
    except:
        out_dir += '_' + time.strftime("%Y%m%dT%H%M")
        os.makedirs(out_dir, exist_ok = False)

    # save config
    with open(os.path.join(out_dir, "config.json"), mode="w") as f:
            json.dump(config.__dict__, f, indent=4)

    # parameter setting
    nOD = config.nOD #100
    nRS = config.nRS #100
    nDrv = config.nDrv #50000
    nTsk = config.nTsk #50000
    theta = config.theta #1
    cbarR = config.cbarR #2.5
    base = {"nOD": nOD, "nRS": nRS, "nDrv": nDrv, "nTsk": nTsk, "theta": theta, "cbarR": cbarR}

    for val_ in config.paramVals:
        param_ = setparam_(nOD, nRS, nDrv, nTsk, theta, cbarR)
        # for name in config.paramName:
        name = config.paramName[0]
        if type(base[name]) == int:
            param_.__dict__[name] = int(val_)
        elif type(base[name]) == float:
            param_.__dict__[name] = float(val_)
        if name == 'nDrv':
            param_.__dict__['nTsk'] = int(2*val_)
        print(param_)

        # set data
        link_path = 'data/Winnipeg/link.csv'
        od_path = 'data/Winnipeg/od.csv'
        cSP_load = np.load('data/Winnipeg/cSP.npy')
        dataset = Dataset(link_path, od_path, cSP_load, od_weight=config.od_weight, rs_weight=config.rs_weight)
        expData = dataset.generate_data(param_, N = config.nExp)

        # define model
        csd = CSD(solver='gurobi', nThrd=32, msg=False, solve_dual=True, accuracy=config.fp_tol)
        csd.load_data(param_, expData)

        # solve LP
        recordsLP, solLP = csd.solve_SOP_lp()
        dfResLP = pd.DataFrame(recordsLP)
        # recSOD, solSOD = csd.solve_SOD_bfgs()
        # for n in range(config.nExp):
        #     solLP[n].update(**{"lambda": solSOD[n]["lambda"], "wage": solSOD[n]["wage"]})

        # solve naive LP
        # recordsNLP, solNLP = csd.solve_SOP_naive_lp()
        # dfResNLP = pd.DataFrame(recordsNLP)

        # solve FP
        recordsFP, solFP = csd.solve_fluid_particle(vcg=False)
        dfResFP = pd.DataFrame(recordsFP)

        # evaluate metrics
        # metrics_nlp = csd.compare_w_naive(recordsLP, solLP, recordsNLP, solNLP)
        metrics_fp = csd.compare_results(recordsLP, solLP, recordsFP, solFP)
        # dfMetrics_nlp = pd.DataFrame(metrics_nlp)
        dfMetrics_fp = pd.DataFrame(metrics_fp)

        # save results
        file_path = os.path.join(out_dir, f"res_{str(val_)}.csv")
        # dfRes = pd.concat([dfResLP, dfResNLP, dfResFP, dfMetrics_fp, dfMetrics_nlp], axis=1)
        dfRes = pd.concat([dfResLP, dfResFP, dfMetrics_fp], axis=1)
        for name in config.paramName:
            dfRes[name] = val_
        print(dfRes)
        dfRes.to_csv(file_path, index=False)

        # save param_ as json
        with open(os.path.join(out_dir, f"param_{str(val_)}.json"), mode="w") as f:
            json.dump(asdict(param_), f)
