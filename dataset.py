import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass
from typing import List
from tqdm import tqdm

@dataclass
class Param:
    nOD: int
    nRS: int
    nDrv: int
    nTsk: int
    theta: float
    cbarR: float

@dataclass
class Data:
    x: np.ndarray # number of tasks for each RS -> T x 1
    q: np.ndarray # number of drivers for each OD -> W x 1
    cDetr: np.ndarray # detour cost -> W x T
    cOpr: np.ndarray # detour cost -> T x 1
    cAtom: dict # atomic detour cost -> {w: np.ndarray of Aod x T}

def setparam_(nOD: int, nRS: int, nDrv: int, nTsk: int,
                theta: float, cbarR: float) -> Param:
    return Param(
        nOD, nRS, nDrv, nTsk, theta, cbarR
    )

class Dataset(object):

    def __init__(self,
            link_path: str = 'data/Winnipeg/link.csv',
            od_path: str = 'data/Winnipeg/od.csv',
            cSP = None,
            od_weight = False,
            rs_weight = False
            ):
        
        # weight for sampling ODs
        self.od_weight = od_weight
        self.rs_weight = rs_weight

        # define network
        self.link_data = pd.read_csv(link_path)
        self.od_data = pd.read_csv(od_path)
        self.link_data['cost'] = self.link_data['free_flow_time']

        # pre-computation of shortest path costs between all possible ODs
        if cSP is not None:
            self.cSP = cSP
        else:
            print("Pre-computation of shortest path costs")
            G = nx.from_pandas_edgelist(self.link_data, source='init_node', target='term_node', edge_attr=['cost'], create_using=nx.DiGraph)
            origins = self.od_data['origin'].unique()
            dests = self.od_data['destination'].unique()
            zones = np.unique(np.append(origins, dests))
            Z = zones.max() + 1
            self.cSP = np.zeros((Z,Z), dtype=np.float64)
            for o in tqdm(zones):
                for d in zones:
                    self.cSP[o,d] = nx.shortest_path_length(G, source=o, target=d, weight='cost')

    def generate_data(self, param_: Param, N: int) -> List[Data]:
        print(f"Generate {N} data for {param_}")
        nOD = param_.nOD
        nRS = param_.nRS
        nDrv = param_.nDrv
        nTsk = param_.nTsk

        # sample datasets
        expData = []
        for n in tqdm(range(N)):
            # sample OD and RS
            od_weights = None if self.od_weight else self.od_data.flow
            rs_weights = None if self.rs_weight else self.od_data.flow
            od = self.od_data.sample(nOD, replace=False, weights=od_weights) # length W = nOD
            rs = self.od_data.sample(nRS, replace=False, weights=rs_weights) # length T = nRS
            od.index = np.arange(nOD)
            rs.index = np.arange(nRS)

            # allocate drivers to OD and tasks to RS: with non-zero constraint
            while True:
                q_p = None if self.od_weight else od.flow / od.flow.values.sum()
                x_p = None if self.rs_weight else rs.flow / rs.flow.values.sum()
                qSample = np.random.choice(np.arange(nOD), nDrv, p=q_p)
                xSample = np.random.choice(np.arange(nRS), nTsk, p=x_p)
                q = np.unique(qSample, return_counts=True)[1]
                x = np.unique(xSample, return_counts=True)[1]
                if (len(q.nonzero()[0]) == nOD) and (len(x.nonzero()[0]) == nRS):
                    break

            # cost: detour cost of (od, rs) = or + rs + sd - od
            # cost matrix
            o_vec = od['origin'].values
            d_vec = od['destination'].values
            r_vec = rs['origin'].values
            s_vec = rs['destination'].values
            cOD = self.cSP[o_vec, d_vec] # W x 1
            cRS = self.cSP[r_vec, s_vec] # T x 1
            cOR = self.cSP[o_vec][:, r_vec] # W x T
            cSD = self.cSP[s_vec][:, d_vec] # T x W

            cDetr = cOR + cRS[np.newaxis, :] + cSD.T - cOD[:, np.newaxis] # W x T
            cOpr = param_.cbarR * cRS # T x 1

            cAtom = {}
            for w in range(nOD):
                qod = q[w]
                Cod = cDetr[w,:]
                cAtom[w] = Cod - np.random.gumbel(scale=1/param_.theta, size=(qod, nRS))

            expData.append(Data(
                x = x,
                q = q,
                cDetr = cDetr,
                cOpr = cOpr,
                cAtom = cAtom
            ))
        return expData

if __name__ == '__main__':
    nOD = 10
    nRS = 10
    nDrv = 10000
    nTsk = 10000
    theta = 1.0
    cbarR = 2.5

    param_ = setparam_(nOD, nRS, nDrv, nTsk, theta, cbarR)
    print(param_)

    dataset = Dataset()
    expData = dataset.generate_data(param_, N=5)
    # print(exData)
