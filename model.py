# %%
from dataset import *
import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
import pulp as pl
from tqdm import tqdm
from utils import Timer
timer = Timer()

def get_stats(d: np.ndarray, dName: str = 'd') -> dict:
    return {
        f'sum_{dName}': np.sum(d),
        f'mean_{dName}': np.mean(d),
        f'max_{dName}': np.max(d),
        f'min_{dName}': np.min(d),
        f'std_{dName}': np.std(d),
        f'nonneg_{dName}': (d >= 0).sum(),
    }

class CSD(object):

    def __init__(self,
            solver: str = 'gurobi', nThrd: int = 1, msg: bool = False,
            accuracy: float = 1e-5, maxIter: int = 25000, solve_dual: bool = False):
        self.eps = 1e-8
        self.f_lb = 5
        self.tol = accuracy
        self.maxIter = maxIter
        self.solver = {
            'cmc': pl.PULP_CBC_CMD(threads = nThrd, msg = msg),
            'gurobi': pl.GUROBI(threads = nThrd, msg = msg),
            'cplex': pl.CPLEX_CMD(threads = nThrd, msg = msg),
        }.get(solver)
        self.solve_dual = solve_dual

    def load_data(self, param_: Param, expData: List[Data]):
        self.nOD = param_.nOD
        self.nRS = param_.nRS
        self.nDrv = param_.nDrv
        self.nTsk = param_.nTsk
        self.theta = param_.theta
        self.cbarR = param_.cbarR
        self.nExp = len(expData)
        self.data = expData

    def solve_SOP_lp(self) -> Tuple[List[dict], List[dict]]:
        """
        Naive method to solve [SO-P]

        Returns:
            Tuple[np.ndarray, float]: optimal solution and optimal value.
        """
        print("Solving matching problem as naive LP")

        records, solutions = [], []
        for n in range(self.nExp):
            # input
            data_ = self.data[n]
            x = data_.x
            q = data_.q
            cDetr = data_.cDetr
            cOpr = data_.cOpr
            cAtom = data_.cAtom
            C = cDetr - cOpr[np.newaxis, :]
            c = list(cAtom.values())
            c = np.concatenate(c, axis=0) # |A| x |RS|
            c_ = c - cOpr[np.newaxis,:]

            # define model
            model = pl.LpProblem(name="matching", sense=pl.LpMinimize)
            # define decision variables
            y = np.array(pl.LpVariable.matrix('y', (range(self.nDrv), range(self.nRS)), lowBound=0, upBound=1))
            # define objective
            model += pl.lpDot(c_.flatten(), y.flatten()), 'Objective'
            # set constraints
            for i in range(self.nDrv):
                model += pl.lpSum(y[i,:]) == 1, f'Supply{i}'
            for j in range(self.nRS):
                model += pl.lpSum(y[:,j]) <= x[j], f'Demand{j}'
            # solve LP
            timer.start()
            result = model.solve(self.solver)
            runtime = timer.stop()
            yArray = np.array([y[i,j].varValue for i in range(self.nDrv) for j in range(self.nRS)]).reshape(self.nDrv, self.nRS)
            qi = np.cumsum(np.append([0], q)).astype(np.int64)
            f = [yArray[qi[i]:qi[i+1],:].sum(axis=0) for i in range(self.nOD)] # list of vectors of (|RS| x 1)
            f = np.vstack(f) # |OD| x |RS|
            p = f / q[:,np.newaxis]
            zopt = -pl.value(model.objective)
            indU = -c_ * yArray # |A| x |RS|
            odU = [indU[qi[i]:qi[i+1],:].sum(axis=0) for i in range(self.nOD)] # list of vectors of (|RS| x 1)
            odU = np.vstack(odU) # |OD| x |RS|
            zopt_od = odU.sum(axis=1)

            # Lagrangian multipliers
            # rho = [model.getConstrByName(f'Supply{i}').Pi for i in range(self.nDrv)]
            lamd = -np.array([model.constraints[f'Demand{j}'].pi for j in range(self.nRS)])
            wage = np.clip(cOpr - lamd, 0., cOpr)
            records.append({
                "data": n,
                "status": pl.LpStatus[result],
                "opt_val": zopt,
                "cpu_time": runtime,
            })
            solutions.append({
                "y": yArray,
                "f": f,
                "p": p,
                "lambda": lamd,
                "wage": wage,
                "opt_val_OD": zopt_od,
            })
            print(f"Problem solved for data {n}. Zopt: {zopt:.3f}; with {runtime:.3f}s")

        return records, solutions

    def solve_SOP_naive_lp(self) -> Tuple[List[dict], List[dict]]:
        """
        Naive mechanism to solve [SO-P] where only deterministic utility is considered

        Returns:
            Tuple[np.ndarray, float]: optimal solution and optimal value.
        """
        print("Solving matching problem as naive LP")

        records, solutions = [], []
        for n in range(self.nExp):
            # input
            data_ = self.data[n]
            x = data_.x
            q = data_.q
            cDetr = data_.cDetr
            cOpr = data_.cOpr
            cAtom = data_.cAtom
            C = cDetr - cOpr[np.newaxis, :] # |OD| x |RS|
            c = {}
            for w, cAod in cAtom.items():
                c[w] = np.tile(C[w,:], (cAod.shape[0],1))
            cflat = list(c.values())
            c_ = np.concatenate(cflat, axis=0) # |A| x |RS|

            # define model
            model = pl.LpProblem(name="matching", sense=pl.LpMinimize)
            # define decision variables
            y = np.array(pl.LpVariable.matrix('y', (range(self.nDrv), range(self.nRS)), lowBound=0, upBound=1))
            # define objective
            model += pl.lpDot(c_.flatten(), y.flatten()), 'Objective'
            # set constraints
            for i in range(self.nDrv):
                model += pl.lpSum(y[i,:]) == 1, f'Supply{i}'
            for j in range(self.nRS):
                model += pl.lpSum(y[:,j]) <= x[j], f'Demand{j}'
            # solve LP
            timer.start()
            result = model.solve(self.solver)
            runtime = timer.stop()
            yArray = np.array([y[i,j].varValue for i in range(self.nDrv) for j in range(self.nRS)]).reshape(self.nDrv, self.nRS)
            qi = np.cumsum(np.append([0], q)).astype(np.int64)
            f = [yArray[qi[i]:qi[i+1],:].sum(axis=0) for i in range(self.nOD)] # list of vectors of (|RS| x 1)
            f = np.vstack(f) # |OD| x |RS|
            p = f / q[:,np.newaxis]
            zopt = -pl.value(model.objective)
            indU = -c_ * yArray # |A| x |RS|
            odU = [indU[qi[i]:qi[i+1],:].sum(axis=0) for i in range(self.nOD)] # list of vectors of (|RS| x 1)
            odU = np.vstack(odU) # |OD| x |RS|
            zopt_od = odU.sum(axis=1)

            # Lagrangian multipliers
            # rho = [model.getConstrByName(f'Supply{i}').Pi for i in range(self.nDrv)]
            lamd = -np.array([model.constraints[f'Demand{j}'].pi for j in range(self.nRS)])
            wage = np.clip(cOpr - lamd, 0., cOpr)
            records.append({
                "data": n,
                "status": pl.LpStatus[result],
                "opt_val": zopt,
                "cpu_time": runtime,
            })
            solutions.append({
                "y": yArray,
                "f": f,
                "p": p,
                "lambda": lamd,
                "wage": wage,
                "opt_val_OD": zopt_od,
            })
            print(f"Problem solved for data {n}. Zopt: {zopt:.3f}; with {runtime:.3f}s")

        return records, solutions

    def solve_SOD_lp(self) -> Tuple[List[dict], List[dict]]:
        """
        Naive method to solve [SO-D]

        Returns:
            Tuple[np.ndarray, float]: optimal solution and optimal value.
        """
        print("Solving dual problem as naive LP")

        records, solutions = [], []
        for n in tqdm(range(self.nExp)):
            # input
            data_ = self.data[n]
            x = data_.x
            q = data_.q
            cDetr = data_.cDetr
            cOpr = data_.cOpr
            cAtom = data_.cAtom
            C = cDetr - cOpr[np.newaxis, :]
            c = list(cAtom.values())
            c = np.concatenate(c, axis=0) # |A| x |RS|
            c_ = cOpr[np.newaxis,:] - c

            # define model
            model = pl.LpProblem(name="matching", sense=pl.LpMinimize)
            # define decision variables
            rho = pl.LpVariable.dicts('rho', range(self.nDrv), cat='Continuous', lowBound=0, upBound=None)
            lamd = pl.LpVariable.dicts('lamd', range(self.nRS), cat='Continuous', lowBound=0, upBound=None)
            # define objective
            model += pl.lpSum(rho) + pl.lpDot(x, lamd), 'Objective'
            # set constraints
            for i in range(self.nDrv):
                for j in range(self.nRS):
                    model += c_[i,j] - lamd[j] <= rho[i]
            # solve LP
            timer.start()
            result = model.solve(self.solver)
            runtime = timer.stop()
            rho = np.array([rho[a].varValue for a in range(self.nDrv)])
            lamd = np.array([lamd[i].varValue for i in range(self.nRS)])
            zopt = pl.value(model.objective)
            records.append({
                "data": n,
                "status": pl.LpStatus[result],
                "opt_val": zopt,
                "cpu_time": runtime,
            })
            solutions.append({
                "rho": rho,
                "lambda": lamd,
            })
            print(f"Problem solved for data {n}. Zopt: {zopt:.3f}; with {runtime:.3f}s")

        return records, solutions

    def solve_SOD_bfgs(self) -> Tuple[List[dict], List[dict]]:
        """
        BFGS method to solve [SO-D]

        Returns:
            Tuple[np.ndarray, float]: optimal solution and optimal value.
        """
        print("Solving dual problem as naive LP")

        records, solutions = [], []
        for n in range(self.nExp):
            # input
            data_ = self.data[n]
            x = data_.x
            q = data_.q
            cDetr = data_.cDetr
            cOpr = data_.cOpr
            cAtom = data_.cAtom
            C = cDetr - cOpr[np.newaxis, :]
            c = list(cAtom.values())
            c = np.concatenate(c, axis=0) # |A| x |RS|
            c_ = cOpr[np.newaxis,:] - c

            # define model
            def f(lamd):
                maxU = np.max(c_ - lamd[np.newaxis,:], axis=1)
                return maxU.sum() + x.dot(lamd)

            # solve
            init_lamd = np.ones(self.nRS, dtype=np.float64)
            bounds = [(0, val_) for val_ in cOpr]
            timer.start()
            res = minimize(f, x0=init_lamd, method="L-BFGS-B", bounds=bounds) #, options={'disp': disp}, callback=callbackF
            runtime = timer.stop()

            # probability
            lamd = res.x
            wage = cOpr - lamd
            # objective value
            z = res.fun

            # return results
            records.append({
                "data": n,
                "opt_val": z,
                "cpu_time": runtime,
            })
            solutions.append({
                "lambda": lamd,
                "wage": wage
            })
            print(f"Problem solved for data {n}. Zopt: {z:.3f}; with {runtime:.3f}s")

        return records, solutions

    def solve_fluid_particle(self, vcg: bool = False) -> Tuple[List[dict], List[dict]]:
        """
        Fluid particle decomposition approach to solve [SO-P]
            - Master Problem [SO-A-P] solved by Bregman's balancing method
            - Sub Problem [SO-Sub/(od)] solved by LP solver

        Returns:
            Tuple[np.ndarray, float]: optimal solution and optimal value.
        """
        print("Solving matching problem with fluid-particle decomposition approach")

        records = []
        solutions = []
        for n in range(self.nExp):
            # input
            data_ = self.data[n]
            x = data_.x
            q = data_.q
            cDetr = data_.cDetr
            cOpr = data_.cOpr
            cAtom = data_.cAtom
            C = cDetr - cOpr[np.newaxis, :]

            # solve master problem
            masterRes, masterSol = self.solve_SOAP_balancing(n, cDetr, cOpr, q, x)
            lamb = masterSol["lambda"]
            # if self.solve_dual:
            #     dualRes, dualSol = self.solve_SOAD_bfgs(n, cDetr, cOpr, q, x)
            #     lamb = dualSol["lambda"]
            wage = np.clip(cOpr - lamb, 0., cOpr)
            masterSol.update(**{"wage": wage, "lambda": lamb})
            # print(wage)
            # print(get_stats(wage, "wage"))

            # solve sub problem
            p = masterSol["p"]
            f = p * q[:,np.newaxis] # no. tasks allocated for each od pair
            y, pi = [], []
            sub_time, sub_count = 0, 0
            zSubs = np.zeros((self.nOD,), dtype=np.float64)
            sigma = np.zeros((self.nOD, self.nRS), dtype=np.float64)
            wageSub = np.zeros((self.nOD, self.nRS), dtype=np.float64)
            for w in range(self.nOD):
                fod = f[w,:].round()
                qod = int(min(q[w], fod.sum()))
                if qod > 0:
                    cAod = cAtom[w][np.arange(qod),:] # qod x |RS| # - cOpr[np.newaxis,:]
                    subRes = self.solve_SO_Sub_lp(w, cAod, cOpr, fod)
                    y.append(subRes["y"])
                    zSubs[w] = subRes["opt_val"]
                    sub_time += subRes["cpu_time"]
                    sub_count += 1
                    sigma[w,:] = subRes["sigma"]
                    wageSub[w,:] = subRes["wage"]
                    # print("sub-LP's z =", subRes["opt_val"])
                    # print("sub-LP's y =", subRes["y"].nonzero())

                    if self.solve_dual:
                        # solve SO-D/sub
                        subRes_D = self.solve_SOD_Sub_bfgs(w, cAod, cOpr, fod, lamb)
                        sigma[w,:] = subRes_D["sigma"]
                        wageSub[w,:] = subRes_D["wage"]

                # VCG mechanism
                if vcg:
                    vcgRes = self.solve_SO_Sub_vcg(w, cAod, cOpr, fod, subRes["opt_val"], subRes["y"])
                    # print(get_stats(vcgRes["piInd"], "pi"))
                    # print(vcgRes["piSum"])
                    # print(vcgRes["piMean"].round(2))
                    pi.append(vcgRes["piMean"])

            # aggregation
            cpu_time = masterRes["cpu_time_master"] + sub_time
            cpu_time_ave = masterRes["cpu_time_master"] + sub_time / sub_count
            masterRes.update(**{"z_sub": zSubs.sum(),
                                "cpu_time_sub": sub_time, "no_sub": sub_count,
                                "cpu_time_fp_total": cpu_time, "cpu_time_fp_ave": cpu_time_ave})
            masterSol.update(**{"y": y, "f": f, "pi": pi, "sigma": sigma, "wageSub": wageSub, "z_sub_OD": zSubs})
            records.append(masterRes)
            solutions.append(masterSol)

            print(f"Problem solved for data {n}. Zopt: {zSubs.sum():.3f}; with {cpu_time:.3f}s (ave. {cpu_time_ave:.3f}s)")

        return records, solutions

    def solve_SOAP_balancing(self, n: int, cDetr: np.ndarray, cOpr: np.ndarray, q: np.ndarray, x: np.ndarray) -> dict:
        """
        Bregman's balancing method for solving the problem [SO-A-P].

        Args:
            n (int): number of dataset
            cDetr (np.ndarray): detour cost matrix |OD| x |RS|
            cOpr (np.ndarray): operation cost matrix |RS| x 1
            q (np.ndarray): vector of no. drivers |OD| x 1
            x (np.ndarray): vector of no. tasks |RS| x 1

        Returns:
            dict: optimal solution
        """

        # solve
        nIter = 0
        K = np.exp(-self.theta * (cDetr - cOpr[np.newaxis, :]))
        u = np.ones(self.nOD, dtype=np.float64)
        v = np.ones(self.nRS, dtype=np.float64)
        timer.start()
        while True:
            nIter += 1
            u_ = 1 / K.dot(v) # for equation (when all drivers participating in tasks)
            if self.nDrv > self.nTsk:
                u_ = np.clip(u_, None, 1.) # for inequality (when some drivers do not participate)
            v_ = x / (K.T.dot(u_ * q)) # for equation (when all tasks delivered by drivers)
            if self.cbarR > 0:
                v_ = np.clip(v_, None, 1.) # for inequality (when some tasks can be delivered by manager)
            uGrad = np.max(np.abs(u - u_)/u) #np.linalg.norm(u - u_, ord=np.inf)
            vGrad = np.max(np.abs(v - v_)/v) #np.linalg.norm(v - v_, ord=np.inf)
            converged = (uGrad < self.tol) * (vGrad < self.tol) + (nIter >= self.maxIter)
            u = u_
            v = v_
            if converged:
                print(f"Balancing method converged for data {n} at {nIter} iter w. udif {uGrad:.5f} and vdif {vGrad:.5f}")
                break
        runtime = timer.stop()

        # probability
        p = np.diag(u).dot(K).dot(np.diag(v)) #np.diag(u) @ K @ np.diag(v)
        lamd = -np.log(v)/self.theta # Lagrangian multiplier
        # objective value
        z = ((cOpr[np.newaxis, :] - cDetr) * p).sum(axis=1) - (1/self.theta) * (p * np.log(p)).sum(axis=1)
        Z = q.dot(z)
        solution = {"p": p, "lambda": lamd}
        res = {
            "data": n,
            "cpu_time_master": runtime,
            "iteration": nIter,
            "u_grad": uGrad,
            "v_grad": vGrad,
            "z_master": Z,
        }
        return res, solution
    
    def solve_SOAD_bfgs(self, n: int, cDetr: np.ndarray, cOpr: np.ndarray, q: np.ndarray, x: np.ndarray) -> dict:
        """
        Solving the problem [SO-A-D] with BFGS.

        Args:
            n (int): number of dataset
            cDetr (np.ndarray): detour cost matrix |OD| x |RS|
            cOpr (np.ndarray): vector of operation cost |RS| x 1
            q (np.ndarray): vector of no. drivers |OD| x 1
            x (np.ndarray): vector of no. tasks |RS| x 1

        Returns:
            dict: optimal solution
        """

        # define objective function
        def f(lamd):
            # expected maximum utility |OD| x 1
            S = np.log(np.sum(np.exp(self.theta * (cOpr[np.newaxis,:] - cDetr - lamd[np.newaxis,:])), axis=1)) / self.theta
            return q.dot(S) + x.dot(lamd)

        # solve
        init_lamd = np.ones(self.nRS, dtype=np.float64)
        bounds = [(0, val_) for val_ in cOpr]
        timer.start()
        res = minimize(f, x0=init_lamd, method="L-BFGS-B", bounds=bounds) #, options={'disp': disp}, callback=callbackF
        runtime = timer.stop()

        # probability
        lamd = res.x
        exp_v = np.exp(self.theta * (cOpr[np.newaxis,:] - cDetr - lamd[np.newaxis,:]))
        p = exp_v / exp_v.sum(axis=1, keepdims=True)
        # objective value
        z = res.fun

        # return results
        solution = {"p": p, "lambda": lamd}
        res = {
            "data": n,
            "cpu_time_master_dual": runtime,
            "z_master_dual": z,
        }
        return res, solution

    def solve_SO_Sub_lp(self, w: int, cAod: np.ndarray, cOpr: np.ndarray, fod: np.ndarray) -> dict:
        """
        Solve [SO/Sub(od)] by LP solver

        Args:
            w (int): id of OD pair
            cAod (np.ndarray): atomic cost matrix |A_od| x |RS|
            cOpr (np.ndarray): operation cost vector |RS| x 1
            fod (np.ndarray): vector of no. tasks |RS| x 1

        Returns:
            dict: optimal solution
        """
        qod = cAod.shape[0]
        c = cAod - cOpr[np.newaxis,:] # qod x |RS|

        # define LP model
        model = pl.LpProblem(name="matching", sense=pl.LpMinimize)
        # define decision variables
        y = np.array(pl.LpVariable.matrix('y', (range(qod), range(self.nRS)), lowBound=0, upBound=1)) #, cat='Binary'
        # define objective
        model += pl.lpDot(c.flatten(), y.flatten()), 'Objective'
        # set constraints
        for i in range(qod):
            model += pl.lpSum(y[i,:]) == 1, f'Supply{i}'
        for j in range(self.nRS):
            model += pl.lpSum(y[:,j]) <= fod[j], f'Demand{j}'
        # solve LP
        timer.start()
        result = model.solve(self.solver)
        runtime = timer.stop()
        yArray = np.array([y[i,j].varValue for i in range(qod) for j in range(self.nRS)]).reshape(c.shape)
        # wage
        sigma = -np.array([model.constraints[f'Demand{j}'].pi for j in range(self.nRS)])
        wage = np.clip(cOpr - sigma, 0., cOpr)
        return {
            "od": w,
            "status": pl.LpStatus[result],
            "opt_val": -pl.value(model.objective), #revert to maximum value
            "y": yArray,
            "cpu_time": runtime,
            "sigma": sigma,
            "wage": wage,
        }

    def solve_SO_Sub_northwest(self, w: int, cAod: np.ndarray, cOpr: np.ndarray, fod: np.ndarray) -> dict:
        """
        Solve [SO/Sub(od)] by Northwest corner rule
        *** This cannot be useful because matrix C does not satisfy Monge property! ***

        Args:
            w (int): id of OD pair
            cAod (np.ndarray): atomic cost matrix |A_od| x |RS|
            cOpr (np.ndarray): operation cost vector |RS| x 1
            fod (np.ndarray): vector of no. tasks |RS| x 1

        Returns:
            dict: optimal solution
        """
        qod = cAod.shape[0]
        c = cOpr[np.newaxis,:] - cAod # qod x |RS|
        
        # prepare variables used in the algorithm
        q = np.ones(qod, dtype=np.int16) # qod x 1
        d = fod.copy() # |RS| x 1
        
        # counters
        a = 0 # counter for driver (supply)
        i = 0 # counter for RS pair (demand)
        # matching pattern
        y = np.zeros((qod, self.nRS), dtype=np.int16)
        timer.start()
        while True:
            # step 1: solution
            y[a,i] = min(q[a], d[i])
            # step 2: update
            q[a] -= y[a,i]
            d[i] -= y[a,i]
            if q[a] == 0: a += 1
            if d[i] == 0: i += 1
            # step 3: check the constraints
            supply_satisfied = (y.sum(axis=1) == 1).all()
            demand_satisfied = (y.sum(axis=0) == fod).all()
            if supply_satisfied and demand_satisfied:
                break
        runtime = timer.stop()

        # evaluate objective function
        z = (c * y).sum()
        return {
            "od": w,
            "opt_val": z,
            "y": y,
            "cpu_time": runtime,
        }
    
    def solve_SOD_Sub_bfgs(self, w: int, cAod: np.ndarray, cOpr: np.ndarray, fod: np.ndarray, lamd: np.ndarray) -> dict:
        """
        Solve [SO-D/Sub(od)] by BFGS method

        Args:
            w (int): id of OD pair
            cAod (np.ndarray): atomic cost matrix |A_od| x |RS|
            cOpr (np.ndarray): operation cost vector |RS| x 1
            fod (np.ndarray): vector of no. tasks |RS| x 1

        Returns:
            dict: optimal solution
        """
        qod = cAod.shape[0]
        c = cAod - cOpr[np.newaxis,:] # qod x |RS|

        # define model
        def f(sigma):
            maxU = np.max((-c - sigma[np.newaxis,:]), axis=1)
            return maxU.sum() + fod.dot(sigma)
        
        # solve
        init_sigma = np.ones(self.nRS, dtype=np.float64) * lamd
        bounds = [(0, val_) for val_ in cOpr]
        timer.start()
        res = minimize(f, x0=init_sigma, method="L-BFGS-B", bounds=bounds) #, options={'disp': disp}, callback=callbackF
        runtime = timer.stop()

        # solution
        sigma = res.x
        sigma = sigma * (fod > 0)
        wage = cOpr - sigma
        z = res.fun
        # print(fod, sigma)

        return {
            "od": w,
            "sigma": sigma,
            "wage": wage,
            "opt_val": z, 
            "cpu_time": runtime,
        }

    def solve_SO_Sub_vcg(self,
        w: int,
        cAod: np.ndarray, cOpr: np.ndarray, fod: np.ndarray,
        zopt: float, yod: np.ndarray) -> dict:
        """
        Solve [SO/Sub(od)] by VCG mechanism

        Args:
            w (int): id of OD pair
            cAod (np.ndarray): atomic cost matrix |A_od| x |RS|
            cOpr (np.ndarray): operation cost vector |RS| x 1
            fod (np.ndarray): vector of no. tasks |RS| x 1
            zopt (float): optimal objective value of sub solved by LP
            yod (np.ndarray): optimal solution of sub solved by LP |A_od| x |RS|

        Returns:
            dict: optimal solution
        """
        print(f"Solving [SO-Sub/({w})] by VCG mechanism")
        qod = cAod.shape[0]
        c = cOpr[np.newaxis,:] - cAod # |Aod| x |RS|
        # individual profit in optimal state
        cy = (c * yod).sum(axis=1) # |Aod| x 1
        # number of tasks except for each individual
        fy = (fod[np.newaxis,:] - yod).astype(np.int64)
        # others' surplus in optimal state
        z_minus = zopt - cy # |Aod| x 1
        z_plus = zopt + cy # |Aod| x 1
        # calculate others' surplus without individual
        z_wo = np.zeros_like(z_minus, dtype=np.float64)
        for a in tqdm(range(qod)):
            cod_a = np.delete(-c.copy(), [a], axis=0) # for minimization
            # define LP model
            model = pl.LpProblem(name="matching", sense=pl.LpMinimize)
            # define decision variables
            y = np.array(pl.LpVariable.matrix('y', (range(qod-1), range(self.nRS)), lowBound=0, upBound=1))
            # define objective
            model += pl.lpDot(cod_a.flatten(), y.flatten()), 'Objective'
            # set constraints
            for i in range(qod-1):
                # sum y = 0 only for i = a
                model += pl.lpSum(y[i,:]) == 1
            for j in range(self.nRS):
                model += pl.lpSum(y[:,j]) <= fod[j]
            # solve LP
            # timer.start()
            result = model.solve(self.solver)
            # runtime = timer.stop()
            z_wo[a] = -pl.value(model.objective) # revert to maximize
        # VCG payment
        piInd = (z_plus - z_wo) # |Aod| x 1 #z_wo - z_minus
        piSum = (piInd[:,np.newaxis] * yod).sum(axis=0) # |RS| x 1
        piMean = piSum / np.clip(yod.sum(axis=0), self.eps, None) # |RS| x 1
        # print(piMean.round(2))
        return {
            "z_opt": zopt,
            "z_minus": z_minus,
            "z_without": z_wo,
            "piInd": piInd,
            "piSum": piSum,
            "piMean": piMean
        }

    def compare_results(self,
            recLP: List[dict], solLP: List[dict],
            recAprx: List[dict], solAprx: List[dict]
            ) -> List[dict]:

        metrics = []
        for n in range(self.nExp):
            nDrvs = self.data[n].q # |OD| x 1
            nTsks = self.data[n].x # |RS| x 1
            cOpr = self.data[n].cOpr # |RS| x 1
            
            # relative error of objective values
            zlp = recLP[n]["opt_val"]
            zlp_od = solLP[n]["opt_val_OD"]
            zfp = solAprx[n]["z_sub_OD"]
            relDif = np.abs((zlp - zfp.sum()) / zlp)
            zDif = self.get_diff(zlp_od, zfp, "zOD")
            # task partition patterns
            f_lp = solLP[n]["f"] # |OD| x |RS|
            f_fp = solAprx[n]["f"] # |OD| x |RS|
            f_fp_int = f_fp.round()
            fLPFP_idxs = (f_lp > self.f_lb) * (f_fp > self.f_lb)
            fLPFPint_idxs = (f_lp > self.f_lb) * (f_fp_int > self.f_lb)
            fDif = self.get_diff(f_lp[fLPFP_idxs], f_fp[fLPFP_idxs], "fLPFP")
            fintDif = self.get_diff(f_lp[fLPFPint_idxs], f_fp_int[fLPFPint_idxs], "fintLPFP")
            # task partition probability
            p = solLP[n]["p"]
            q = solAprx[n]["p"]
            # for residual probability = ratio of tasks being operated by manager
            pResidual = 1. - p.sum(axis=1)
            qResidual = 1. - q.sum(axis=1)
            p_add = np.hstack([p, pResidual.reshape(-1,1)]) # |OD| x |RS|+1
            q_add = np.hstack([q, qResidual.reshape(-1,1)]) # |OD| x |RS|+1
            pDif = self.get_diff(p_add, p_add, "p")
            # KL divergence
            KL = np.zeros(self.nOD, dtype=np.float64)
            for i in range(self.nOD):
                for j in range(self.nRS + 1):
                    if p_add[i,j] > 0:
                        KL[i] += p_add[i,j] * np.log(p_add[i,j]/q_add[i,j])
            # stats of KL
            statsKL = get_stats(KL, "KL")
            # reward patterns
            w_lp = solLP[n]["wage"] # |RS| x 1
            w_fp = solAprx[n]["wage"] # |RS| x 1
            wSub_fp = solAprx[n]["wageSub"] # |OD| x |RS| # * (f_fp > self.f_lb)
            # wSub_masked = np.ma.masked_equal(wSub_fp, cOpr) # to take nonzero mean
            wSub_masked = np.ma.masked_equal(wSub_fp, 0) # to take nonzero mean
            wSub_mean = wSub_masked.mean(axis=0).data # to take nonzero mean
            wDif_lp = self.get_diff(w_lp, w_fp, "wLPFP")
            wDif_fp = self.get_diff(w_fp, wSub_mean, "wMasSub")
            wnormDif_lp = self.get_diff(w_lp/cOpr, w_fp/cOpr, "wnormLPFP")
            wnormDif_fp = self.get_diff(w_fp/cOpr, wSub_mean/cOpr, "wnormMasSub")
            # od-aggregate reward
            pw_lp = w_lp[np.newaxis,:] * q  # |OD| x |RS|
            pw_fp = w_fp[np.newaxis,:] * p  # |OD| x |RS|
            pwSub_fp = solAprx[n]["wageSub"] * p  # |OD| x |RS|
            pwDif_lp = self.get_diff(pw_lp, pw_fp, "pwLPFP")
            pwDif_fp = self.get_diff(pw_fp, pwSub_fp, "pwMasSub")
            # reward
            rew_lp = w_lp[np.newaxis,:] * f_lp  # |OD| x |RS|
            rew_fp = w_fp[np.newaxis,:] * f_fp  # |OD| x |RS|
            rewSub_fp = solAprx[n]["wageSub"] * f_fp_int  # |OD| x |RS|
            aggR_odrs_Dif_lp = self.get_diff(rew_lp, rew_fp, "aggR_odrs_LPFP")
            aggR_odrs_Dif_fp = self.get_diff(rew_fp, rewSub_fp, "aggR_odrs_MasSub")
            aggR_od_Dif_lp = self.get_diff(rew_lp.sum(axis=1)/nDrvs, rew_fp.sum(axis=1)/nDrvs, "aggR_od_LPFP")
            aggR_od_Dif_fp = self.get_diff(rew_fp.sum(axis=1)/nDrvs, rewSub_fp.sum(axis=1)/nDrvs, "aggR_od_MasSub")
            aggR_rs_Dif_lp = self.get_diff(rew_lp.sum(axis=0)/nTsks, rew_fp.sum(axis=0)/nTsks, "aggR_rs_LPFP")
            aggR_rs_Dif_fp = self.get_diff(rew_fp.sum(axis=0)/nTsks, rewSub_fp.sum(axis=0)/nTsks, "aggR_rs_MasSub")
            # payment of manager
            f_rs_lp = f_lp.sum(axis=0) # |RS| x 1
            f_rs_fp = f_fp.sum(axis=0) # |RS| x 1
            pay_lp = (w_lp * f_rs_lp + cOpr * (nTsks - f_rs_lp))/nTsks
            pay_fp = (w_fp * f_rs_fp + cOpr * (nTsks - f_rs_fp))/nTsks
            paySub_fp = ((solAprx[n]["wageSub"] * f_fp_int).sum(axis=0) + cOpr * (nTsks - f_fp_int.sum(axis=0)))/nTsks
            payDif_lp = self.get_diff(pay_lp, pay_fp, "payLPFP")
            payDif_fp = self.get_diff(pay_fp, paySub_fp, "payMasSub")
            # cost saving patterns
            lmd_lp = solLP[n]["lambda"] # |RS| x 1
            lmd_fp = solAprx[n]["lambda"] # |RS| x 1
            sgm_fp = solAprx[n]["sigma"] # |OD| x |RS|
            sgm_masked = np.ma.masked_equal(sgm_fp, 0)
            sgm_mean = sgm_masked.mean(axis=0).data
            # lLPFP_idxs = (lmd_lp > 1.) * (lmd_fp > 1.)
            # lMasSub_idxs = (lmd_fp > 1.) * (sgm_mean > 1.)
            lmdDif = self.get_diff(lmd_lp, lmd_fp, "lmdLPFP")
            sgmDif = self.get_diff(lmd_fp, sgm_mean, "lmdMasSub")
            # store metrics
            metrics.append({
                'rel_dif_z': relDif,
                **zDif,
                **pDif,
                **fDif,
                **fintDif,
                **wDif_lp,
                **wDif_fp,
                **lmdDif,
                **sgmDif,
                **statsKL,
            })
        return metrics

    def compare_w_naive(self,
            recLP: List[dict], solLP: List[dict],
            recNLP: List[dict], solNLP: List[dict]
            ) -> List[dict]:

        metrics = []
        for n in range(self.nExp):
            nDrvs = self.data[n].q # |OD| x 1
            nTsks = self.data[n].x # |RS| x 1
            cOpr = self.data[n].cOpr # |RS| x 1
            
            # relative error of objective values
            zlp = recLP[n]["opt_val"]
            zlp_od = solLP[n]["opt_val_OD"]
            znlp = recNLP[n]["opt_val"]
            znlp_od = solNLP[n]["opt_val_OD"]
            relDif = np.abs((zlp - znlp) / zlp)
            zDif = self.get_diff(zlp_od, znlp_od, "zOD_LP_NLP")
            # task partition patterns
            f_lp = solLP[n]["f"] # |OD| x |RS|
            f_fp = solNLP[n]["f"] # |OD| x |RS|
            fLPFP_idxs = (f_lp > self.f_lb) * (f_fp > self.f_lb)
            fDif = self.get_diff(f_lp[fLPFP_idxs], f_fp[fLPFP_idxs], "fLP_NLP")
            # task partition probability
            p = solLP[n]["p"]
            q = solNLP[n]["p"]
            # for residual probability = ratio of tasks being operated by manager
            pResidual = 1. - p.sum(axis=1)
            qResidual = 1. - q.sum(axis=1)
            p_add = np.hstack([p, pResidual.reshape(-1,1)]) # |OD| x |RS|+1
            q_add = np.hstack([q, qResidual.reshape(-1,1)]) # |OD| x |RS|+1
            pDif = self.get_diff(p_add, p_add, "p")
            print(p_add[p_add.nonzero()])
            print(q_add[q_add.nonzero()])
            # KL divergence
            KL = np.zeros(self.nOD, dtype=np.float64)
            for i in range(self.nOD):
                for j in range(self.nRS + 1):
                    if p_add[i,j] > 0:
                        KL[i] += p_add[i,j] * np.log(p_add[i,j]/q_add[i,j])
            # stats of KL
            statsKL = get_stats(KL, "KL_LP_NLP")
            # reward patterns
            w_lp = solLP[n]["wage"] # |RS| x 1
            w_fp = solNLP[n]["wage"] # |RS| x 1
            wDif_lp = self.get_diff(w_lp, w_fp, "wLP_NLP")
            wnormDif_lp = self.get_diff(w_lp/cOpr, w_fp/cOpr, "wnormLP_NLP")
            # od-aggregate reward
            pw_lp = w_lp[np.newaxis,:] * q  # |OD| x |RS|
            pw_fp = w_fp[np.newaxis,:] * p  # |OD| x |RS|
            pwDif_lp = self.get_diff(pw_lp, pw_fp, "pwLP_NLP")
            # reward
            rew_lp = w_lp[np.newaxis,:] * f_lp  # |OD| x |RS|
            rew_fp = w_fp[np.newaxis,:] * f_fp  # |OD| x |RS|
            aggR_odrs_Dif_lp = self.get_diff(rew_lp, rew_fp, "aggR_odrs_LP_NLP")
            aggR_od_Dif_lp = self.get_diff(rew_lp.sum(axis=1)/nDrvs, rew_fp.sum(axis=1)/nDrvs, "aggR_od_LP_NLP")
            aggR_rs_Dif_lp = self.get_diff(rew_lp.sum(axis=0)/nTsks, rew_fp.sum(axis=0)/nTsks, "aggR_rs_LP_NLP")
            # payment of manager
            f_rs_lp = f_lp.sum(axis=0) # |RS| x 1
            f_rs_fp = f_fp.sum(axis=0) # |RS| x 1
            pay_lp = (w_lp * f_rs_lp + cOpr * (nTsks - f_rs_lp))/nTsks
            pay_fp = (w_fp * f_rs_fp + cOpr * (nTsks - f_rs_fp))/nTsks
            payDif_lp = self.get_diff(pay_lp, pay_fp, "payLP_NLP")
            # cost saving patterns
            lmd_lp = solLP[n]["lambda"] # |RS| x 1
            lmd_fp = solNLP[n]["lambda"] # |RS| x 1
            lmdDif = self.get_diff(lmd_lp, lmd_fp, "lmdLP_NLP")
            # store metrics
            metrics.append({
                'rel_dif_z_LP_NLP': relDif,
                **zDif,
                **pDif,
                **fDif,
                **wDif_lp,
                **lmdDif,
                **statsKL,
            })
        return metrics


    def get_diff(self, a: np.ndarray, b: np.ndarray, name: str = ""):
        a = np.nan_to_num(a, nan=0)
        b = np.nan_to_num(b, nan=0)
        abs_dif = np.abs(a - b)
        rel_true_dif = abs_dif / a
        # assert (np.maximum(a,b) >= 0).all(), f"a and b should be equal to or greater than zero: a:{a}, b:{b}"
        deno = np.maximum(a,b) #* (np.maximum(a,b) > 0) + 1. * (np.maximum(a,b) <= 0)
        rel_dif = abs_dif / deno
        rel_perc_dif = 2 * abs_dif / (a + b + self.eps)
        # nonzero = (a > 0) * (b > 0)
        # overone = (a > 1.) * (b > 1.)
        # n_oo = overone.sum()
        # if n_oo == 0: overone = [0]
        # rel_true_dif_nonzero = abs_dif[nonzero] / a[nonzero]
        # rel_true_dif_overone = abs_dif[overone] / a[overone]
        # rel_dif_nonzero = abs_dif[nonzero] / np.maximum(a[nonzero], b[nonzero])
        # rel_dif_overone = abs_dif[overone] / np.maximum(a[overone], b[overone])
        # RPD_nonzero = 2 * abs_dif[nonzero] / (a[nonzero] + b[nonzero])
        # RPD_overone = 2 * abs_dif[overone] / (a[overone] + b[overone])
        r2 = r2_score(a, b)
        corr = np.corrcoef(a,b)[0,1]
        rmse = np.sqrt(mean_squared_error(a,b))
        # rmsle = np.sqrt(mean_squared_log_error(np.clip(a, 0, None), np.clip(b, 0, None)))
        # rmse_nonzero = np.sqrt(mean_squared_error(a[nonzero],b[nonzero]))
        # rmsle_nonzero = np.sqrt(mean_squared_log_error(a[nonzero],b[nonzero]))
        # rmse_overone = np.sqrt(mean_squared_error(a[overone],b[overone]))
        # rmsle_overone = np.sqrt(mean_squared_log_error(a[overone],b[overone]))
        return {
            f"max_abs_dif_{name}": abs_dif.max(),
            f"mean_abs_dif_{name}": abs_dif.mean(),
            f"max_rel_true_dif_{name}": rel_true_dif.max(),
            # f"max_rel_true_dif_nz_{name}": rel_true_dif_nonzero.max(),
            # f"max_rel_true_dif_oo_{name}": rel_true_dif_overone.max(),
            f"mean_rel_true_dif_{name}": rel_true_dif.mean(),
            # f"mean_rel_true_dif_nz_{name}": rel_true_dif_nonzero.mean(),
            # f"mean_rel_true_dif_oo_{name}": rel_true_dif_overone.mean(),
            f"max_rel_dif_{name}": rel_dif.max(),
            # f"max_rel_dif_nz_{name}": rel_dif_nonzero.max(),
            # f"max_rel_dif_oo_{name}": rel_dif_overone.max(),
            f"mean_rel_dif_{name}": rel_dif.mean(),
            # f"mean_rel_dif_nz_{name}": rel_dif_nonzero.mean(),
            # f"mean_rel_dif_oo_{name}": rel_dif_overone.mean(),
            # f"max_RPD_{name}": rel_perc_dif.max(),
            # f"max_RPD_nz_{name}": RPD_nonzero.max(),
            # f"max_RPD_oo_{name}": RPD_overone.max(),
            # f"mean_RPD_{name}": rel_perc_dif.mean(),
            # f"mean_RPD_nz_{name}": RPD_nonzero.mean(),
            # f"mean_RPD_oo_{name}": RPD_overone.mean(),
            # f"no_nonzeros_{name}": nonzero.sum(),
            # f"no_overones_{name}": n_oo,
            f"corr_{name}": corr,
            f"R2_{name}": r2,
            f"RMSE_{name}": rmse,
            # f"RMSLE_{name}": rmsle,
            # f"RMSE_nz_{name}": rmse_nonzero,
            # f"RMSLE_nz_{name}": rmsle_nonzero,
            # f"RMSE_oo_{name}": rmse_overone,
            # f"RMSLE_oo_{name}": rmsle_overone,
        }

# %%
if __name__ == '__main__':
    np.random.seed(123)

    nOD = 10
    nRS = 10
    nDrv = 10000
    nTsk = 10000
    theta = 1.
    cbarR = 3.0
    param_ = setparam_(nOD, nRS, nDrv, nTsk, theta, cbarR)
    print(param_)

    # %%
    link_path = 'data/Winnipeg/link.csv'
    od_path = 'data/Winnipeg/od.csv'
    cSP_load = np.load('data/Winnipeg/cSP.npy')
    dataset = Dataset(link_path, od_path, cSP_load)
    expData = dataset.generate_data(param_, N = 1)
    # print(exData)

    # %%
    # define model
    csd = CSD(solver='gurobi', nThrd=32, msg=False, solve_dual=True)
    csd.load_data(param_, expData)
    
    # %%
    recSOD, solSOD = csd.solve_SOD_bfgs()
    print("SO-D wage:", solSOD[0]["wage"].round(2))

    # %%
    # solve LP
    recordsLP, solLP = csd.solve_SOP_lp()
    dfResLP = pd.DataFrame(recordsLP)
    print(dfResLP) #.describe()

    # %%
    # solve FP
    recordsFP, solFP = csd.solve_fluid_particle(vcg=False)
    dfResFP = pd.DataFrame(recordsFP)
    print(dfResFP) #.describe()
    solFP[0]["wage"].round(2)
    
    # %%
    # evaluate metrics
    metrics = csd.compare_results(recordsLP, solLP, recordsFP, solFP)
    dfMetrics = pd.DataFrame(metrics)

    # %%
    wageLP = solSOD[0]["wage"]
    wageMaster = solFP[0]["wage"]
    wageSub = solFP[0]["wageSub"] * (solFP[0]["f"] > 2.)
    # masked = np.ma.masked_equal(solFP[0]["wageSub"], csd.data[0].cOpr)
    masked = np.ma.masked_equal(wageSub, 0)
    wageSub = masked.mean(axis=0).data
    print("SO-D wage:", wageLP.round(2))
    print("SOA-D wage:", wageMaster.round(2))
    print("SOD-Sub wage:", wageSub.round(2))
    print(csd.get_diff(wageLP, wageMaster))
    print(csd.get_diff(wageLP, wageSub))
    print(csd.get_diff(wageMaster, wageSub))
    
    # %%
    lamdLP = solSOD[0]["lambda"]
    lamdMaster = solFP[0]["lambda"]
    masked = np.ma.masked_equal(solFP[0]["sigma"], 0)
    sigma = masked.mean(axis=0).data
    print("SO-D lamd:", lamdLP.round(1))
    print("SOA-D lamd:", lamdMaster.round(1))
    print("SOD-Sub sigma:", sigma.round(1))
    print(csd.get_diff(lamdLP, lamdMaster))
    print(csd.get_diff(lamdLP, sigma))
    print(csd.get_diff(lamdMaster, sigma))
    
    # %%
    pi = solFP[0]["pi"]
    piA = np.vstack(pi)
    piA.round(2)
    deno = np.clip((piA.round(2) > 0).sum(axis=0), 1e-2, None)
    # piA.sum(axis=0).round(2)
    print("VCG wage", (piA.sum(axis=0) / deno).round(2))

    # %%
    metrics = csd.compare_results(recordsLP, solLP, recordsFP, solFP)
    print(metrics)
    print(get_stats(metrics[0]["KL"], "KL"))
