'''
version 1.1 - 2024-10-24
@author: tommaso
'''

import docplex.mp.model as cplex
from src.module import *
from typeguard import typechecked

@typechecked
class Solver:
    '''
    Adamo, T., Ghiani, G., & Guerriero, E. (2021). On path ranking in time-dependent graphs. Computers & Operations Research, 135, 105446.
    https://doi.org/10.1016/j.cor.2021.105446
    
    Adamo, T., Ghiani, G., Greco, P., & Guerriero, E. (2023). Learned upper bounds for the time-dependent travelling salesman problem. IEEE Access, 11, 2001-2011.
    https://doi.org/10.1109/ACCESS.2022.3233852
    '''

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
    
    def computePotentialSpeedBreakpoints(self, action=None) -> list[float]:
        BP = set()
        N = self.graph.vertices
        for i in N:
            for j in N:
                if i != j:
                    a = self.graph.arcs[i][j]
                    inf = a.bp[0]
                    sup = a.computeArrivalTime(a.bp[-1]) 
                    inf = max(a.bp[0], self.graph.timeWindows[i].opening)
                    sup = min(a.computeArrivalTime(a.bp[-1]), a.computeArrivalTime(self.graph.timeWindows[i].closing))
                    potentialSpeedBP = a.computePotentialSpeedBreakpoints(inf, sup)
                    if '=' in action:
                        a.potentialSpeedBP = potentialSpeedBP 
                    if '&' in action:
                        if len(BP) == 0:
                            BP = set(potentialSpeedBP)
                        else:
                            BP.intersection_update(potentialSpeedBP)
                    elif '|' in action:
                        BP.update(potentialSpeedBP)
        return sorted(BP)
    
    @classmethod
    def lower(cls, originalArc, fittingArc, inf:float, sup:float, onlyNegative=False, maxIterations=1000) -> tuple:
        '''
        Update the fittingArc length in order to be a lower bound on the arrival time function of originalArc.
        '''
        f = originalArc.computeArrivalTimeFunction()
        for iteration in range(maxIterations):            
            g = fittingArc.computeArrivalTimeFunction()
            
            S = [ t for t in set(f.bp) | set(g.bp) | {inf, sup} if t >= inf and t <= sup ]
            minimum = INFINITY
            t1 = 0.
            for t in S:
                diff = f.computeArrivalTime(t) - g.computeArrivalTime(t)
                if diff < minimum:
                    minimum = diff
                    t1 = t
            r = mycompare(minimum, 0.)
            if r == 0 or (onlyNegative and r > 0):
                break
            length = fittingArc.computeLen(t1, f.computeArrivalTime(t1))    
            if fittingArc.length == length:
                break
            print(f"LOWER iterative call {iteration} to close gap: {minimum}")
            fittingArc.length = length       
        return minimum, fittingArc.length 
    
    @classmethod
    def upper(cls, originalArc, fittingArc, inf:float, sup:float, onlyNegative=False, maxIterations=1000) -> tuple:
        '''
        Update the fittingArc length in order to be an upper bound on the arrival time function of originalArc.
        '''
        f = originalArc.computeArrivalTimeFunction()
        
        for iteration in range(maxIterations):
            g = fittingArc.computeArrivalTimeFunction()
            S = [ t for t in set(f.bp) | set(g.bp) | {inf, sup} if t >= inf and t <= sup ]    
            minimum = INFINITY
            t1 = 0.
            for t in S:
                diff = g.computeArrivalTime(t) - f.computeArrivalTime(t)
                if diff < minimum:
                    minimum = diff
                    t1 = t
            r = mycompare(minimum, 0.)
            if r == 0 or (onlyNegative and r > 0):
                break
            length = fittingArc.computeLen(t1, f.computeArrivalTime(t1))
            if fittingArc.length == length:
                break
            print(f"UPPER iterative call {iteration} to close gap: {minimum}")
            fittingArc.length = length
        return minimum, fittingArc.length
    
    def solve(self, initialBP, folder: str, numThreads = 4, logOutput = True) -> None:
        N = self.graph.vertices
        BP = set(initialBP) 
        for i in N:
            BP.add(self.graph.timeWindows[i].opening)
            BP.add(self.graph.timeWindows[i].closing)
        BP = sorted(BP)
        H = len(BP)
        minTP = min(BP[h+1]-BP[h] for h in range(H-1))
        print(f"{H} potential breakpoints for the common speed profile, rho = 1/{minTP}")
        print("Potential breakpoints ", BP)
        
        mdl = cplex.Model("CTCP")
        y = mdl.continuous_var_list(H, name='y')
        x_min = mdl.continuous_var_matrix(N, N, name='xm')
        x_max = mdl.continuous_var_matrix(N, N, name='xM')
        mdl.minimize(sum(x_max[i, j] - x_min[i, j] for i in N for j in N if i != j))
        for i in N:
            for j in N:
                if i != j:
                    a = self.graph.arcs[i][j]
                    inf = self.graph.timeWindows[i].opening
                    sup = self.graph.timeWindows[i].closing
                    a.potentialSpeedBP = sorted(t for t in set(BP) | { inf, sup } if t >= inf and t <= sup)             
                    K = len(a.potentialSpeedBP)
                    for k in range(K):
                        gamma = a.computeArrivalTime(a.potentialSpeedBP[k])
                        summed = 0.
                        for l in range(H):
                            T_l = BP[l]
                            if T_l < a.potentialSpeedBP[k]:
                                continue
                            if gamma < T_l:
                                break
                            # gamma >= T_l                
                            if l < H - 1 and gamma > BP[l + 1]:
                                summed += (BP[l + 1] - T_l) * y[l]
                            else:  # gamma <= T_lplus1
                                summed += (gamma - T_l) * y[l]                
                        mdl.add_constraint(x_min[i, j] <= summed)
                        mdl.add_constraint(summed <= x_max[i, j])
        for h in range(H):
            mdl.add_constraint(minTP * y[h] >= 1.)        
        mdl.context.cplex_parameters.threads = numThreads
        mdl.print_information()
        s = mdl.solve(log_output=logOutput)
        if s == None:
            print('No solution')
        else:
            print(f"z_value = {mdl.objective_value}")
            commonSpeed = np.asarray([y[h].solution_value for h in range(H)])
            
            # remove unnecessary breakpoints
            BP = np.asarray(BP)
            mask = np.empty(H, bool)
            prev = None
            for h in range(H):
                curr = y[h].solution_value
                mask[h] = prev != curr
                prev = curr
            mask[-1] = True
            commonSpeed = commonSpeed[mask]
            BP = BP[mask]            
            H = len(BP)
            
            maxSpeed = np.max(commonSpeed)          
            with open(folder+"/approximations.txt", 'w') as f:
                s = " ".join([f"{BP[h]} {commonSpeed[h]}" for h in range(H)])
                f.write(f"COMMON_SPEED_PROFILE {H} {s}\n")
                for i in N:
                    for j in N:
                        if i != j:
                            print("Processing arc:", i, j)
                            a = self.graph.arcs[i][j]
                            LB = Arc(x_min[i,j].solution_value, maxSpeed, commonSpeed/maxSpeed, BP)
                            Solver.lower(a, LB, self.graph.timeWindows[i].opening, self.graph.timeWindows[i].closing, False)
                            UB = Arc(x_max[i,j].solution_value, maxSpeed, commonSpeed/maxSpeed, BP)
                            Solver.upper(a, UB, self.graph.timeWindows[i].opening, self.graph.timeWindows[i].closing, False)
                            l = LB.length
                            u = UB.length
                            f.write(f"{i} {j} {l} {u}\n")                      
                dummy:Arc = Arc(1., maxSpeed, commonSpeed/maxSpeed, BP)
                for i in N:
                    tw = self.graph.timeWindows[i]
                    A = dummy.computeLen(0., tw.opening)
                    B = dummy.computeLen(0., tw.closing)
                    f.write(f"{A} {B}\n")
                    
