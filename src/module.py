'''
version 1.0 - 2024-08-09
@author: tommaso
'''

import numpy as np 
import time
from typeguard import typechecked
from typing import Iterable
from abc import ABC, abstractmethod 

INFINITY = float('inf')
PRECISION = 3
EPSILON = 10 ** -PRECISION
DEFAULT_ARC_LENGTH = 100


@typechecked
def setPrecision(value:int) -> None:
    global PRECISION
    global EPSILON
    PRECISION = value
    EPSILON = 10 ** -PRECISION


@typechecked
def myround(value: float) -> float:
    return round(value, PRECISION)

 
@typechecked
def mycompare(lhs: float, rhs: float) -> int:
    '''
    It returns:
    0 if lhs and rhs are closer than EPSILON
    -1 if lhs is numerically less than rhs
    1 if lhs is numerically greater than rhs
    '''
    if abs(rhs - lhs) <= EPSILON:
        return 0
    if lhs < rhs:
        return -1
    else:
        return 1


@typechecked
def findIndexLinear(orderedList:Iterable[float], t: float) -> int: 
    n = len(orderedList)
    for i in range(n):
        if orderedList[i] > t:
            return i - 1
    return n - 1


@typechecked
def findIndex(orderedList:Iterable[float], t: float) -> int: 
    k = np.searchsorted(orderedList, t, side="right") - 1
    # print(l, k, orderedList, t)
    assert findIndexLinear(orderedList, t) == k     
    return int(k)


@typechecked
class Base(ABC):

    @abstractmethod
    def computeDepartureTime(self, end: float) -> float:
        pass
    
    @abstractmethod 
    def computeArrivalTime(self, start: float) -> float:
        pass
    
    @abstractmethod
    def computeArrivalTimeFunction(self):
        pass
    
    def computeTraversalTime(self, t: float) -> float:
        return self.computeArrivalTime(t) - t
    
    def computeBackwardTraversalTime(self, t: float) -> float:
        return t - self.computeDepartureTime(t)
    
    def __generateTimeSequence(self, t: float, inf: float, sup: float) -> set[float]:
        if mycompare(t, inf) < 0 or mycompare(t, sup) > 0:
            return set()
        S = { myround(t) }        
        t1 = t
        while True:
            t1 = self.computeArrivalTime(t1)
            if mycompare(t1, sup) > 0:
                break
            else: 
                if t1 in S:
                    break
                else:
                    S.add(myround(t1))
        t1 = t
        while True:
            try:
                t1 = self.computeDepartureTime(t1)
            except IndexError:
                break
            if mycompare(t1, inf) < 0:
                break
            else:
                if t1 in S:
                    break
                else:
                    S.add(myround(t1))
        return S
    
    def computePotentialSpeedBreakpoints(self, inf: float, sup: float) -> list[float]:
        S = set()
        for t in self.bp:
            S |= self.__generateTimeSequence(t, inf, sup)
        S |= self.__generateTimeSequence(inf, inf, sup)
        S |= self.__generateTimeSequence(sup, inf, sup)
        return sorted(S)
    
    def computeSpeedFunction(self, length: float, log:bool=False):
        '''
        It computes a stepwise speed function according to:
         
        Ghiani, G., & Guerriero, E. (2014). A note on the Ichoua, Gendreau, and Potvin (2003) travel time model. Transportation Science, 48(3), 458-462.
        https://doi.org/10.1287/trsc.2013.0491
        '''
        t0 = time.time()
        potentialBP = self.computePotentialSpeedBreakpoints(self.bp[0], self.computeArrivalTime(self.bp[-1]))  # self.values[-1] 
        # potentialBP = self.computePotentialSpeedBreakpoints()
        t1 = time.time()
        H = len(potentialBP)
        x = [.0 for _ in range(H)]
        # x = np.empty(H, float)        
        mask = [False for _ in range(H)]
        succ = .0
        for h in range(H - 1, -1, -1):
            A = [.0 for _ in range(H)]
            # A = np.empty(H, float)
            gamma = self.computeArrivalTime(potentialBP[h])
            summed = .0
            for l in range(h, H):
                T_l = potentialBP[l]
                if gamma < T_l:
                    break
                # gamma >= T_l                
                if l < H - 1 and gamma > potentialBP[l + 1]:
                    A[l] = potentialBP[l + 1] - T_l
                else:  # gamma <= T_lplus1 
                    A[l] = gamma - T_l                
                # A[l] = min(T_lplus1 - T_l, max(.0, gamma - T_l))                
                if l > h:
                    summed += A[l] * x[l]
            x[h] = (length - summed) / A[h]
            if h < H - 1 and mycompare(x[h], succ) != 0: 
                mask[h + 1] = True
            succ = x[h]
        mask[0] = True
        t2 = time.time()
        
        # remove unnecessary breakpoints
        maxSpeed = np.max(x)
        speedBP = [ potentialBP[0] ]
        prev = x[0]
        speedValues = [ prev ]
         
        for h in range(1, H):
            if mycompare(prev, x[h]) != 0:
                speedBP.append(potentialBP[h])
                prev = x[h]
                speedValues.append(prev)
        
        # assert np.equal(np.asarray(potentialBP)[mask], np.asarray(speedBP)).all()
        # assert np.equal(np.asarray(x)[mask], np.asarray(speedValues)).all()

        # TODO: unlock this variant
        # speedBP = np.append(np.asarray(potentialBP)[mask], sup)
        # speedValues = np.asarray(x)[mask]        
        t3 = time.time()
        if log:
            print("Potential BP computed in:", t1 - t0, "s")
            print("Speed values computed in:", t2 - t1, "s")
            print("Unnecessary BP removed in:", t3 - t2, "s")
            print(f"Retained only {len(speedBP)} of {H} potential speed breakpoints")
        return Arc(length, maxSpeed, np.asarray(speedValues) / maxSpeed, np.asarray(speedBP))


@typechecked
class Graph:
    '''
    It models a time-dependent graph. 
    '''

    def __init__(self, vertices:np.ndarray, arcs:dict, timeWindows:np.ndarray=np.empty(0)) -> None:
        self.vertices = vertices
        self.arcs = arcs
        self.timeWindows = timeWindows
    
    def __repr__(self) -> str:
        return self.__dict__.__repr__()
    
    def writeRoadGraph(self, filePath: str) -> None:
        n = len(self.vertices)
        with open(filePath, 'w') as f:
            f.write(f"VERTICES {n}\n")
            m = sum(len(self.arcs[i]) for i in self.arcs)
            f.write(f"ARCS {m}\n")
            for i in self.arcs:
                for j in self.arcs[i]:
                    a = self.arcs[i][j]
                    if isinstance(a, PWLFunction):
                        a = a.computeSpeedFunction(DEFAULT_ARC_LENGTH)
                    s = " ".join([f"{myround(a.bp[h])} {myround(a.jams[h])}" for h in range(a.numSteps())])
                    f.write(f"{i} {j} {a.length} {myround(a.maxSpeed)} {a.numSteps()} {s}\n")

    def writeCustomerBasedGraphTopology(self, filePath: str) -> None:
        n = len(self.vertices)
        with open(filePath, 'w') as f:
            f.write(f"CUSTOMERS {n}\n")
            for i in self.vertices:
                for j in self.vertices:
                    if i != j:
                        a = self.arcs[i][j]
                        if isinstance(a, PWLFunction):
                            gamma = a
                        else:
                            gamma = a.computeArrivalTimeFunction()
                        s = " ".join([f"{myround(gamma.bp[h])} {myround(gamma.values[h]-gamma.bp[h])}" for h in range(gamma.size())])
                        f.write(f"{i} {j} {gamma.size()} {s}\n")
    
    def writeTimeWindows(self, filePath: str) -> None:
        with open(filePath, 'w') as f:
            for w in self.timeWindows:
                f.write(f"{w.opening} {w.closing}\n")
    
    def readApproximations(self, filePath: str) -> tuple:
        with open(filePath) as f:
            tokens = f.readline().strip().split(" ")
            n = int(tokens[1])
            BP = np.asarray([ float(tokens[2 + h * 2]) for h in range(n)])
            commonSpeed = np.asarray([ float(tokens[3 + h * 2]) for h in range(n)])
            for l in f:
                tokens = l.strip().split(" ")
                if len(tokens) < 4:
                    break
                i = int(tokens[0])
                j = int(tokens[1])
                self.arcs[i][j].minLength = float(tokens[2])
                self.arcs[i][j].maxLength = float(tokens[3])
        return BP, commonSpeed
    
    def dijkstra(self, start: int, startTime: float, goal: int | None=None) -> np.ndarray:
        if start not in self.vertices or (goal != None and goal not in self.vertices):
            return None
        # P = np.full(len(self.vertices), -1, int)
        D = [INFINITY for _ in self.vertices]
        # C = [False for _ in self.vertices]
        C = set(self.vertices)
        D[start] = startTime
        k = start        
        while k != goal:
            for i in self.arcs[k]: 
                try:
                    arrivalTime = self.arcs[k][i].computeArrivalTime(D[k])
                except:
                    # print(k, i, D[k])  # TODO: check this case
                    raise IndexError
                if arrivalTime < D[i]:
                    D[i] = arrivalTime
                    # P[i] = k
            # C[k] = True
            C.remove(k)            
            lowest = INFINITY
            k = None
            for i in C:
                if D[i] < lowest:
                    lowest = D[i]
                    k = i
            if lowest == INFINITY:  # unreachable goal
                break
        return D
    
    def computeArrivalTime(self, sequence:Iterable, startTime: float) -> float: 
        t = startTime
        u = sequence[0]
        if u in self.vertices:
            for i in range(len(sequence) - 1):
                v = sequence[i + 1]
                if v in self.vertices:
                    t = self.arcs[u][v].computeArrivalTime(t)
                    u = v
                else:
                    raise ReferenceError
        else:
            raise ReferenceError
        return t


@typechecked
class Arc(Base): 
    '''
    It models an arc in a time-dependent graph with a fixed length and a stepwise function for the speed according to:
    
    Ichoua, S., Gendreau, M., & Potvin, J. Y. (2003). Vehicle dispatching with time-dependent travel times. European journal of operational research, 144(2), 379-396.
    https://doi.org/10.1016/S0377-2217(02)00147-9
    '''
    
    def __init__(self, length: float, maxSpeed: float=1., jams:np.ndarray=np.empty(0), bp:np.ndarray=np.empty(0)) -> None:
        self.length = length
        self.jams = jams
        self.maxSpeed = maxSpeed        
        self.bp = bp
        assert(len(bp) == len(jams)) 
        # self.speed = tuple(maxSpeed*jams[h] for h in range(len(jams)))

    def speed(self, h: int) -> float:
        return self.maxSpeed * self.jams[h]
    
    def numSteps(self) -> int:
        return len(self.bp)
    
    def __repr__(self) -> str:
        return self.__dict__.__repr__()
        
    def computeArrivalTime(self, start: float) -> float:
        cost = self.length
        if cost == .0:
            return start
        BP = self.bp
        k = findIndex(BP, start)
        if k == -1:
            raise IndexError
        t = start
        t1 = t + cost / self.speed(k)                    
        while k < len(BP) - 1 and t1 > BP[k + 1]:
            cost = cost - self.speed(k) * (BP[k + 1] - t)
            t = BP[k + 1]
            t1 = t + cost / self.speed(k + 1)
            k += 1    
        return t1

    def computeDepartureTime(self, end: float) -> float:
        cost = self.length
        if cost == .0:
            return end
        BP = self.bp
        k = findIndex(BP, end)
        if k == -1:
            raise IndexError                     
        t = end
        t1 = t - cost / self.speed(k)
        while t1 < BP[k]:
            cost = cost - self.speed(k) * (t - BP[k])
            t = BP[k]
            if k < 1:
                raise IndexError
            t1 = t - cost / self.speed(k - 1)
            k -= 1
        return t1
    
    def computeLen(self, start: float, end: float) -> float:
        tau = end - start
        if tau <= .0:
            return .0        
        BP = self.bp
        k = findIndex(BP, start)
        if k == -1:
            raise IndexError        
        t = start
        length = .0
        while k < len(BP) - 1 and end > BP[k + 1]:
            length += (BP[k + 1] - t) * self.speed(k)
            t = BP[k + 1]
            k += 1
        length += (end - t) * self.speed(k)
        return length

    def __computeArrivalTimeBP(self) -> list[float]:
        ''' 
        It computes the potential breakpoints for the arrival time function according to:
        
        Vidal, T., Martinelli, R., Pham, T. A., & Hà, M. H. (2021). Arc routing with time-dependent travel times and paths. Transportation Science, 55(3), 706-724. 
        https://doi.org/10.1287/trsc.2020.1035
        ''' 
        BP = self.bp
        S = set(BP)
        for i in range(-1, -len(BP), -1):
            try:
                d = self.computeDepartureTime(BP[i])
                if d < BP[0]:  # mycompare(d, BP[0]) < 0: 
                    break
                S.add(d)
            except:
                break
        return sorted(S)

    def computeArrivalTimeFunction(self):
        bp = np.asarray(self.__computeArrivalTimeBP())
        values = np.asarray([self.computeArrivalTime(t) for t in bp])
        
        # remove unnecessary breakpoints
        mask = np.empty(len(values), bool)
        s0 = .0
        for h in range(len(values) - 1):
            y0 = values[h]
            y1 = values[h + 1]
            x0 = bp[h]
            x1 = bp[h + 1]
            s1 = (y1 - y0) / (x1 - x0)
            mask[h] = mycompare(s0, s1) != 0
            s0 = s1
        mask[-1] = True      
        return PWLFunction(bp[mask], values[mask])

    
@typechecked
class PWLFunction(Base):
    '''
    It models a piecewise linear (PWL) arrival time function.
    It uses two parallel arrays to represent the segments of the PWL function.
    '''
    
    def __init__(self, bp:np.ndarray, values:np.ndarray) -> None:
        assert(len(bp) == len(values))  
        self.bp = bp
        self.values = values 
    
    def size(self) -> int: 
        return len(self.bp)      
        
    def slope(self, h:int) -> float:
        if h >= self.size() - 1:
            return 1.  # \tau is constant in the long-run
        if h < 0:
            raise IndexError
        y0 = self.values[h]
        y1 = self.values[h + 1]
        x0 = self.bp[h]
        x1 = self.bp[h + 1]
        return (y1 - y0) / (x1 - x0)
    
    def slopes(self) -> list[float]:
        return [self.slope(h) for h in range(self.size() - 1)]
        
    def isFIFO(self) -> bool:
        return bool(np.all([mycompare(self.slope(h), .0) > 0 for h in range(self.size() - 1)]))
    
    def inverse(self):
        return PWLFunction(self.values, self.bp)
    
    def computeArrivalTime(self, t: float) -> float:
        h = findIndex(self.bp, t)
        if h < 0:
            raise IndexError
        x0 = self.bp[h]
        y0 = self.values[h]
        return y0 + self.slope(h) * (t - x0)        
    
    def computeDepartureTime(self, t: float) -> float:
        h = findIndex(self.values, t)
        if h < 0:
            raise IndexError        
        x0 = self.values[h]        
        y0 = self.bp[h]
        if h >= self.size() - 1:
            s = 1.
        else:
            x1 = self.values[h + 1]
            y1 = self.bp[h + 1]
            s = (y1 - y0) / (x1 - x0)                
        r = y0 + s * (t - x0)
        # assert mycompare(self.inverse().computeArrivalTime(t), r) == 0
        return r  

    def computeArrivalTimeFunction(self):
        return self
    
    def __repr__(self) -> str:
        return self.__dict__.__repr__()


@typechecked    
class TimeWindow:
    '''
    A class to represent a single time interval.
    '''

    def __init__(self, opening: float, closing: float) -> None:
        self.opening = opening
        self.closing = closing
        
    def duration(self) -> float:
        return self.closing - self.opening
    
    def intersection(self, other):
        a = max(self.opening, other.opening);
        b = min(self.closing, other.closing);
        if mycompare(b, a) >= 0:
            return TimeWindow(a, b)
        else:
            return None
        
    def overlaps(self, other) -> bool:
        return self.intersection(other) != None
    
    def contains(self, t: float) -> bool:
        return mycompare(t, self.opening) >= 0 and mycompare(t, self.closing) <= 0
        
    def before(self, other) -> bool:
        return mycompare(self.closing, other.opening) <= 0
    
    def after(self, other) -> bool: 
        return mycompare(self.opening, other.closing) >= 0
    
    def __repr__(self) -> str:
        return self.__dict__.__repr__()


@typechecked
def readTopology(graphFile: str, twFile: str="") -> Graph:
    A = {}
    with open(graphFile) as f:
        line = f.readline().strip()
        n = int(line.split(" ")[1])
        for _ in range(n * n - n):
            line = f.readline().strip()
            tokens = line.split(" ")
            i = int(tokens[0])
            j = int(tokens[1])
            if i not in A:
                A[i] = {}
            H = range(int(tokens[2]))
            BP = np.asarray(tuple(float(tokens[3 + h * 2]) for h in H))
            values = np.asarray(tuple(float(tokens[4 + h * 2]) + BP[h] for h in H))
            A[i][j] = PWLFunction(BP, values)
            if not A[i][j].isFIFO():
                raise RuntimeError(f"FIFO violated for arc ({i}, {j})") 
    N = tuple(range(n))
    if twFile != "":
        with open(twFile) as f:
            TW = tuple(TimeWindow(*tuple(float(v) for v in f.readline().strip().split(" "))) for i in N)
    else:
        TW = ()
    return Graph(np.asarray(N), A, np.asarray(TW))

    
@typechecked
def readInstance(graphFile: str, twFile:str) -> Graph:
    with open(graphFile) as f:
        line = f.readline().strip()
        N = tuple(range(int(line.split(" ")[1])))
        line = f.readline().strip()
        M = range(int(line.split(" ")[1]))
        A = {}
        for _ in M:
            line = f.readline().strip()
            tokens = line.split(" ")
            i = int(tokens[0])
            j = int(tokens[1])
            if i not in A:
                A[i] = {}
            H = range(int(tokens[4]))
            BP = np.asarray(tuple(float(tokens[5 + h * 2]) for h in H))
            jams = np.asarray(tuple(float(tokens[6 + h * 2]) for h in H))
            A[i][j] = Arc(float(tokens[2]), float(tokens[3]), np.asarray(jams), np.asarray(BP))
    with open(twFile) as f:
        TW = tuple(TimeWindow(*tuple(float(v) for v in f.readline().strip().split(" "))) for i in N)
    return Graph(np.asarray(N), A, np.asarray(TW))

