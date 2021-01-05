import numpy as np
import copy as cp

class CAssist:
    def __init__(self, CacheSize):
        self.CacheSize = CacheSize
        self.UsedSpace = 0
        self.Caches = {}
        self.NRTs = {}
        self.LRTs = {}
        self.Timer = 0
        self.hits = 0
        self.misses = 0
        self.Af = {}

    def reset(self):
        self.hits = 0
        self.misses = 0

    def rank(self):
        find_key = -1
        find_val = -10000
        for ky in self.Caches.keys():
            lrt = self.LRTs[ky]
            nrt = self.NRTs[ky]
            real_nrt = max(lrt, nrt) * max(int(self.Caches[ky] / 1024), 1)
            if real_nrt > find_val:
                find_val = real_nrt
                find_key = ky
        return find_key

    def evict(self):
        find_key = self.rank()
        self.UsedSpace -= self.Caches[find_key]
        self.Caches.pop(find_key)
        self.NRTs.pop(find_key)
        return find_key

    def decide(self, req, nrt):
        Id = req['id']
        Size = req['size']
        Nrt = nrt
        InCache = 0
        if Id in self.Caches.keys():
            InCache = 1

        if InCache == 1:
            self.hits += 1
            self.NRTs[Id] = np.exp(Nrt)
            self.Af[Id] += 1
        else:
            self.misses += 1
        if InCache == 0:
            self.Caches[Id] = Size
            self.NRTs[Id] = np.exp(Nrt)
            self.Af[Id] = 1
            self.UsedSpace += Size

        self.LRTs[Id] = self.Timer
        evict_id = []
        while self.UsedSpace > self.CacheSize:
            di = self.evict()
            if self.Af[di] == 1:
                evict_id.append(di)
        self.Timer += 1
        return evict_id

class LSO:
    def __init__(self, reqs, nrts, cachesys):
        self.CacheSys = cachesys
        self.NRTs = nrts
        self.Reqs = cp.deepcopy(reqs)
        self.RealNRTs = []
        self.EstNRTs = {}
        self.Init_Sols = []
        self.AddReqs = []
        self.K = len(reqs)
        self.Hits_all = 0
        self.Sols_All = []

    def sampling(self, L):
        RDs = np.random.rand(L)
        Locs = []
        for r in RDs:
            l = int(r * 10 * len(self.Reqs))
            loc = l % len(self.Reqs)
            Locs.append(loc)
        for l in Locs:
            req = self.Reqs[l]
            self.Reqs.append(req)

    def limits(self, cachesize):
        RealNRTs = []
        Keys = []
        EstNRTs = {}
        Trace = cp.deepcopy(self.Reqs)
        for i in range(len(Trace)):
            req1 = Trace[i]
            id1 = req1['id']
            nrt = -1
            Break = 0
            for j in range(i + 1, len(Trace)):
                req2 = Trace[j]
                id2 = req2['id']
                if id1 == id2:
                    nrt = j - i
                    Break = 1
                    break
            if Break == 1:
                if id1 in EstNRTs.keys():
                    EstNRTs[id1].append(nrt)
                else:
                    EstNRTs[id1] = [nrt]
            RealNRTs.append(nrt)
            Keys.append(id1)

        ESTs = {}
        for ky in EstNRTs.keys():
            get_nrts = EstNRTs[ky]
            m_nrt = np.mean(get_nrts)
            ESTs[ky] = m_nrt

        for i in range(len(Keys)):
            key = Keys[i]
            est_nrt = 2 * len(Trace)
            if key in ESTs.keys():
                est_nrt = ESTs[key]
            if RealNRTs[i] == -1:
                RealNRTs[i] = est_nrt

        self.EstNRTs = cp.deepcopy(ESTs)
        Overlaps = []
        Valids = []
        for i in range(len(self.Reqs)):
            next1 = RealNRTs[i]
            count = 0
            valid_time = i + int(next1)
            if valid_time > len(self.Reqs):
                if next1 < len(Trace):
                    Overlaps.append(int(next1) / 2)
                else:
                    Overlaps.append(100000)
            else:
                for j in range(i + 1, valid_time):
                    next2 = RealNRTs[j]
                    if j + next2 < valid_time:
                        count += 1
                Overlaps.append(count)
                Valids.append(count)
        Per1 = np.percentile(Overlaps, 5)
        Per2 = np.percentile(Overlaps, 95)

        Sols = []
        Asis = CAssist(cachesize)
        LastLocs = {}
        for i in range(len(self.Reqs)):
            req = self.Reqs[i]
            id = req['id']
            nrt = self.NRTs[i]
            LastLocs[id] = i
            Sols.append(0.8)
            zero_ids = Asis.decide(req, nrt)
            for ky in zero_ids:
                loc = LastLocs[ky]
                Sols[loc] = 0.2

        self.CacheSys.reset()
        Cacheall = cp.deepcopy(self.CacheSys)
        Sols_all = []
        for i in range(self.K):
            req_here = self.Reqs[-self.K + i]
            nrt_here = self.NRTs[-self.K + i]
            Cacheall.decide(req_here, 1.0, nrt_here)
            Sols_all.append(1.0)
        self.Hits_all = Cacheall.hits
        self.Sols_all = Sols_all[-self.K:]

        NumZero = 0
        NumOne = 0
        for i in range(len(Overlaps)):
            lap = Overlaps[i]
            if lap > Per2:
                if Sols[i] != 0.2:
                    NumZero += 1
                Sols[i] = 0
            if lap <= Per1 and lap >= 0:
                if Sols[i] != 0.8:
                    NumOne += 1
                Sols[i] = 1

        self.Init_Sols = cp.deepcopy(Sols)
        self.sampling(int(self.K))

    def get_opt(self, hits_now):
        if hits_now < self.Hits_all:
            return self.Hits_all, self.Sols_all, 0
        GetOpt = []
        GetHits = 0
        self.CacheSys.reset()
        Cachesys_now = cp.deepcopy(self.CacheSys)
        RDs = np.random.rand(len(self.Reqs))
        gamma = 0.8
        TotalReq = self.Reqs
        for i in range(len(TotalReq)):
            prob = 1.0
            nrt = 9.3
            req = TotalReq[i]
            Id = req['id']
            if i < self.K:
                prob = self.Init_Sols[i]
                nrt = self.NRTs[i]
                rd = RDs[i]
                if rd <= prob:
                    prob = 0.8
                else:
                    prob = 0.2
                GetOpt.append(prob)
            if i >= self.K:
                if Id in self.EstNRTs.keys():
                    nrt = self.EstNRTs[Id]
            InCache = 0
            if Id in Cachesys_now.Caches.keys():
                InCache = 1
            if InCache == 1:
                if i < self.K:
                    GetHits += 1
                else:
                    GetHits += pow(gamma, i - self.K)
            Cachesys_now.decide(req, prob, nrt)
        return GetHits, GetOpt, 1

