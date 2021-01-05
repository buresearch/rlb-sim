import copy as cp

class Cache_Sys:
    def __init__(self,space):
        self.space = space
        self.UsedSpace = 0
        self.Caches = {}
        self.b_recency = {}
        self.timer = 0
        self.hits = 0
        self.misses = 0
        self.access_freq = {}
        self.LCurrent = 0

    def reset(self):
        self.hits = 0
        self.misses = 0

    def rank(self):
        Get_Recency = cp.deepcopy(self.b_recency)
        find_val = pow(10,10)
        find_key = None
        for ky in Get_Recency.keys():
            Rec = Get_Recency[ky]
            if Rec < find_val:
                find_val = Rec
                find_key = ky
        return find_key,find_val

    def evict(self):
        find_key,find_val = self.rank()
        self.UsedSpace -= self.Caches[find_key]
        self.Caches.pop(find_key)
        self.LCurrent = self.b_recency[find_key]
        self.b_recency.pop(find_key)
        self.access_freq.pop(find_key)

    def admit(self,req,prob,nrt,T=0.5):
        Id = req['id']
        Size = req['size']
        Prob = prob
        InCache = 0
        if Id in self.Caches.keys():
            InCache = 1

        if InCache == 1:
            self.hits += 1
            self.b_recency.pop(Id)
            self.access_freq[Id] += 1
            self.b_recency[Id] = self.LCurrent + self.access_freq[Id]  / Size * Prob
        else:
            self.misses += 1
        if Prob >= T and InCache == 0:
                self.Caches[Id] = Size
                self.b_recency[Id] = self.LCurrent + 1  / Size * Prob
                self.access_freq[Id] = 1
                self.UsedSpace += Size
        while self.UsedSpace > self.space:
            self.evict()
        self.timer += 1



















