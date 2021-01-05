import CacheSysLRB as CS

class Frame:
    def __init__(self,cache_size,trace,k=10000):
        self.CacheSize = cache_size
        self.Trace = trace
        self.K = k
        self.Dataset = []
        self.CacheSys = CS.Cache_Sys(cache_size)
        self.Last_Req_Time = {}
        self.PastFeas = {}
        self.Deltas = {}
        self.ECDs = {}
        self.Trained = 0
        self.Prediction = None
        self.Decision = []
        self.PastDis = {}
        self.Hash_EDC = {}
        self.process_data()

    def process_data(self):
        timer = 0
        for i in range(len(self.Trace)):
            data = self.Trace[i]
            req = {}
            req['id'] = data[1]
            req['timestamp'] = timer
            req['size'] = data[2]
            self.Dataset.append(req)
            timer += 1

    def main(self):
        DynReqs = []
        Timer = 0
        HITs = []

        for i in range(len(self.Dataset)):
            req = self.Dataset[i]
            self.CacheSys.admit(req)
            DynReqs.append(req)

            if len(DynReqs) % self.K == 0 and len(DynReqs) > 0:
                Hits_Now = self.CacheSys.hits
                HITs.append(Hits_Now)
                print('Hits:',Hits_Now)
                self.CacheSys.reset()
                DynReqs = []
            Timer += 1

        return HITs

