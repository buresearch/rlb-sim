import numpy as np
import time
import Model

class Cache_Sys:
    def __init__(self,space):
        self.space = space
        self.UsedSpace = 0
        self.Caches = {}
        self.recency = {}
        self.timer = 0
        self.hits = 0
        self.misses = 0
        self.Prediction = None
        self.Last_Req_Time = {}
        self.PastFeas = {}
        self.Deltas = {}
        self.ECDs = {}
        self.TrainX = []
        self.TrainY = []
        self.DynReqs = []

    def extractor(self,req):
        Id = req['id']
        Time = req['timestamp']
        Size = req['size']
        InReq = 0
        if Id in self.Last_Req_Time.keys():
            InReq = 1
        if InReq == 0:
            deltas = []
            ecds = []
            for i in range(32):
                deltas.append(pow(10,10))
            for j in range(10):
                ecds.append(0.0)
            self.Deltas[Id] = deltas
            self.ECDs[Id] = ecds
        if InReq == 1:
            Lrt = self.Last_Req_Time[Id]
            delta = Time - Lrt
            old_deltas = self.Deltas[Id]
            old_ecds = self.ECDs[Id]
            deltas = []
            ecds = []
            deltas.append(delta)
            for i in range(31):
                deltas.append(old_deltas[i])
            for j in range(10):
                last = pow(2,-delta/pow(2,10+j))
                ecd_now = 1+old_ecds[j] * last
                ecds.append(ecd_now)
            self.Deltas[Id] = deltas
            self.ECDs[Id] = ecds

        get_deltas = self.Deltas[Id]
        get_ecds = self.ECDs[Id]
        get_size = Size
        features = [get_size]
        for i in range(32):
            features.append(get_deltas[i])
        for j in range(10):
            features.append(get_ecds[j])
        return features

    def train(self):
        Pred = Model.XGBM(self.TrainX, self.TrainY)
        self.Prediction = Pred.model()
        self.TrainX = []
        self.TrainY = []

    def reset(self):
        self.hits = 0
        self.misses = 0

    def rank(self):
        find_key = None
        find_val = 0
        if self.Prediction == None:
            for ky in self.Caches.keys():
                lrt = self.timer - self.Last_Req_Time[ky]
                if lrt > find_val:
                    find_val = lrt
                    find_key = ky
            return find_key

        Features = []
        Keys = []
        Kys = list(self.Caches.keys())
        for l in range(len(Kys)):
            ky = Kys[l]
            Keys.append(ky)
            feas_now = self.PastFeas[ky]
            lrt = self.timer - self.Last_Req_Time[ky]
            feas_new = [feas_now[0],lrt] + feas_now[1:32]
            feas_new += feas_now[-10:]
            Features.append(feas_new)

        Preds = self.Prediction.predict(Features)
        find_val = 0
        for i in range(len(Preds)):
            pred = Preds[i]
            if pred > find_val:
                find_val = pred
                find_key = Keys[i]
        return find_key

    def evict(self):
        find_key = self.rank()
        self.UsedSpace -= self.Caches[find_key]
        self.Caches.pop(find_key)

    def admit(self,req):
        Id = req['id']
        Size = req['size']
        self.DynReqs.append(Id)
        print(self.timer)

        features = self.extractor(req)
        if Id in self.Last_Req_Time.keys():
            Pastfeas = self.PastFeas[Id]
            Label = np.log1p(self.timer - self.Last_Req_Time[Id])
            self.TrainX.append(Pastfeas)
            self.TrainY.append(Label)
        self.PastFeas[Id] = features
        self.Last_Req_Time[Id] = self.timer

        if len(self.DynReqs) == 10000:
            self.train()
            self.DynReqs = []

        InCache = 0
        if Id in self.Caches.keys():
            InCache = 1
        if InCache == 1:
            self.hits += 1
            self.UsedSpace += Size - self.Caches[Id]
            self.Caches[Id] = Size
        else:
            self.misses += 1

        if InCache == 0:
            self.Caches[Id] = Size
            self.UsedSpace += Size

        while self.UsedSpace > self.space:
            self.evict()

        self.timer += 1


