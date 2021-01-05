import numpy as np
import copy as cp
import Models
import CacheSysRLBP as CS
import LSO as GL

class Frame:
    def __init__(self, cache_size, trace, k=10000):
        self.CacheSize = cache_size
        self.Trace = trace
        self.K = k
        self.Dataset = []
        self.CacheSys = CS.Cache_Sys(cache_size)
        self.process_data()
        self.Last_Req_Time = {}
        self.PastFeas = {}
        self.Deltas = {}
        self.ECDs = {}
        self.Trained = 0
        self.Prediction = None
        self.Decision = []
        self.Adm = None

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

    def extractor(self, req):
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
                ecd_now = 1.0 + old_ecds[j] * pow(2, -delta / pow(2, 10 + j))
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

    def edm(self, feas):
        Num = len(self.Decision)
        Sum = 0
        for i in range(Num):
            Sum += (i + 1)
        decision = 0.0
        for i in range(Num):
            model = self.Decision[i]
            res = model.predict(np.array([feas]))
            decision += res[0] * (i + 1) / Sum
        return decision

    def main(self):
        WinReqs = []
        WinFeas = []
        WinLabels = []
        TrainX = []
        TrainY = []
        WinDecisions = []
        WinNRTs = []
        DynReqs = []
        HisReqs = []
        Timer = 0
        CSys = [cp.deepcopy(self.CacheSys)]
        DHits = []
        HITs = []
        num_req = 0

        for i in range(len(self.Dataset)):
            req = self.Dataset[i]
            Id = req['id']
            print(i+1)
            features = self.extractor(req)

            Nrt = 9.3
            Prob = 0.8
            if self.Trained == 1:
                if Id in self.PastFeas.keys():
                    Nrt = self.Prediction.predict(np.array([features]))[0]
            fea_deci = [Nrt]
            for i in range(3):
                fea_deci.append(features[i + i])
            if len(self.Decision) > 0:
                Prob = self.edm(fea_deci)

            self.CacheSys.admit(req, Prob, Nrt)
            WinDecisions.append(Prob)

            if Id in self.PastFeas.keys():
                Pastfeas = self.PastFeas[Id]
                WinFeas.append(Pastfeas)
                Label = np.log(Timer - self.Last_Req_Time[Id])
                WinLabels.append(Label)

            self.PastFeas[Id] = features
            TrainX.append(fea_deci)
            WinReqs.append(req)
            WinNRTs.append(Nrt)
            DynReqs.append(req)
            HisReqs.append(req)
            num_req += 1

            if len(WinReqs) % self.K == 0 and len(WinReqs) > 0:
                Hits_Now = self.CacheSys.hits
                DHits.append(Hits_Now)
                HITs.append(Hits_Now)
                self.CacheSys.reset()
                New_CSys = cp.deepcopy(self.CacheSys)
                CSys.append(New_CSys)

                Pred = Models.XGBM(WinFeas, WinLabels)
                self.Prediction = Pred.model()
                self.Trained = 1

                WinFeas = []
                WinLabels = []
                DynReqs = []
                LSO_Compt = GL.LSO(WinReqs, WinNRTs, CSys[0])
                LSO_Compt.limits(self.CacheSize)
                Iter = 0
                NumMC = 5
                Bad = 0
                Good = 0
                Hits_Now = Hits_Now + 10
                while Iter < NumMC:
                    Hits_New, Sol_New, GooN = LSO_Compt.get_opt(Hits_Now)
                    if Hits_New > Hits_Now:
                        if len(self.Decision) == 0:
                            Dec = Models.FNN(TrainX[:self.K], Sol_New[:self.K])
                            self.Adm = Dec.model()
                            Deci = self.Adm
                            self.Decision.append(Deci)
                        else:
                            self.Adm.fit(np.array(TrainX[:self.K]), np.array(Sol_New[:self.K]))
                            Deci = self.Adm
                            self.Decision.append(Deci)
                        #Dec = Models.XGBM(TrainX[:self.K], Sol_New[:self.K])
                        #Deci = Dec.model()
                        #self.Decision.append(Deci)
                        if len(self.Decision) > 5:
                            self.Decision = self.Decision[1:]
                        Good += 1
                        Hits_Now = Hits_New
                    else:
                        Bad += 1

                    if Good == 3:
                        break
                    if Bad == 2 and Good == 0:
                        break
                    if GooN == 0:
                        break

                    Iter += 1

                CSys = CSys[1:]
                TrainX = []
                WinDecisions = []
                WinNRTs = []
                WinReqs = []
            self.Last_Req_Time[Id] = Timer
            Timer += 1

        return HITs
