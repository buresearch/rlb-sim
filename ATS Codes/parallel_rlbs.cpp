#include "parallel_rlbs.h"

void ParallelRLBCache::train_admission(vector<vector<float>> feas, vector<float> decisions){
    int cols2=feas[0].size(),rows2=feas.size();
    float train_deci[rows2][cols2];
    float train_labels_deci[cols2];
    for(int i=0;i<rows2;i++)
    {
        for(int j=0;j<cols2;j++)
        {
	   train_deci[i][j] = feas[i][j];
        }
	   train_labels_deci[i] = decisions[i];
    }

    int num_one = 0;
    for(int i=0;i<decisions.size();i++){
      float opt_now = decisions[i];
      if(opt_now > 0.5)
           num_one++;
      if(i<200){
            outOpt<<opt_now<<" ";
      }
    }	

    DMatrixHandle h_train_deci[1];
    XGDMatrixCreateFromMat((float *) train_deci, rows2, cols2, -1, &h_train_deci[0]);
    XGDMatrixSetFloatInfo(h_train_deci[0], "label", train_labels_deci, rows2);

    BoosterHandle h_booster=nullptr;
    XGBoosterCreate(h_train_deci, 1, &h_booster);
    XGBoosterSetParam(h_booster, "booster", "gbtree");
    XGBoosterSetParam(h_booster, "objective", "reg:squarederror")
    XGBoosterSetParam(h_booster, "max_depth", "10");
    XGBoosterSetParam(h_booster, "eta", "0.1");
    XGBoosterSetParam(h_booster, "min_child_weight", "1");
    XGBoosterSetParam(h_booster, "subsample", "0.5");
    XGBoosterSetParam(h_booster, "colsample_bytree", "1");
    XGBoosterSetParam(h_booster, "num_parallel_tree", "1");
    XGBoosterSetParam(h_booster, "num_threads", "4");

    for (int iter=0; iter<50; iter++)
        XGBoosterUpdateOneIter(h_booster, iter, h_train_deci[0]);
    std::swap(admit,h_booster);
    Trained_Adm = 1;
       
    XGDMatrixFree(h_train_deci[0]);
    XGBoosterFree(h_booster);
}

void ParallelRLBCache::train(){
    training_data_mutex.lock();
    vector<vector<uint64_t>> G_CREQs;
    vector<float> G_CFEAs;
    vector<vector<float>> G_CXs;
    vector<vector<float>> G_DFEAs;
    vector<vector<uint64_t>> G_REQs;
    vector<float> G_Decisions;
    vector<float> G_CObjs;
    vector<float> G_NRTs;
    unordered_map<uint64_t,float> G_BRecency = Init_BRecency;
    Init_BRecency = B_Recency;
    map<uint64_t, uint64_t> G_LRTs = InitLRTs;
    InitLRTs = DynLRTs;
    map<uint64_t,uint64_t> G_I_Caches = Init_Caches;
    Init_Caches = Dyn_Caches;
    uint64_t Hits_now = Dyn_Hits;
    Dyn_Hits = 0;
    map<uint64_t,uint64_t> G_AFreq = Init_AccessFreq;
    Init_AccessFreq = AcFq;
    uint64_t G_UP = Init_UsedSpace;
    Init_UsedSpace = _currentSize;
    float G_LCR = Init_LCurrent;
    Init_LCurrent = LCurrent;
    
    std::swap(G_REQs,PastReqs);
    std::swap(G_CREQs,CReqs);
    std::swap(G_CXs,CXs);
    std::swap(G_DFEAs,DFeas);
    std::swap(G_Decisions,Decisions);
    std::swap(G_CObjs,CObjs);
    std::swap(G_NRTs,NRTs);
    
    int K = G_CREQs.size();
    vector<float>().swap(Decisions);
    vector<float>().swap(CObjs);
    vector<float>().swap(NRTs);
    vector<vector<float>>().swap(CXs);
    vector<vector<float>>().swap(DFeas);
    vector<vector<uint64_t>>().swap(CReqs);
    vector<vector<uint64_t>>().swap(PastReqs);
    
    uint64_t fix_timer = init_timer;
    init_timer = timer;
    
    training_data_mutex.unlock();
    uint64_t g_up = 0;
    auto It = G_I_Caches.begin();
    while(It != G_I_Caches.end()){
        uint64_t gsize = It->second;
        g_up +=gsize;
        It++;
    }
    
    uint64_t max_training = 5;
    uint64_t iters = 0;

    Cache_Sim cachesys_all;
    cachesys_all.init(_cacheSize,G_I_Caches,fix_timer,G_BRecency,G_LRTs,G_AFreq,G_LCR);
    cachesys_all.reset();
    vector<float> AllOPTs;
    for(int i=0;i<G_CREQs.size();i++){
    	  vector<uint64_t> req_now = G_CREQs[i];
    	  float nrt_now	= G_NRTs[i];
    	  AllOPTs.push_back(1.0);
    	  cachesys_all.decide(req_now,1.0,nrt_now);
    }
    uint64_t Hits_all = cachesys_all.hits;
    Hits_now = Hits_now + 50;
    bool trained_all = 0;
    if(Hits_all > Hits_now){
        Trained_Adm = 0;
        Hits_now = Hits_all + 50;
    }
    
    if(trained_all == 0){
       LSO LSO_Compt;
       LSO_Compt.init(_cacheSize,G_I_Caches,fix_timer,G_BRecency,G_LRTs,G_AFreq);
       LSO_Compt.get_limit(G_CREQs,G_NRTs);
       out5.close();

       while(iters < max_training){
    	     LSO_Compt.lso(G_CREQs,G_NRTs,G_CREQs.size(),G_LCR);
	         if(Hits_now < LSO_Compt.Hits){
	         Hits_now = LSO_Compt.Hits;
       	       vector<float> OPT_Now = LSO_Compt.OPTs;
       	       train_admission(G_DFEAs,OPT_Now);
       	  }
       	 ++iters;
        }
     }

    int cols1=G_CXs[0].size(),rows1=int(G_CXs.size());
    float train_nrt[rows1][cols1];
    float train_labels_nrt[rows1];
    for(int i=0;i<rows1;i++){
      for(int j=0;j<cols1;j++){
	 train_nrt[i][j] = G_CXs[i][j];
      } 
      train_labels_nrt[i] = G_CObjs[i];
    }
    	
    DMatrixHandle h_train_nrt[1];
    XGDMatrixCreateFromMat((float *) train_nrt, rows1, cols1, -1, &h_train_nrt[0]);
    XGDMatrixSetFloatInfo(h_train_nrt[0], "label", train_labels_nrt, rows1);

    BoosterHandle h_booster_nrt=nullptr;
    XGBoosterCreate(h_train_nrt, 1, &h_booster_nrt);
    XGBoosterSetParam(h_booster_nrt, "booster", "gbtree");
    XGBoosterSetParam(h_booster_nrt, "objective", "reg:squarederror");
    XGBoosterSetParam(h_booster_nrt, "max_depth", "10");
    XGBoosterSetParam(h_booster_nrt, "eta", "0.1");
    XGBoosterSetParam(h_booster_nrt, "min_child_weight", "1");
    XGBoosterSetParam(h_booster_nrt, "subsample", "0.5");
    XGBoosterSetParam(h_booster_nrt, "colsample_bytree", "1");
    XGBoosterSetParam(h_booster_nrt, "num_parallel_tree", "1");
    XGBoosterSetParam(h_booster_nrt, "num_threads", "4");

    for (int iter=0; iter<50; iter++)
      XGBoosterUpdateOneIter(h_booster_nrt, iter, h_train_nrt[0]);

    booster_mutex.lock();
    std::swap(nrt_predict, h_booster_nrt);
    booster_mutex.unlock();
    XGDMatrixFree(h_train_nrt[0]);
    XGBoosterFree(h_booster_nrt);

    vector<float>().swap(G_CFEAs);
    vector<float>().swap(G_Decisions);
    vector<float>().swap(G_CObjs);
    vector<vector<float>>().swap(G_CXs);
    vector<vector<float>>().swap(G_DFEAs);
    vector<vector<uint64_t>>().swap(G_CREQs);
    vector<vector<uint64_t>>().swap(G_REQs);
    vector<float>().swap(G_NRTs);
        
    training_data_mutex.unlock();

}

vector<float> ParallelRLBCache::extractor(vector<uint64_t> Req)
{
    uint64_t Id = (uint64_t)Req[1];
    uint64_t time = (uint64_t)Req[0];
    float size = (uint64_t)Req[2] / 1.0;

    bool In = 0;
    auto it = DynLRTs.find(Id);
    if(it != DynLRTs.end())
    {
       In = 1;
    }
    
    if(In == 0)
    {
        vector<float> deltas;
        vector<float> ecds;
        for(int i=0;i<32;i++)
            deltas.push_back(pow(10,10));
        for(int j=0;j<10;j++)
            ecds.push_back(0);
        Deltas[Id] = deltas;
        ECDs[Id] = ecds;
    }
    
    if(In == 1)
    {
        uint64_t last_time = DynLRTs[Id];
        float delta = time - last_time;
        vector<float> old_deltas = Deltas[Id];
        vector<float> old_ecds = ECDs[Id];
        vector<float> new_deltas;
        vector<float> new_ecds;
        new_deltas.push_back(delta);
        for(int i=0;i<old_deltas.size()-1;i++)
            new_deltas.push_back(old_deltas[i]);
        for(int j=0;j<10;j++)
        {
            float ecd_now = 1.0 + old_ecds[j] * pow(2,-delta / pow(2,10+j));
            new_ecds.push_back(ecd_now);
        }
        Deltas[Id] = new_deltas;
        ECDs[Id] = new_ecds;
    }
    
    vector<float> get_deltas = Deltas[Id];
    vector<float> get_ecds = ECDs[Id];
    float get_size = size;
    vector<float> Features;
    Features.push_back(get_size);
    for(int i=0;i<get_deltas.size();i++)
        Features.push_back(get_deltas[i]);
    for(int j=0;j<10;j++)
        Features.push_back(get_ecds[j]);

    return Features;
}
    
void ParallelRLBCache::async_lookup(const uint64_t &key) {
    auto InCache = key_map.find(key);
    if (InCache != key_map.end()) {
	  training_data_mutex.lock();
	  Dyn_Hits++;
	  uint64_t size = All_Sizes[key];
    vector<uint64_t> req_now;
	  req_now.push_back(timer);
	  req_now.push_back(key);
	  req_now.push_back(size);
    vector<float> Fea = extractor(req_now);

    float nrt = 9.3;
    if(Trained == 1 && DynLRTs.find(key) != DynLRTs.end()){
           const int sample_rows = 1;
           int cols = Fea.size();
           float test[sample_rows][cols];
           for (int j=0;j<cols;j++)
           	test[0][j] = Fea[j];
           DMatrixHandle h_test;
           XGDMatrixCreateFromMat((float *) test, sample_rows, cols, -1, &h_test);
           int out_len=0;
           bst_ulong f=1;
           const float *res;
           XGBoosterPredict(nrt_predict, h_test,0 ,0, 1,&f,&res);
           XGDMatrixFree(h_test);
           nrt = res[0];
    	}

     float decision = 0.8;
     if(Trained == 1 && Trained_Adm == 1){
    	    const int sample_rows = 1;
          int cols = 6;
          float test[sample_rows][cols];
          for (int j=0;j<cols-1;j++){
           	test[0][j] = Fea[j+1];
          }
          test[0][cols-1] = nrt;
          DMatrixHandle h_test;
          XGDMatrixCreateFromMat((float *) test, sample_rows, cols, -1, &h_test);
          int out_len=0;
          bst_ulong f=1;
          const float *res;
          XGBoosterPredict(admit, h_test,0 , 0, 1, &f, &res);
          decision = res[0];
          XGDMatrixFree(h_test);      
    	}

    PastReqs.push_back(req_now);
    CReqs.push_back(req_now);
    if(DynLRTs.find(key) != DynLRTs.end()){
         vector<float> PastFeature = PastFeas[key];
    	   uint64_t CLRT = DynLRTs[key];
    	   float CY = log(timer-CLRT+0.01);
    	   CObjs.push_back(CY);
    	   CXs.push_back(PastFeature);					      		}

    	   CFeas.push_back(Fea);
    	   DynLRTs[key] = timer;
    	   PastFeas[key] = Fea;
    	   vector<float> DFea_now;
    	   for (int j=0;j<5;j++){
    		    DFea_now.push_back(Fea[j+1]);
    	   }
    	   DFea_now.push_back(nrt);
    	   DFeas.push_back(DFea_now);

        if(AcFq.find(key)!=AcFq.end()){
            AcFq[key] += 1;
        }
        else{
            AcFq[key] = 1;
        }
        float get_size = std::max(size / 1024 + 0.1,1.0);
        float update_val = LCurrent + AcFq[key] * decision / get_size / 1.0;//
  
        if(B_Recency.find(key) != B_Recency.end()){
             B_Recency.erase(key);
             unordered_map<uint64_t,float>(B_Recency).swap(B_Recency);
        }
        B_Recency[key] = update_val;
    	
    	  NRTs.push_back(nrt);
    	  Decisions.push_back(decision);
	      ++timer;
    }
    training_data_mutex.unlock();
}

void ParallelRLBSCache::async_admit(const uint64_t &key, const int64_t &size, const uint16_t *extra_features) {
     auto InCache = key_map.find(key);
     if (InCache == key_map.end()) {  
	      training_data_mutex.lock();

    	  All_Sizes[key] = size;
    	  vector<uint64_t> req_now;
	      req_now.push_back(timer);
	      req_now.push_back(key);
	      req_now.push_back(size);

    	  vector<float> Fea = extractor(req_now);

    	  float nrt = 9.3;
    	  if(Trained == 1 && DynLRTs.find(key) != DynLRTs.end()){
        	const int sample_rows = 1;
        	int cols = Fea.size();
        	float test[sample_rows][cols];
        	for (int j=0;j<cols;j++)
           	    test[0][j] = Fea[j];
        	DMatrixHandle h_test;
        	XGDMatrixCreateFromMat((float *) test, sample_rows, cols, -1, &h_test);
        	int out_len=0;
        	bst_ulong f=1;
        	const float *res;
        	XGBoosterPredict(nrt_predict, h_test,0 ,0,1,&f,&res);
        	XGDMatrixFree(h_test);
        	nrt = res[0];
    	   }

    	  float decision = 0.8;
        if(Trained == 1 && Trained_Adm == 1){
    		  const int sample_rows = 1;
        	int cols = 6;
        	float test[sample_rows][cols];
        	for (int j=0;j<cols-1;j++){
           		test[0][j] = Fea[j+1];
        	}
        	test[0][cols-1] = nrt;
        	DMatrixHandle h_test;
        	XGDMatrixCreateFromMat((float *) test, sample_rows, cols, -1, &h_test);
        	int out_len=0;
        	bst_ulong f=1;
          const float *res;
          XGBoosterPredict(admit, h_test, 0 ,0, 1, &f, &res);
          decision = res[0];
          XGDMatrixFree(h_test);
                
    	}

    	PastReqs.push_back(req_now);
    	CReqs.push_back(req_now);

    	if(DynLRTs.find(key) != DynLRTs.end()){
            vector<float> PastFeature = PastFeas[key];
    	    uint64_t CLRT = DynLRTs[key];
    	    float CY = log(timer-CLRT+0.01);
    	    CObjs.push_back(CY);
    	    CXs.push_back(PastFeature);
   	 }

    	CFeas.push_back(Fea);
    	DynLRTs[key] = timer;
    	PastFeas[key] = Fea;
    	vector<float> DFea_now;
    	for (int j=0;j<5;j++){
            DFea_now.push_back(Fea[j+1]);
    	}
    	DFea_now.push_back(nrt);
    	DFeas.push_back(DFea_now);
    	NRTs.push_back(nrt);
    	Decisions.push_back(decision);

	    training_data_mutex.unlock();

    	if (decision >= admit_threshold) {
             Dyn_Caches[key] = size;
             if(AcFq.find(key)!=AcFq.end()){
                 AcFq[key] += 1;
             }
             else{
                 AcFq[key] = 1;
             }
             float get_size = std::max(size / 1024 + 0.1,1.0);
             float update_val = LCurrent + AcFq[key] * decision / get_size / 1.0;//
             if(B_Recency.find(key) != B_Recency.end()){
                 B_Recency.erase(key);
                 unordered_map<uint64_t,float>(B_Recency).swap(B_Recency);
             }
             B_Recency[key] = update_val;
             key_map.insert({key, KeyMapEntry{.list_idx=0, .list_pos = (uint32_t) in_cache_metas.size()}});
             auto shard_id = key%n_shard;
             size_map_mutex[shard_id].lock();
             size_map[shard_id].insert({key, size});
             size_map_mutex[shard_id].unlock();
             auto lru_it = in_cache_lru_queue.request(key);
             in_cache_metas.emplace_back(key, size, timer, extra_features, lru_it);
             _currentSize += size;
      }

    	while (_currentSize > _cacheSize) {
        	evict();
    	}
        ++timer;
    }
}

pair<uint64_t, uint32_t> ParallelRLBSCache::rank() {
    uint64_t timer_fix = timer;	   
    float find_min_val = -100000.0;
    uint64_t find_min_key = -1;
    auto It = B_Recency.begin();
    while(It != B_Recency.end()){
      uint64_t Id = It->first;
      float val = It->second; 
      if(find_min_val >= val){
         find_min_val = val;
         find_min_key = Id;
       } 
      It++;
    }
    uint64_t delete_id = find_min_key;
    uint64_t find_key = -1;
    uint32_t find_pos = -1;

    auto it = key_map.find(delete_id);
    find_pos = it->second.list_pos;
    auto &meta = in_cache_metas[find_pos];
    find_key = meta._key;	
    LCurrent = B_Recency[find_key];
    B_Recency.erase(find_key);
    Dyn_Caches.erase(find_key);
    AcFq.erase(find_key);

    return {find_key, find_pos};
    
}

void ParallelRLBSCache::evict() {
    auto epair = rank();
    uint64_t &key = epair.first;
    uint32_t &old_pos = epair.second;

    auto &meta = in_cache_metas[old_pos];
    if (!meta._sample_times.empty()) {
        meta._sample_times.clear();
        meta._sample_times.shrink_to_fit();
    }
    in_cache_lru_queue.dq.erase(meta.p_last_request);
    meta.p_last_request = in_cache_lru_queue.dq.end();
    meta.free();
    _currentSize -= meta._size;
    key_map.erase(key);

    auto shard_id = key%n_shard;
    size_map_mutex[shard_id].lock();
    size_map[shard_id].erase(key);
    size_map_mutex[shard_id].unlock();

    uint32_t activate_tail_idx = in_cache_metas.size() - 1;
    if (old_pos != activate_tail_idx) {
        in_cache_metas[old_pos] = in_cache_metas[activate_tail_idx];
        key_map.find(in_cache_metas[activate_tail_idx]._key)->second.list_pos = old_pos;
     }
    in_cache_metas.pop_back();
}
