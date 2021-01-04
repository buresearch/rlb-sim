//
// Created by gangyan on 01/04/21.
//

#ifndef WEBCACHESIM_PARALLEL_RLBS_H
#define WEBCACHESIM_PARALLEL_RLBS_H

//Due to the diffciulty of using Neural Networks in C++, we directly use XGBoost as the admission model
//For simplicity, here set the number of models in EDM is one.
//The gap between C++ and python has not been overcomed in this code, but we will dovote more time in the future to inprove it.
//Due to the gap, some techniques are impelmented to make sure the performance.

#include "parallel_cache.h"
#include <atomic>
#include <unordered_map>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <assert.h>
#include <xgboost/c_api.h>
#include <mutex>
#include <thread>
#include <queue>
#include <shared_mutex>
#include <list>
#include <cmath>
#include "sparsepp/spp.h"
#include <map>
#include <fstream>
#include <exception>

using namespace webcachesim;
using namespace std;
using spp::sparse_hash_map;
typedef uint64_t RLBSKey;
using bsoncxx::builder::basic::kvp;
using bsoncxx::builder::basic::sub_array;

namespace ParallelRLBS {
    uint8_t max_n_past_timestamps = 32;
    uint8_t max_n_past_distances = 31;
    uint8_t base_edc_window = 10;
    const uint8_t n_edc_feature = 10;
    vector<uint32_t> edc_windows;
    vector<double> hash_edc;
    uint32_t max_hash_edc_idx;
    uint32_t memory_window = 1;
    uint32_t n_extra_fields = 0;
    uint32_t batch_size = 10000;
    const uint max_n_extra_feature = 4;
    uint32_t n_feature;
}

struct ParalllelRLBSMetaExtra {
    float _edc[10];
    vector<uint32_t> _past_distances;
    uint8_t _past_distance_idx = 1;

    ParalllelRLBSMetaExtra(const uint32_t &distance) {
        _past_distances = vector<uint32_t>(1, distance);
        for (uint8_t i = 0; i < ParallelRLBS::n_edc_feature; ++i) {
            uint32_t _distance_idx = min(uint32_t(distance / ParallelRLBS::edc_windows[i]),
                                         ParallelRLBS::max_hash_edc_idx);
            _edc[i] = ParallelRLBS::hash_edc[_distance_idx] + 1;
        }
    }

    void update(const uint32_t &distance) {
        uint8_t distance_idx = _past_distance_idx % ParallelRLBS::max_n_past_distances;
        if (_past_distances.size() < ParallelRLBS::max_n_past_distances)
            _past_distances.emplace_back(distance);
        else
            _past_distances[distance_idx] = distance;
        assert(_past_distances.size() <= ParallelRLBS::max_n_past_distances);
        _past_distance_idx = _past_distance_idx + (uint8_t) 1;
        if (_past_distance_idx >= ParallelRLBS::max_n_past_distances * 2)
            _past_distance_idx -= ParallelRLBS::max_n_past_distances;
        for (uint8_t i = 0; i < ParallelRLBS::n_edc_feature; ++i) {
            uint32_t _distance_idx = min(uint32_t(distance / ParallelRLBS::edc_windows[i]),
                                         ParallelRLBS::max_hash_edc_idx);
            _edc[i] = _edc[i] * ParallelRLBS::hash_edc[_distance_idx] + 1;
        }
    }
};

class ParallelRLBSMeta {
public:
    uint64_t _key;
    uint32_t _size;
    uint32_t _past_timestamp;
    uint16_t _extra_features[ParallelRLBS::max_n_extra_feature];
    ParalllelRLBSMetaExtra *_extra = nullptr;
    vector<uint32_t> _sample_times;


    ParallelRLBSMeta(const uint64_t &key, const uint64_t &size, const uint64_t &past_timestamp,
                    const uint16_t *&extra_features) {
        _key = key;
        _size = size;
        _past_timestamp = past_timestamp;
        for (int i = 0; i < ParallelRLBS::n_extra_fields; ++i)
            _extra_features[i] = extra_features[i];
    }

    virtual ~ParallelRLBSMeta() = default;

    void emplace_sample(uint32_t &sample_t) {
        _sample_times.emplace_back(sample_t);
    }

    void free() {
        delete _extra;
    }

    void update(const uint32_t &past_timestamp) {
        uint32_t _distance = past_timestamp - _past_timestamp;
        assert(_distance);
        if (!_extra) {
            _extra = new ParalllelRLBSMetaExtra(_distance);
        } else
            _extra->update(_distance);
        _past_timestamp = past_timestamp;
    }

    int feature_overhead() {
        int ret = sizeof(ParallelRLBSMeta);
        if (_extra)
            ret += sizeof(ParalllelRLBSMetaExtra) - sizeof(_sample_times) +
                   _extra->_past_distances.capacity() * sizeof(uint32_t);
        return ret;
    }

    int sample_overhead() {
        return sizeof(_sample_times) + sizeof(uint32_t) * _sample_times.capacity();
    }
};


class ParallelInCacheMeta : public ParallelRLBSMeta {
public:
    list<RLBSKey>::const_iterator p_last_request;

    ParallelInCacheMeta(const uint64_t &key,
                        const uint64_t &size,
                        const uint64_t &past_timestamp,
                        const uint16_t *&extra_features, const list<RLBSKey>::const_iterator &it) :
            ParallelRLBSMeta(key, size, past_timestamp, extra_features) {
        p_last_request = it;
    };

    ParallelInCacheMeta(const ParallelRLBSMeta &meta, const list<RLBSKey>::const_iterator &it) : ParallelRLBSMeta(meta) {
        p_last_request = it;
    };

};

class ParallelInCacheLRUQueue {
public:
    list<RLBSKey> dq;

    list<RLBSKey>::const_iterator request(RLBSKey key) {
        dq.emplace_front(key);
        return dq.cbegin();
    }

    list<RLBSKey>::const_iterator re_request(list<RLBSKey>::const_iterator it) {
        if (it != dq.cbegin()) {
            dq.emplace_front(*it);
            dq.erase(it);
        }
        return dq.cbegin();
    }
};


class Cache_Sim
{
public:
    map<uint64_t,uint64_t> Caches;
    map<uint64_t,uint64_t> LRTs;
    uint64_t CacheSize = 0;
    uint64_t UsedSpace = 0; 
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t timer_h = 0;
    unordered_map<uint64_t,float> BRecency;
    map<uint64_t,uint64_t> Accessfreq;
    float lcurrent;
    
    void init(uint64_t capacity,map<uint64_t,uint64_t> caches, uint64_t timer_now, unordered_map<uint64_t,float> brecency, map<uint64_t,uint64_t> lrts,map<uint64_t,uint64_t> accessfreq,float lcur)//初始化
    {
    	CacheSize = capacity;
    	Caches = caches;
    	auto It = caches.begin();
    	while(It != caches.end()){
        uint64_t size = It->second;
            UsedSpace +=size;
            It++;
    	}
    	timer_h = timer_now;
    	LRTs = lrts;
      BRecency = brecency;
      Accessfreq = accessfreq;
      lcurrent = lcur;
    }

    void reset()
    {
    	hits = 0;
    	misses = 0;
    }
		
    uint64_t rank(){
	     uint64_t find_key = -1;
	     float find_val = -10000000.0;
	     auto It = BRecency.begin();
    	 while(It != BRecency.end()){
    	   uint64_t key = It->first;
         float val = float(It->second);
 	       if(val <= find_val){
 	          find_val = val;
 	          find_key = key;	
 	       }
           It++;
    	  }
			
	     return find_key;
     }
		
     void evict(){
	      uint64_t find_key = rank();
	      UsedSpace -= Caches[find_key];
	      Caches.erase(find_key);
        lcurrent = BRecency[find_key];
	      BRecency.erase(find_key);
        Accessfreq.erase(find_key);
     }

    void decide(vector<uint64_t> Req, float prob, float nrt)
    {
      	uint64_t Id = Req[1];
    	  uint64_t Time = Req[0];
    	  uint64_t Size = Req[2];
    	  float Prob = prob;
    	  float T = 0.5;
    	  float Nrt = nrt;
        ++timer_h;

        if(Size > 131072000){
            Prob = 0.2;
        }
    	
    	  bool In = 0;
    	 if(Caches.find(Id) != Caches.end()){
           In = 1;
    	 }
    
    	 if(In == 1)
    	 {
           ++hits;
           BRecency.erase(Id);
           unordered_map<uint64_t,float>(BRecency).swap(BRecency);
           Accessfreq[Id] += 1;
           float get_size = std::max(Size / 1024 + 0.1,1.0);
           BRecency[Id] = exp(nrt);//lcurrent + Accessfreq[Id] * Prob / get_size /1.0;//
           UsedSpace = UsedSpace + Size - Caches[Id];
           Caches[Id] = Size;
    	 }
    	 else
    	 {
           ++misses;
    	 }
    
    	 if(Prob >= T && In == 0)
    	 {
           Accessfreq[Id] = 1;
           Caches[Id] = Size;
           float get_size = std::max(Size / 1024 + 0.1,1.0);
           BRecency[Id] = lcurrent + Accessfreq[Id] * Prob / get_size / 1.0;//
           UsedSpace += Size;
    	}

      LRTs[Id] = timer_h;
    	while(UsedSpace > CacheSize)
    	    evict();
    }
};

class Assist
{
public:
    uint64_t Cachesize;
    uint64_t Usedspace = 0;
    unordered_map<uint64_t,uint64_t> Caches;
    map<uint64_t,float> Nrts;
    map<uint64_t,uint64_t> Lrts;
    uint64_t timer_h = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    map<uint64_t,uint64_t> AcFq;
    
    void Init(uint64_t cachesize){
        Cachesize = cachesize;
    }

    void reset(){
       hits = 0;
       misses = 0;
    }

    uint64_t rank(){
        uint64_t find_key;
	      float find_val = -1;
	      auto It = Caches.begin();
    	  while(It != Caches.end()){
    	     uint64_t key = It->first;
           float val = It->second;
           float lrt = float(timer_h - Lrts[key]);
           uint64_t size = Caches[key];
           float val_now = std::max(val,lrt) * size;
 	      if(val_now >= find_val){
 	         find_val = val_now;
 	         find_key = key;	
 	      }
        It++;
    	 }
			
	    return find_key;
    }

    uint64_t evict(){
        uint64_t find_key = rank();
	      Usedspace -= Caches[find_key];
	      Caches.erase(find_key);
        return find_key;
    }

   vector<uint64_t> admit(vector<uint64_t> Req, float nrt){
      vector<uint64_t> useid;
      uint64_t Id = Req[1];
      uint64_t Size = Req[2];

      bool InCache = 0;
      if(Caches.find(Id) != Caches.end()){
        InCache = 1;
      }
     
      if(InCache==1){
        ++hits;
        Caches.erase(Id);
        Caches[Id] = Size;
        Nrts[Id] = std::exp(nrt);
        AcFq[Id] += 1;
      }
      else{
        ++misses;
      }

      if(InCache==0){
        Caches[Id] = Size;
        Nrts[Id] = std::exp(nrt);
        AcFq[Id] = 1;
        Usedspace += Size;
      }

      Lrts[Id] = timer_h;

      while(Usedspace>Cachesize){
        uint64_t di = evict();
        if(AcFq[di] == 1){
          useid.push_back(di);
        }
      }

      ++timer_h;     

      return useid;
   }

};

class LSO
{
public:
    uint64_t Cachesize;
    vector<float> Lims;
    vector<float> OPTs;
    uint64_t Hits;
    map<uint64_t,uint64_t> Caches;
    uint64_t GTimer;
    unordered_map<uint64_t,float> BRecency;
    map<uint64_t,uint64_t> Lrts;
    map<uint64_t,uint64_t> Accessfreq;
    
	
    void init(uint64_t cachesize, map<uint64_t,uint64_t> caches, uint64_t timer_now, unordered_map<uint64_t,float> brecency, map<uint64_t,uint64_t> lrts,map<uint64_t,uint64_t> accessfreq){
      Caches = caches;
      GTimer = timer_now;
      BRecency = brecency;
      Lrts = lrts;
      Accessfreq = accessfreq;
      Cachesize = cachesize;
	}
	
    void get_limit(vector<vector<uint64_t>> Reqs,vector<float> Pnrts){
    	map<uint64_t,vector<uint64_t>> Occurs;
    	for(int i=0;i<Reqs.size();i++){
           vector<uint64_t> req = Reqs[i];
           uint64_t Time = req[0];
           uint64_t Id = req[1];
           uint64_t Size = req[2];
           Occurs[Id].push_back(Time);
    	}
    
    	map<uint64_t,vector<uint64_t>> NRTs;
    	map<uint64_t,uint64_t> allocate;
      map<uint64_t,float> Est_Nrt;
     
    	map<uint64_t,vector<uint64_t>>::iterator It1;
    	It1 = Occurs.begin();
    	while(It1 != Occurs.end()){
           uint64_t key_now = It1->first;
           allocate[key_now] = 0;
           vector<uint64_t> occurs = Occurs[key_now];
           float est_nrt = 10000.0;
           vector<uint64_t> NRTs_now;
           if(occurs.size()>1){
               float sum_now = 0.0;
               float count_now = 0.0;
               for(int j=0;j<occurs.size()-1;j++){
                  uint64_t nt = occurs[j+1] - occurs[j];
                  sum_now += nt;
                  count_now++;
                  NRTs_now.push_back(nt);
               }
               est_nrt = uint64_t(sum_now / count_now);
           }
           NRTs_now.push_back(est_nrt);
           NRTs[key_now] = NRTs_now;
           Est_Nrt[key_now] = est_nrt;
           It1++;
    	}
    
    	vector<uint64_t> Nexts;
    	for(int l=0;l<Reqs.size();l++){
           vector<uint64_t> req = Reqs[l];
           uint64_t key_now = req[1];
           uint64_t counter_now = allocate[key_now];
           vector<uint64_t> NRTs_now = NRTs[key_now];
           float nrt_now = NRTs_now[counter_now];
           Nexts.push_back(nrt_now);
           allocate[key_now]++;
    	 }

    	vector<uint64_t> Overlaps;
    	vector<uint64_t> Valids_Back;
    	vector<uint64_t> Valid_Overlap;
    	for(int m=0;m<Reqs.size();m++){
           vector<uint64_t> req = Reqs[m];
           uint64_t key_now = req[1];
           uint64_t next_now = Nexts[m];
           uint64_t num = 0;
           uint64_t valid_time = m+next_now;
           if(m+next_now>Reqs.size())
               valid_time = Reqs.size();
           if(m+next_now>Reqs.size()){
               Overlaps.push_back(100000);
           }
           else{
            for(int n=m+1;n<valid_time;n++){
                vector<uint64_t> req_new = Reqs[n];
                uint64_t next_new = Nexts[n];
                if(n+next_new < m+next_now){
                    num++;
                }
             }
            uint64_t size_now = 1.0;
            Overlaps.push_back(num);
            Valid_Overlap.push_back(num);
           }
        }
    
    	Valids_Back = Valid_Overlap;
    	std::sort (Valids_Back.begin(),Valids_Back.begin()+Valids_Back.size());
      
    	uint64_t loc1 = uint64_t(Valids_Back.size()*0.05);
    	uint64_t loc2 = uint64_t(Valids_Back.size()*0.95);
    	uint64_t Per1 = Valids_Back[loc1];
    	uint64_t Per2 = Valids_Back[loc2];

      vector<float> Sols;
      Assist AT;
      AT.Init(Cachesize);
      map<uint64_t,uint64_t> LastLocs;
      for(int i=0;i<Reqs.size();i++){
	        vector<uint64_t> req = Reqs[i];
	        uint64_t Id = req[1];
	        float pnrt = Pnrts[i];
          vector<uint64_t> getid = AT.admit(req,pnrt);
          Sols.push_back(0.8);
          LastLocs[Id] = i;
          for(int j=0;j<getid.size();j++){
                 uint64_t loc = LastLocs[getid[j]];
                 Sols[loc] = 0.2;
            }
      }   

    	for(int i=0;i<Overlaps.size();i++){
           uint64_t lap = Overlaps[i];
           if(lap <= Per1 && lap != -1){
 	            Sols[i] = 1.0;
           }
           else if(lap >= Per2){
              Sols[i] = 0.0;
           }
         }
         Lims = Sols;
}
      void lso(vector<vector<uint64_t>> Reqs, vector<float> Pnrts,uint64_t K,float LCR){      
          Cache_Sim Cachesys;
          Cachesys.init(Cachesize,Caches,GTimer,BRecency,Lrts,Accessfreq,LCR);
          Cachesys.reset();
          vector<float> get_opt;
	        uint64_t NumOne = 0;
	        uint64_t hits = 0;
	        float gamma = 0.8;

          srand((unsigned)time(NULL));
	        for(int i=0;i<Reqs.size();i++){
	            vector<uint64_t> req = Reqs[i];
	            uint64_t Id = req[1];
	            float nrt = K;
	            float decision = 0.8;
              uint64_t lim = Lims[i];
	            if(i < K){
		             nrt = Pnrts[i];	
		             decision = Lims[i];
		             float rd = (rand() % 100000) / 100000.0;
		          if(rd>decision){
                    decision = 0.2;
		           }
                else{
                    decision = 0.8;
                }
	         }
		
	        map<uint64_t,uint64_t> Caches_Now = Cachesys.Caches;
	        bool In = 0;
	        if(Caches_Now.find(Id) != Caches_Now.end())
		          In = 1;
				
	        if(In == 1){
		         if(i < K)
		           hits += 1;
		         else
		           hits += std::pow(gamma,i-K);	
	        }

          get_opt.push_back(decision);
          Cachesys.decide(req,decision,nrt);
        }
        
        Hits = hits;
	      OPTs = get_opt;
}
    
};

struct KeyMapEntry {
    unsigned int list_idx: 1;
    unsigned int list_pos: 31;
};

class ParallelRLBSCache : public ParallelCache {
public:
    sparse_hash_map<uint64_t, KeyMapEntry> key_map;

    vector<ParallelInCacheMeta> in_cache_metas;
    vector<ParallelRLBSMeta> out_cache_metas;

    map<uint64_t,vector<float>> PastFeas;
    Cache_Sim Simulator;
    vector<uint64_t> B_Recency_Key;
    vector<float> B_Recency_Val;
    unordered_map<uint64_t,float> B_Recency;
    unordered_map<uint64_t,float> Init_BRecency;
    vector<float> Decisions;
    vector<float> NRTs;
    vector<vector<uint64_t>> CReqs;
    vector<vector<float>> CFeas;
    vector<float> CObjs;
    vector<vector<float>> CXs;
    vector<vector<float>> DFeas;
    map<uint64_t,uint64_t> Init_Caches;
    map<uint64_t,uint64_t> Dyn_Caches;
    uint64_t Init_UsedSpace = 0;
    uint64_t Init_Hits = 0;
    uint64_t Dyn_Hits = 0;
    vector<vector<uint64_t>> PastReqs;
    float CHits = 0;
    map<uint64_t,uint64_t> All_Sizes;
    float admit_threshold = 0.5;
    map<uint64_t,uint64_t> AccessFreq;
    map<uint64_t,uint64_t> Init_AccessFreq;
    vector<BoosterHandle> Models;
    vector<BoosterHandle> Init_Models;
    uint64_t NumModel = 3;
    float LCurrent = 0.0;
    float Init_LCurrent = 0.0;
    map<uint64_t,uint64_t> AcFq;
    
    map<uint64_t, uint64_t> InitLRTs;
    map<uint64_t, uint64_t> DynLRTs;
    map<uint64_t, vector<float>> Deltas;
    map<uint64_t, vector<float>> ECDs;
		
    uint32_t timer = 0;
    uint64_t init_timer = 0;
    int Trained = 0;
    int Trained_Adm = 0;
    uint64_t NumReq = 0;

    BoosterHandle nrt_predict=nullptr;
    BoosterHandle admit=nullptr;
		
    ParallelInCacheLRUQueue in_cache_lru_queue;
    sparse_hash_map<uint64_t, uint64_t> negative_candidate_queue;
    std::mutex training_data_mutex;
    std::mutex RLBS_mutex;

    std::thread training_thread;

    BoosterHandle booster = nullptr;
    std::mutex booster_mutex;
    bool if_trained = false;

    std::default_random_engine _generator = std::default_random_engine();
    std::uniform_int_distribution<std::size_t> _distribution =         std::uniform_int_distribution<std::size_t>();

    ~ParallelRLBSCache() {
        keep_running = false;
        //outFile.open("/home/gangyan/桌面/Test_Ats.txt",ios::app);
        if (lookup_get_thread.joinable())
            lookup_get_thread.join();
        if (training_thread.joinable())
            training_thread.join();
        if (print_status_thread.joinable())
            print_status_thread.join();
    }

		void init_with_params(const map<string, string> &params) override {
        negative_candidate_queue.reserve(ParallelRLBS::memory_window);
        ParallelRLBS::max_n_past_distances = ParallelRLBS::max_n_past_timestamps - 1;
        ParallelRLBS::edc_windows = vector<uint32_t>(ParallelRLBS::n_edc_feature);
        training_thread = std::thread(&ParallelRLBSCache::async_training, this);
    }


    void print_stats() override {
        std::cerr << "cache size: " << _currentSize << "/" << _cacheSize << " (" << ((double) _currentSize) / _cacheSize
                  << ")" << std::endl
                  << "in/out metadata " << in_cache_metas.size() << " / " << out_cache_metas.size() << std::endl;
    }

    void async_training() {
        while (keep_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (CReqs.size() >= 10000) {
              Trained = 0;
              train();
              Trained = 1;
            }
        }
    }
    
    void async_lookup(const uint64_t &key) override;

    void
    async_admit(const uint64_t &key, const int64_t &size, const uint16_t extra_features[max_n_extra_feature]) override;

    void evict();

    void forget();

    vector<float> extractor(vector<uint64_t> Req);
    
    void train();
    
    void train_admission(vector<vector<float>> feas, vector<float> decisions);


    pair<uint64_t, uint32_t> rank();

    void sample();

    bool has(const uint64_t &id) {
        auto it = key_map.find(id);
        if (it == key_map.end())
            return false;
        return !it->second.list_idx;
    }

    void update_stat(bsoncxx::v_noabi::builder::basic::document &doc) override {
        uint64_t feature_overhead = 0;
        uint64_t sample_overhead = 0;
        for (auto &m: in_cache_metas) {
            feature_overhead += m.feature_overhead();
            sample_overhead += m.sample_overhead();
        }
        for (auto &m: out_cache_metas) {
            feature_overhead += m.feature_overhead();
            sample_overhead += m.sample_overhead();
        }

        doc.append(kvp("n_metadata", to_string(key_map.size())));
        doc.append(kvp("feature_overhead", to_string(feature_overhead)));
        doc.append(kvp("sample ", to_string(sample_overhead)));

        int res;
        auto importances = vector<double>(ParallelRLBS::n_feature, 0);

        if (booster) {
            res = LGBM_BoosterFeatureImportance(booster,
                                                0,
                                                1,
                                                importances.data());
            if (res == -1) {
                cerr << "error: get model importance fail" << endl;
                abort();
            }
        }

        doc.append(kvp("model_importance", [importances](sub_array child) {
            for (const auto &element : importances)
                child.append(element);
        }));
    }

};




static Factory<ParallelRLBSCache> factoryParallelRLBS("ParallelRLBS");
#endif //WEBCACHESIM_RLBS_H
