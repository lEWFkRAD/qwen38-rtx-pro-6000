#include "strata/lsa/cuda_adapter.h"
#include "strata/lsa/maps.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cmath>
#include <climits>
#include <cstdio>
#include <memory>
#include <string>

namespace {
constexpr int kDim=64, kInput=384, kTopP=385, kTopK=1024, kWindow=512, kSink=4;
constexpr int kWidth=kTopK+kWindow+kSink;
struct MapHandle {
    float* data=nullptr;
    float temperature[12]={};
    uint64_t identity=0;
    ~MapHandle(){if(data) cudaFree(data);}
};
int ordinal(int layer){return layer>=3 && layer<=47 && (layer-3)%4==0?(layer-3)/4:-1;}

__device__ float sum_warp(float value) {
    for(int delta=16;delta;delta>>=1) value+=__shfl_xor_sync(0xffffffff,value,delta);
    return value;
}
template<bool Query> __global__ void project(const int8_t* codes,const uint16_t* scales,
 const int32_t* table,int physical_pages,int page_size,int start,const int32_t* positions,const float* q,
 const float* map,float* out) {
    const int row=blockIdx.x,d=threadIdx.x;
    __shared__ float inputs[kInput];
    __shared__ float norms[2];
    // Gather each of the 384 input features once. Its original head-sum or
    // dequantization arithmetic is unchanged; each output keeps its serial
    // 384-term multiplication/addition order after this shared staging barrier.
    for(int f=d;f<kInput;f+=kDim) {
        const int head=f/192,coordinate=64+f%192;
        float x=0;
        if constexpr(Query) {
            for(int group=0;group<12;++group)
                x+=q[((size_t)row*24+head*12+group)*256+coordinate];
        } else {
            const int position=positions?positions[row]:start+row;
            const int page=position>=0 && position<131072 && position/page_size<physical_pages
                ?table[position/page_size]:-1;
            if(page>=0 && page<physical_pages) {
                const size_t r=((size_t)page*2+head)*page_size+position%page_size;
                x=(float)codes[r*256+coordinate]*__half2float(__ushort_as_half(scales[r*4+coordinate/64]));
            }
        }
        inputs[f]=x;
    }
    __syncthreads();
    float acc=0;
    for(int f=0;f<kInput;++f)
        acc=__fadd_rn(acc,__fmul_rn(inputs[f],map[f*kDim+d]));
    const float n=sum_warp(acc*acc);
    if((d&31)==0) norms[d/32]=n;
    __syncthreads();
    const float denominator=fmaxf(sqrtf(norms[0]+norms[1]),1e-12f);
    float value=acc/denominator;
    if constexpr(Query) value=__bfloat162float(__float2bfloat16_rn(value));
    out[(size_t)row*kDim+d]=value;
}

__global__ void score(const float* query,const float* centroids,const int32_t* positions,
 int buckets,int64_t* keys) {
    const int bucket=blockIdx.x*8+threadIdx.x/32,row=blockIdx.y,lane=threadIdx.x%32;
    if(bucket>=buckets) return;
    const int pos=positions[row],visible=pos>=0 && pos<131072 ? min((pos/512)*64,buckets):0;
    float dot=0;
    if(bucket<visible) {
        dot=query[(size_t)row*64+lane]*centroids[(size_t)bucket*64+lane];
        dot+=query[(size_t)row*64+lane+32]*centroids[(size_t)bucket*64+lane+32];
    }
    dot=sum_warp(dot);
    if(lane==0) {
        int32_t ordered=__float_as_int(dot);
        if(ordered<0) ordered^=0x7fffffff;
        const uint64_t packed=((uint64_t)(uint32_t)ordered<<32)|(uint32_t)(0x7fffffff-bucket);
        keys[(size_t)row*buckets+bucket]=bucket<visible && isfinite(dot)?(int64_t)packed:LLONG_MIN;
    }
}
__global__ void fill_offsets(int32_t* offsets,const int32_t* positions,int queries,int buckets) {
    if(queries==1) {
        // Captured decode keeps capacity-sized storage and launch bounds, but
        // only its causal prefix participates in the segmented sort. The end
        // offset is read from device position on every graph replay, including
        // zero-visible prefixes. CUB accepts an empty segment without reading it.
        if(threadIdx.x==0) offsets[0]=0;
        if(threadIdx.x==1) {
            const int pos=positions[0];
            offsets[1]=pos>=0 && pos<131072?min((pos/512)*64,buckets):0;
        }
    } else if(threadIdx.x<=queries) offsets[threadIdx.x]=threadIdx.x*buckets;
}

// A CTA walks 256-member tiles in score-ranked bucket/member order. Exclusive
// integer ranks preserve the exact serial expansion while independent loads,
// initialization and local-window writes run in parallel. No score math changes.
__global__ void expand(const int64_t* sorted,const int32_t* members,const int32_t* positions,
 int buckets,int32_t* selected,int32_t* cells,int32_t* steps) {
    constexpr int Threads=256,MembersPerBucket=16,BucketsPerTile=Threads/MembersPerBucket;
    using Scan=cub::BlockScan<int,Threads>;
    __shared__ Scan::TempStorage scan;
    __shared__ int filled;
    const int row=blockIdx.x,pos=positions[row],tid=threadIdx.x;
    const bool valid_position=pos>=0 && pos<131072;
    const int visible=valid_position?min((pos/512)*64,buckets):0;
    int32_t* out=cells+(size_t)row*kWidth;
    int32_t* picks=selected+(size_t)row*kTopP;
    for(int j=tid;j<kWidth;j+=Threads) out[j]=-1;
    // Publish every selected bucket even when expansion reaches 1024 keys early.
    for(int j=tid;j<kTopP;j+=Threads) {
        int bucket=-1;
        if(j<min(kTopP,visible)) {
            const int64_t key=sorted[(size_t)row*buckets+j];
            if(key!=LLONG_MIN) {
                const int decoded=0x7fffffff-(int)(key&0xffffffff);
                if(decoded>=0 && decoded<visible) bucket=decoded;
            }
        }
        picks[j]=bucket;
    }
    if(tid==0) filled=0;
    __syncthreads();
    int window_count=0,sink_count=0;
    if(valid_position) {
        const int window_start=max(0,pos+1-kWindow);
        for(int first=0;first<min(kTopP,visible) && filled<kTopK;first+=BucketsPerTile) {
            const int pick=first+tid/MembersPerBucket;
            const int bucket=pick<kTopP?picks[pick]:-1;
            const int id=bucket>=0?members[(size_t)bucket*MembersPerBucket+tid%MembersPerBucket]:-1;
            const int keep=id>=kSink && id<window_start;
            int prefix=0,total=0;
            Scan(scan).ExclusiveSum(keep,prefix,total);
            const int at=filled+prefix;
            if(keep && at<kTopK) out[at]=id;
            // Reusing scan storage and advancing the shared base both wait for
            // all threads, including the tile that crosses the 1024-key budget.
            __syncthreads();
            if(tid==0) filled=min(kTopK,filled+total);
            __syncthreads();
        }
        window_count=pos+1-window_start;
        sink_count=min(kSink,window_start);
        for(int j=tid;j<window_count;j+=Threads) out[filled+j]=window_start+j;
        if(tid<sink_count) out[filled+window_count+tid]=tid;
    }
    if(tid==0) {
        steps[row*4]=pos;
        steps[row*4+1]=valid_position?pos+1:0;
        steps[row*4+2]=valid_position?(pos+1)/4:0;
        steps[row*4+3]=filled+window_count+sink_count;
    }
}
bool shape(int queries,int buckets) {
    return queries>0 && queries<=128 && buckets>0 && buckets<=16384 && buckets%64==0;
}
}
extern "C" void* lsa_maps_open(const char* path,char* error,size_t capacity) {
    auto fail=[&](const std::string& message)->void* {
        if(error && capacity) std::snprintf(error,capacity,"%s",message.c_str());
        return nullptr;
    };
    if(error && capacity) error[0]=0;
    if(!path) return fail("missing LSA map path");
    try {
        strata::lsa::Maps maps;std::string message;
        if(!strata::lsa::load_maps(path,maps,message)) return fail(message);
        auto handle=std::make_unique<MapHandle>();
        handle->identity=maps.artifact_fingerprint;
        auto code=cudaMalloc((void**)&handle->data,24*384*64*sizeof(float));
        if(code!=cudaSuccess) return fail(cudaGetErrorString(code));
        for(int i=0;i<12;++i) {
            handle->temperature[i]=maps.layers[i].temperature;
            for(int key=0;key<2;++key) {
                const auto& matrix=key?maps.layers[i].key:maps.layers[i].query;
                code=cudaMemcpy(handle->data+(i*2+key)*384*64,matrix.data(),384*64*sizeof(float),cudaMemcpyHostToDevice);
                if(code!=cudaSuccess) return fail(cudaGetErrorString(code));
            }
        }
        // Pageable H2D copies may return after staging. The first consumer can
        // use a nonblocking stream, so publish only fully uploaded weights.
        code=cudaStreamSynchronize(nullptr);
        if(code!=cudaSuccess) return fail(cudaGetErrorString(code));
        return handle.release();
    } catch(const std::exception& ex){return fail(ex.what());}
}
extern "C" const float* lsa_maps_query(void* pointer,int layer) {
    auto* h=(MapHandle*)pointer;const int i=ordinal(layer);
    return h && i>=0?h->data+i*2*384*64:nullptr;
}
extern "C" const float* lsa_maps_key(void* pointer,int layer) {
    auto* h=(MapHandle*)pointer;const int i=ordinal(layer);
    return h && i>=0?h->data+(i*2+1)*384*64:nullptr;
}
extern "C" float lsa_maps_temperature(void* pointer,int layer) {
    auto* h=(MapHandle*)pointer;const int i=ordinal(layer);
    return h && i>=0?h->temperature[i]:NAN;
}
extern "C" uint64_t lsa_maps_identity(void* pointer) {
    const auto* h=(const MapHandle*)pointer;
    return h?h->identity:0;
}
extern "C" void lsa_maps_close(void* pointer){delete (MapHandle*)pointer;}
extern "C" int lsa_project_keys(const int8_t* codes,const uint16_t* scales,const int32_t* table,
 int physical_pages,int page_size,int start,int count,const float* map,float* output,void* stream) {
    if(!codes||!scales||!table||!map||!output||!stream||physical_pages<=0||physical_pages>131072||
       page_size<=0||page_size>512||start<0||count<=0||start>131072-count||
       (int64_t)start+count>(int64_t)physical_pages*page_size)
        return cudaErrorInvalidValue;
    project<false><<<count,64,0,(cudaStream_t)stream>>>(codes,scales,table,physical_pages,page_size,start,nullptr,nullptr,map,output);
    return cudaGetLastError();
}
extern "C" int lsa_project_key_steps(const int8_t* codes,const uint16_t* scales,const int32_t* table,
 int physical_pages,int page_size,const int32_t* positions,int count,const float* map,float* output,void* stream) {
    if(!codes||!scales||!table||!map||!output||!stream||!positions||physical_pages<=0||physical_pages>131072||
       page_size<=0||page_size>512||count<=0||count>128) return cudaErrorInvalidValue;
    project<false><<<count,64,0,(cudaStream_t)stream>>>(codes,scales,table,physical_pages,page_size,0,positions,nullptr,map,output);
    return cudaGetLastError();
}
extern "C" int lsa_project_queries(const float* query,int count,const float* map,float* output,void* stream) {
    if(!query||!map||!output||!stream||count<=0||count>128) return cudaErrorInvalidValue;
    project<true><<<count,64,0,(cudaStream_t)stream>>>(nullptr,nullptr,nullptr,0,0,0,nullptr,query,map,output);
    return cudaGetLastError();
}
extern "C" int lsa_select_temp_bytes(int queries,int buckets,size_t* bytes) {
    if(!bytes||!shape(queries,buckets)) return cudaErrorInvalidValue;
    *bytes=0;
    return cub::DeviceSegmentedRadixSort::SortKeysDescending(nullptr,*bytes,
      (int64_t*)nullptr,(int64_t*)nullptr,queries*buckets,queries,
      (int32_t*)nullptr,(int32_t*)nullptr,0,64);
}
extern "C" int lsa_select(const float* query,const float* centroids,const int32_t* members,
 const int32_t* positions,int queries,int buckets,int64_t* keys_in,int64_t* keys_out,
 int32_t* offsets,void* temporary,size_t bytes,
 int32_t* selected,int32_t* cells,int32_t* steps,void* stream) {
    if(!shape(queries,buckets)||!query||!centroids||!members||!positions||!keys_in||!keys_out||
       !offsets||!temporary||!selected||!cells||!steps||!stream)
        return cudaErrorInvalidValue;
    size_t needed=0;
    auto error=(cudaError_t)lsa_select_temp_bytes(queries,buckets,&needed);
    if(error!=cudaSuccess) return error;
    if(bytes<needed) return cudaErrorInvalidValue;
    auto st=(cudaStream_t)stream;
    score<<<dim3((buckets+7)/8,queries),256,0,st>>>(query,centroids,positions,buckets,keys_in);
    fill_offsets<<<1,256,0,st>>>(offsets,positions,queries,buckets);
    error=cudaGetLastError();
    if(error!=cudaSuccess) return error;
    error=cub::DeviceSegmentedRadixSort::SortKeysDescending(temporary,bytes,keys_in,keys_out,
      queries*buckets,queries,offsets,offsets+1,0,64,st);
    if(error!=cudaSuccess) return error;
    expand<<<queries,256,0,st>>>(keys_out,members,positions,buckets,selected,cells,steps);
    return cudaGetLastError();
}
extern "C" size_t lsa_attention_scratch_floats(void) {
    return strata::kernels::qsa_decode_attn_scratch_floats(kWidth,strata::kernels::qsa_real_shapes());
}
extern "C" int lsa_attention(const float* query,const int8_t* k,const int8_t* v,
 const uint16_t* ks,const uint16_t* vs,const int32_t* table,int page_size,int queries,
 const int32_t* cells,const int32_t* steps,float* scratch,float* output,void* stream) {
    if(!query||!k||!v||!ks||!vs||!table||!cells||!steps||!scratch||!output||!stream||
       page_size<=0||page_size>512||queries<=0||queries>128) return cudaErrorInvalidValue;
    auto s=strata::kernels::qsa_real_shapes();s.page_size=page_size;
    strata::kernels::QsaAttnPools p;p.k_q=k;p.v_q=v;p.k_scale=ks;p.v_scale=vs;p.page_table=table;
    strata::kernels::qsa_decode_attn_batch(query,p,cells,steps,kWidth,s,scratch,output,queries,stream);
    return cudaGetLastError();
}
