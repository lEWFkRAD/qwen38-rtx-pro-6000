#include "strata/lsa/ward.hpp"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <climits>

namespace {
constexpr int D=64,P=128,T=512,B=64,CAP=16;
constexpr unsigned long long EMPTY=0x7fffffffffffffffULL;

// This is _square_sum in the pinned Triton Ward kernel: four-coordinate
// runs then halves. Explicit FMAs and add.rn preserve that chosen order.
__device__ float square_sum(const float* transposed, int row) {
    float runs[16];
#pragma unroll
    for(int i=0;i<16;++i) {
        const float a=transposed[(4*i)*P+row],b=transposed[(4*i+1)*P+row];
        const float c=transposed[(4*i+2)*P+row],d=transposed[(4*i+3)*P+row];
        runs[i]=__fmaf_rn(d,d,__fmaf_rn(c,c,__fmaf_rn(a,a,__fmul_rn(b,b))));
    }
#pragma unroll
    for(int half=8;half;half>>=1)
#pragma unroll
        for(int i=0;i<half;++i) runs[i]=__fadd_rn(runs[i],runs[i+half]);
    return runs[0];
}
// The pinned Triton3.7.1 Ward PTX uses sqrt.approx.ftz.f32 even for its
// IEEE dot mode. CUDA sqrtf defaults to correctly-rounded sqrt instead; at
// near-copy costs their last-bit difference changes a mutual partner. Match
// the actual pinned square-root instruction, without changing gates or merges.
__device__ float ward_sqrt(float value) {
    float result;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
}
__device__ unsigned long long cost_key(float cost,int partner) {
    int ordered=__float_as_int(cost);
    if(ordered<0) ordered^=0x7fffffff;
    const unsigned long long positive=(unsigned long long)((long long)ordered+(1LL<<31));
    return (positive<<12)|(unsigned)partner;
}
__device__ void fail(int32_t* scratch,int error) { atomicCAS(scratch,0,error); }
__device__ float warp_sum(float x) {
#pragma unroll
    for(int d=16;d;d>>=1) x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));
    return x;
}

template<bool Step> __global__ void close_kernel(const float* history,int capacity,const int32_t* position,
 int first,float* centroids,int32_t* members,int32_t* diagnostics,int32_t* out_labels,float* objective) {
    const int tid=threadIdx.x;
    int block=first+(int)blockIdx.x;
    if constexpr(Step) {
        const int pos=*position;
        if(pos<0 || pos>=capacity) {if(tid==0) fail(diagnostics,-1);return;}
        if((pos&511)!=511) return;
        block=pos/512;
    }
    // Host checks assure range calls refer only to complete in-capacity blocks.
    const int start=block*T;
    if(start<0 || start>capacity-T) {if(tid==0) fail(diagnostics,-1);return;}
    __shared__ float work[D*P];       // transposed: adjacent threads own adjacent pools
    __shared__ float square[P];
    __shared__ int sizes[P],alive[P],parent[P],best[P],labels[P];
    __shared__ unsigned long long nearest[P],mutual[P];
    __shared__ int count,rounds,applied,mutual_count,error;
    if(tid==0) {count=P;rounds=0;applied=1;error=0;}
    if(tid<P) {sizes[tid]=4;alive[tid]=1;parent[tid]=tid;}
    for(int i=tid;i<D*P;i+=blockDim.x) {
        const int d=i/P,p=i%P;
        const float* x=history+(std::size_t)(start+p*4)*D+d;
        // PyTorch's reduction of each consecutive4-key pool adds in input order.
        float value=__fadd_rn(__fadd_rn(__fadd_rn(x[0],x[D]),x[2*D]),x[3*D]);
        if(!isfinite(value)) atomicCAS(&error,0,-2);
        work[i]=value;
    }
    __syncthreads();
    if(tid<P) {square[tid]=square_sum(work,tid);if(!isfinite(square[tid])) atomicCAS(&error,0,-2);}
    __syncthreads();
    while(count>B && applied>0 && rounds<P && error==0) {
        if(tid<P) {
            unsigned long long key=EMPTY;
            if(alive[tid]) {
                const float ar=ward_sqrt(fmaxf(square[tid],0.0f));
                for(int col=0;col<P;++col) {
                    if(col==tid || !alive[col] || sizes[tid]+sizes[col]>CAP) continue;
                    float dot=0;
#pragma unroll
                    for(int d=0;d<D;++d) dot=__fmaf_rn(work[d*P+tid],work[d*P+col],dot);
                    const float merged=ward_sqrt(fmaxf(__fadd_rn(__fadd_rn(square[tid],square[col]),__fmul_rn(2.0f,dot)),0.0f));
                    const float cost=__fsub_rn(__fadd_rn(ar,ward_sqrt(fmaxf(square[col],0.0f))),merged);
                    if(!isfinite(cost)) {atomicCAS(&error,0,-2);continue;}
                    const auto candidate=cost_key(cost,col);
                    if(candidate<key) key=candidate;
                }
            }
            nearest[tid]=key;
            best[tid]=key<EMPTY?(int)(key&4095ULL):-1;
        }
        __syncthreads();
        if(tid<P) {
            const int partner=best[tid];
            mutual[tid]=partner>=0 && tid<partner && best[partner]==tid ? nearest[tid]:EMPTY;
        }
        __syncthreads();
        if(tid==0) {
            mutual_count=0;
            for(int row=0;row<P;++row) mutual_count+=(mutual[row]<EMPTY);
            applied=min(mutual_count,count-B);
        }
        __syncthreads();
        if(tid<P && mutual[tid]<EMPTY && error==0) {
            bool keep=true;
            if(mutual_count>count-B) {
                int rank=0;
                for(int row=0;row<P;++row) rank+=(mutual[row]<mutual[tid]);
                keep=rank<count-B;
            }
            if(keep) {
                const int partner=best[tid];
#pragma unroll
                for(int d=0;d<D;++d) work[d*P+tid]=__fadd_rn(work[d*P+tid],work[d*P+partner]);
                square[tid]=square_sum(work,tid);
                sizes[tid]+=sizes[partner];alive[partner]=0;parent[partner]=tid;
            }
        }
        __syncthreads();
        if(tid==0) {count-=applied;++rounds;}
        __syncthreads();
    }
    if(tid==0 && error==0 && count!=B) error=-3;
    __syncthreads();
    if(tid<P) {
        int root=tid;
        for(int jumps=0;jumps<P && parent[root]!=root;++jumps) root=parent[root];
        int label=0;
        for(int row=0;row<root;++row) label+=alive[row];
        labels[tid]=label;
        if(parent[root]!=root || label<0 || label>=B) atomicCAS(&error,0,-4);
        if(out_labels) out_labels[(std::size_t)block*P+tid]=label;
    }
    __syncthreads();
    int32_t* output_members=members+(std::size_t)block*B*CAP;
    float* output_centroids=centroids+(std::size_t)block*B*D;
    if(tid<B) {
        for(int slot=0;slot<CAP;++slot) output_members[tid*CAP+slot]=-1;
        int held=0;
        for(int pool=0;pool<P;++pool) if(labels[pool]==tid) {
            for(int k=0;k<4;++k) {
                if(held<CAP) output_members[tid*CAP+held]=start+pool*4+k;
                ++held;
            }
        }
        if(held<4 || held>CAP || held%4) atomicCAS(&error,0,-4);
    }
    __syncthreads();
    // Build exported centroids from original key rows in member order rather
    // than the rounded Ward merge tree. This is the upstream centroid definition.
    for(int i=tid;i<B*D;i+=blockDim.x) {
        const int bucket=i/D,d=i%D;
        float sum=0;
        if(error==0)
            for(int slot=0;slot<CAP;++slot) {
                const int id=output_members[bucket*CAP+slot];
                if(id>=0) sum=__fadd_rn(sum,history[(std::size_t)id*D+d]);
            }
        output_centroids[i]=sum;
    }
    __syncthreads();
    const int lane=tid&31,warp=tid>>5;
    for(int bucket=warp;bucket<B;bucket+=blockDim.x/32) {
        const float a=output_centroids[bucket*D+lane],b=output_centroids[bucket*D+lane+32];
        const float norm=sqrtf(warp_sum(__fadd_rn(__fmul_rn(a,a),__fmul_rn(b,b))));
        const float den=fmaxf(norm,1e-12f);
        output_centroids[bucket*D+lane]=__bfloat162float(__float2bfloat16_rn(a/den));
        output_centroids[bucket*D+lane+32]=__bfloat162float(__float2bfloat16_rn(b/den));
        if(!isfinite(norm)) atomicCAS(&error,0,-2);
    }
    __syncthreads();
    if(tid==0) {
        if(objective) {
            float alignment=0;
            for(int pool=0;pool<P;++pool) if(alive[pool]) alignment+=sqrtf(fmaxf(square[pool],0.0f));
            objective[block]=alignment;
        }
        if(error) fail(diagnostics,error);
        else {atomicAdd(diagnostics+1,1);atomicMax(diagnostics+2,rounds);}
    }
    if(error) {
        for(int i=tid;i<B*CAP;i+=blockDim.x) output_members[i]=-1;
        for(int i=tid;i<B*D;i+=blockDim.x) output_centroids[i]=0;
    }
}
bool valid(const float* history,int capacity,float* centroids,int32_t* members,void* scratch,void* stream) {
    return history && centroids && members && scratch && stream && capacity>0 && capacity<=131072;
}
int launch_range(const float* h,int c,int first,int n,float* out,int32_t* m,void* scratch,
                 int32_t* labels,float* objective,void* stream) {
    if(!valid(h,c,out,m,scratch,stream) || first<0 || n<=0 || first>c/512-n) return cudaErrorInvalidValue;
    close_kernel<false><<<n,256,0,(cudaStream_t)stream>>>(h,c,nullptr,first,out,m,(int32_t*)scratch,labels,objective);
    return cudaGetLastError();
}
}
namespace strata::lsa {
std::size_t ward_scratch_bytes(){return 4*sizeof(int32_t);}
int ward_close_step(const float* h,const int32_t* p,int c,float* out,int32_t* m,void* scratch,void* stream) {
    if(!valid(h,c,out,m,scratch,stream) || !p) return cudaErrorInvalidValue;
    close_kernel<true><<<1,256,0,(cudaStream_t)stream>>>(h,c,p,0,out,m,(int32_t*)scratch,nullptr,nullptr);
    return cudaGetLastError();
}
int ward_close_range(const float* h,int c,int first,int n,float* out,int32_t* m,void* scratch,void* stream) {
    return launch_range(h,c,first,n,out,m,scratch,nullptr,nullptr,stream);
}
}
extern "C" std::size_t lsa_ward_scratch_bytes(){return strata::lsa::ward_scratch_bytes();}
extern "C" int lsa_ward_close_step(const float* h,const int32_t* p,int c,float* out,int32_t* m,void* s,void* st){return strata::lsa::ward_close_step(h,p,c,out,m,s,st);}
extern "C" int lsa_ward_close_range(const float* h,int c,int first,int n,float* out,int32_t* m,void* s,void* st){return strata::lsa::ward_close_range(h,c,first,n,out,m,s,st);}
extern "C" int lsa_ward_close_range_diagnostic(const float* h,int c,int first,int n,float* out,int32_t* m,void* s,int32_t* l,float* o,void* st){
    if(!l || !o) return cudaErrorInvalidValue;
    return launch_range(h,c,first,n,out,m,s,l,o,st);
}
