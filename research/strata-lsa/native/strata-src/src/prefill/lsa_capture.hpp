// Isolated calibration export; inactive unless STRATA_LSA_CAPTURE_DIR is set.
#pragma once
#include <filesystem>
#include <limits>
namespace strata::prefill::lsa_calibration {
inline bool dump(const char* directory,int64_t layer,int64_t ordinal,int64_t p0,int64_t T,int64_t cap,
 const strata::kernels::QsaShapes& s,const strata::kernels::QsaAttnPools& p,int64_t slots,
 const float* q,const float* attn,const int32_t* ids,const int32_t* steps,cudaStream_t stream,std::string& err) {
 try {
  namespace fs=std::filesystem;
  const fs::path root="/home/jeffrey-watts/research/lsa-prep-20261103/flashnext-maps/captures";
  fs::path dir=fs::canonical(directory); auto rel=dir.lexically_relative(root);
  if(rel.empty() || rel.is_absolute() || *rel.begin()==".." || !fs::is_directory(dir))throw std::runtime_error("capture path outside owned capture root");
  if(s.n_head!=24 || s.n_head_kv!=2 || s.head_dim!=256 || s.n_rot!=64 || s.page_size<=0 || s.page_size>512 ||
     p0<0 || T<=0 || T>8192 || p0+T>16384 || cap<=0 || cap>8192 || slots<=0 ||
     !p.k_q || !p.v_q || !p.k_scale || !p.v_scale || !p.page_table || p.k_q4 || p.v_q4)
      throw std::runtime_error("unsupported calibration geometry/cache/bounds");
  auto copy=[&](void* dst,const void* src,size_t bytes){
   auto code=cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToHost,stream);
   if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
  };
  auto sync=[&](){auto code=cudaStreamSynchronize(stream);if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));};
  int64_t keys=p0+T, pages=(keys+s.page_size-1)/s.page_size;
  std::vector<int32_t> table((size_t)pages);copy(table.data(),p.page_table,table.size()*4);sync();
  int64_t maxpage=0;
  for(int32_t page:table){if(page<0 || page>=slots)throw std::runtime_error("invalid physical page");maxpage=std::max(maxpage,(int64_t)page);}
  if((maxpage+1)*s.page_size>65536)throw std::runtime_error("capture physical range too large");
  const size_t rows=(size_t)((maxpage+1)*2*s.page_size);
  std::vector<int8_t> pk(rows*256),pv(rows*256),k((size_t)keys*512),v((size_t)keys*512);
  std::vector<uint16_t> pks(rows*4),pvs(rows*4),ks((size_t)keys*8),vs((size_t)keys*8);
  std::vector<float> hq((size_t)T*6144),ha((size_t)T*6144);
  std::vector<int32_t> hi((size_t)T*(size_t)cap),hs((size_t)T*4),positions,chosen;
  copy(pk.data(),p.k_q,pk.size());copy(pv.data(),p.v_q,pv.size());
  copy(pks.data(),p.k_scale,pks.size()*2);copy(pvs.data(),p.v_scale,pvs.size()*2);
  copy(hq.data(),q,hq.size()*4);copy(ha.data(),attn,ha.size()*4);copy(hi.data(),ids,hi.size()*4);copy(hs.data(),steps,hs.size()*4);sync();
  for(int64_t cell=0;cell<keys;++cell)for(int64_t head=0;head<2;++head){
   size_t src=(size_t)((table[(size_t)(cell/s.page_size)]*2+head)*s.page_size+cell%s.page_size),dst=(size_t)(cell*2+head);
   std::memcpy(k.data()+dst*256,pk.data()+src*256,256);std::memcpy(v.data()+dst*256,pv.data()+src*256,256);
   std::memcpy(ks.data()+dst*4,pks.data()+src*4,8);std::memcpy(vs.data()+dst*4,pvs.data()+src*4,8);
  }
  for(int64_t t=0;t<T;++t)if((p0+t)%32==0 || t+1==T){
   if(hs[(size_t)t*4]!=p0+t || hs[(size_t)t*4+1]!=p0+t+1 || hs[(size_t)t*4+3]<1 || hs[(size_t)t*4+3]>cap)
    throw std::runtime_error("invalid attention step");
   chosen.push_back((int32_t)t);positions.push_back((int32_t)(p0+t));
  }
  const int64_t hdr[18]={0x4c53414d41503031LL,1,layer,ordinal,p0,T,keys,(int64_t)chosen.size(),24,2,256,64,s.page_size,cap,4,32,0,0};
  fs::path final=dir/("layer-"+std::to_string(layer)+"-p"+std::to_string(p0)+".lsacap");fs::path temp=final.string()+".partial";
  if(fs::exists(final))throw std::runtime_error("capture file already exists");
  std::FILE* f=std::fopen(temp.c_str(),"wbx");if(!f)throw std::runtime_error("cannot exclusively create capture");
  bool ok=true;auto write=[&](const void* data,size_t bytes){if(bytes && std::fwrite(data,1,bytes,f)!=bytes)ok=false;};
  write(hdr,sizeof(hdr));write(positions.data(),positions.size()*4);
  for(int32_t t:chosen)write(hq.data()+(size_t)t*6144,6144*4);
  for(int32_t t:chosen)write(ha.data()+(size_t)t*6144,6144*4);
  for(int32_t t:chosen)write(hi.data()+(size_t)t*(size_t)cap,(size_t)cap*4);
  for(int32_t t:chosen)write(hs.data()+(size_t)t*4,16);
  write(k.data(),k.size());write(v.data(),v.size());write(ks.data(),ks.size()*2);write(vs.data(),vs.size()*2);
  if(std::fflush(f)!=0)ok=false;if(std::fclose(f)!=0)ok=false;
  if(!ok)throw std::runtime_error("incomplete capture write; partial file retained");
  fs::rename(temp,final);
  std::fprintf(stderr,"LSA_CAPTURE layer=%lld p0=%lld tokens=%lld queries=%zu file=%s\n",(long long)layer,(long long)p0,(long long)T,chosen.size(),final.c_str());
  return true;
 }catch(const std::exception& e){err=std::string("LSA calibration: ")+e.what();return false;}
}
} // namespace strata::prefill::lsa_calibration
