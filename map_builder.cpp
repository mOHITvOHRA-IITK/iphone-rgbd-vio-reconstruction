// ============================================================
// map_builder.cpp
//
// 3D reconstruction (TSDF mesh) of the recorded scene from the vio_slam results.
//
// Input (one phone recording + its vio_slam output):
//   data.mov, frames2/*.png, data.jsonl, calibration.json
//   slam_results_vio/trajectory_all_frames.txt   optimized pose of every frame
//   slam_results_vio/frame_quality.txt           per-frame dense pose check
//
// Steps
//   1. DEPTH CLEAN-UP per frame: depth edges (flying pixels) and depth under
//      blown-out colour are dropped, then an edge-preserving filter removes
//      noise without blurring object boundaries. Every pixel gets a quality
//      weight from its distance (phone depth noise grows ~z^2) and viewing
//      angle (grazing views are noisy).
//   2. TSDF FUSION of every frame whose pose PASSED the dense check
//      (6 mm voxels, 8^3 voxel blocks, truncation 5 voxels), weighted by
//      that pixel quality. Colour comes from voxels near the surface, and
//      blurred frames (fast motion) count less. Geometry comes only from
//      measured depth.
//   3. MARCHING CUBES on the sparse blocks; a triangle is created only
//      where all 8 cube corners were observed >= MIN_OBSERVATIONS times
//      with enough summed quality (MIN_SURFACE_WEIGHT).
//      Vertices are shared through their voxel edge, so the mesh is
//      connected without seams.
//   4. CLEAN-UP: disconnected pieces smaller than MIN_COMPONENT_TRIANGLES
//      (flying pixels, specks) are removed, then Taubin smoothing (no
//      shrinkage) removes the remaining grain; vertex normals computed.
//   5. LIGHTER MESH by vertex clustering: a coloured mesh for viewing.
//   6. FLOOR PLANE: lowest strong peak of upward-facing surface, refined
//      by RANSAC + least squares (the VIO gravity direction can be off by
//      a fraction of a degree); used to level the top view and to report
//      the floor flatness.
//   7. TOP VIEW + REPORT.
//
// Output (<recording>/reconstruction/):
//   room_mesh.ply              full-detail coloured mesh
//   room_mesh_light.ply        coloured mesh with ~1 cm vertex spacing (viewing, scene_player --mesh)
//   top_view.png               coloured top-down view
//   reconstruction_report.txt
//
// World frame = vio_slam frame (metres, z up / gravity -z,
// p_world = R * p_camera + t); nothing is re-centred, so the camera
// trajectory and the model share one frame.
//
// Usage: map_builder [recording_dir] [--voxel M] [--every N] [--all-frames]
// ============================================================

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <vtkMarchingCubesTriangleCases.h>

#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

// ============================================================
// SETTINGS
// ============================================================
static const std::string DATASET_DIR =
    "/home/mohit/3D/recording_2026-10-04_11-24-56";

static constexpr int DEPTH_W = 256, DEPTH_H = 192;
static constexpr int COLOR_W = 640, COLOR_H = 480;     // colour used for the mesh
static constexpr float DEPTH_SCALE = 0.001f;
static constexpr float MIN_DEPTH = 0.15f, MAX_DEPTH = 3.0f;

static constexpr double DEFAULT_VOXEL = 0.006;         // m
static constexpr int TRUNC_VOXELS = 5;                 // truncation = 5 voxels
static constexpr int MIN_OBSERVATIONS = 3;             // frames that must see a surface
static constexpr int MIN_COMPONENT_TRIANGLES = 2000;   // smaller disconnected pieces are noise

// Depth quality. Phone depth noise sigma(z) = A + B (z - 0.4)^2 (relative weights only).
static constexpr float DEPTH_NOISE_A = 0.0012f, DEPTH_NOISE_B = 0.0019f;
static constexpr int DEPTH_FILTER_RADIUS = 2;            // edge-preserving filter window 5x5
static constexpr float DEPTH_FILTER_SIGMA_PX = 1.5f;
static constexpr float DEPTH_EDGE_JUMP = 0.05f;          // relative depth jump = edge / flying pixel
static constexpr int COLOR_SATURATED = 250;              // depth under blown-out colour is dropped
static constexpr float MIN_INCIDENCE_COS = 0.15f;        // weight floor for grazing views
static constexpr float MIN_SURFACE_WEIGHT = 1.0f;        // summed quality needed to keep a surface
static constexpr int BLUR_WINDOW = 60;                   // frames for the running sharpness median
static constexpr int TAUBIN_ITERATIONS = 10;
static constexpr float TAUBIN_LAMBDA = 0.5f, TAUBIN_MU = -0.53f;

static constexpr double LIGHT_CELL = 0.010;            // vertex spacing of room_mesh_light.ply

static constexpr double TOP_VIEW_PX = 0.01;            // m per pixel
static constexpr double TOP_VIEW_MAX_HEIGHT = 2.2;     // cut the ceiling off the top view

// Recording folder from the command line: a path (absolute or relative to
// the current directory) or just the folder name, looked up next to DATASET_DIR.
static std::string resolveRecording(const std::string& arg){
    std::error_code ec;
    for(const fs::path& p:{fs::path(arg),fs::path(DATASET_DIR).parent_path()/arg})
        if(fs::exists(p/"data.mov",ec)) return fs::canonical(p,ec).string();
    return "";
}

// ============================================================
// BASIC TYPES
// ============================================================
struct Pose {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Pose inverse() const { Pose o; o.R=R.transpose(); o.t=-o.R*t; return o; }
};

struct Intrinsics { float fx=0, fy=0, cx=0, cy=0; bool valid=false; };

struct Mesh {
    std::vector<Eigen::Vector3f> v, n;
    std::vector<std::array<uint8_t,3>> rgb;       // empty = no colour
    std::vector<std::array<int,3>> f;
};

template<typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t n):max_(n){}
    void push(T x){ std::unique_lock<std::mutex> l(m_); full_.wait(l,[&]{return q_.size()<max_;}); q_.push_back(std::move(x)); empty_.notify_one(); }
    bool pop(T& x){ std::unique_lock<std::mutex> l(m_); empty_.wait(l,[&]{return !q_.empty()||done_;}); if(q_.empty()) return false; x=std::move(q_.front()); q_.pop_front(); full_.notify_one(); return true; }
    void finish(){ std::lock_guard<std::mutex> l(m_); done_=true; empty_.notify_all(); }
private:
    std::deque<T> q_; size_t max_; std::mutex m_; std::condition_variable empty_,full_; bool done_=false;
};

// ============================================================
// INPUT
// ============================================================
static size_t matchBracket(const std::string& s,size_t open){
    if(open>=s.size()) return std::string::npos;
    const char o=s[open], c=(o=='[')?']':'}';
    int depth=0;
    for(size_t i=open;i<s.size();++i){
        if(s[i]==o) ++depth;
        else if(s[i]==c && --depth==0) return i;
    }
    return std::string::npos;
}

static bool jsonNumber(const std::string& s,const std::string& key,double& out){
    const std::string k="\""+key+"\":";
    size_t p=s.find(k);
    if(p==std::string::npos) return false;
    const char* b=s.c_str()+p+k.size(); char* e=nullptr;
    out=std::strtod(b,&e);
    return e!=b;
}

// Per-frame depth-resolution intrinsics from data.jsonl (depth camera preferred).
static std::vector<Intrinsics> loadIntrinsics(const std::string& rec){
    double imageW=1920.0, imageH=1440.0;
    {
        std::ifstream f(rec+"/calibration.json");
        if(f.is_open()){
            std::stringstream ss; ss<<f.rdbuf(); std::string s=ss.str(); double v;
            if(jsonNumber(s,"imageWidth",v)&&v>0) imageW=v;
            if(jsonNumber(s,"imageHeight",v)&&v>0) imageH=v;
        }
    }
    const double sx=DEPTH_W/imageW, sy=DEPTH_H/imageH;
    std::vector<Intrinsics> ks;
    std::ifstream f(rec+"/data.jsonl");
    std::string line;
    while(std::getline(f,line)){
        size_t fk=line.find("\"frames\":");
        if(fk==std::string::npos) continue;
        size_t a=line.find('[',fk), b=matchBracket(line,a);
        if(b==std::string::npos) continue;
        const std::string arr=line.substr(a,b-a+1);
        const std::string outer=line.substr(0,fk)+line.substr(b+1);
        double num=-1;
        if(!jsonNumber(outer,"number",num)||num<0) continue;
        Intrinsics best; size_t p=1;
        while(true){
            size_t o=arr.find('{',p); if(o==std::string::npos) break;
            size_t c=matchBracket(arr,o); if(c==std::string::npos) break;
            const std::string obj=arr.substr(o,c-o+1); p=c+1;
            double camInd=0; jsonNumber(obj,"cameraInd",camInd);
            size_t ck=obj.find("\"calibration\":"); if(ck==std::string::npos) continue;
            size_t co=obj.find('{',ck), cc=matchBracket(obj,co); if(cc==std::string::npos) continue;
            const std::string cal=obj.substr(co,cc-co+1);
            double fx,fy,cx,cy;
            if(!jsonNumber(cal,"focalLengthX",fx)||!jsonNumber(cal,"focalLengthY",fy)||
               !jsonNumber(cal,"principalPointX",cx)||!jsonNumber(cal,"principalPointY",cy)) continue;
            if(!best.valid||camInd==1){ best={(float)(fx*sx),(float)(fy*sy),(float)(cx*sx),(float)(cy*sy),true}; }
        }
        if((int)num>=(int)ks.size()) ks.resize((int)num+1);
        ks[(int)num]=best;
    }
    return ks;
}

// trajectory_all_frames.txt: frame_id timestamp tx ty tz qx qy qz qw status ...
static void loadPoses(const std::string& path,std::vector<Pose>& pose,std::vector<char>& tracked){
    std::ifstream in(path);
    std::string line;
    while(std::getline(in,line)){
        if(line.empty()||line[0]=='#') continue;
        std::istringstream ss(line); int id,st=1; double ts,tx,ty,tz,qx,qy,qz,qw;
        if(!(ss>>id>>ts>>tx>>ty>>tz>>qx>>qy>>qz>>qw)||id<0) continue;
        if(!(ss>>st)) st=1;
        if(id>=(int)pose.size()){ pose.resize(id+1); tracked.resize(id+1,0); }
        pose[id].R=Eigen::Quaterniond(qw,qx,qy,qz).normalized().toRotationMatrix();
        pose[id].t=Eigen::Vector3d(tx,ty,tz);
        tracked[id]=st!=0;
    }
}

// frame_quality.txt: frame ... result(PASS / CHECK / LOW_OVERLAP)
static std::unordered_map<int,std::string> loadQuality(const std::string& path){
    std::unordered_map<int,std::string> q;
    std::ifstream in(path);
    std::string line;
    while(std::getline(in,line)){
        if(line.empty()||line[0]=='#') continue;
        std::istringstream ss(line); int id; ss>>id;
        q[id]=line.substr(line.find_last_of(' ')+1);
    }
    return q;
}

static std::vector<std::string> getDepthFiles(const std::string& rec){
    std::vector<std::string> files;
    if(!fs::exists(rec+"/frames2")) return files;
    for(const auto& e:fs::directory_iterator(rec+"/frames2")){
        if(!e.is_regular_file()) continue;
        std::string ext=e.path().extension().string();
        std::transform(ext.begin(),ext.end(),ext.begin(),::tolower);
        if(ext==".png") files.push_back(e.path().string());
    }
    std::sort(files.begin(),files.end());
    return files;
}

// ============================================================
// DEPTH CLEAN-UP + PER-PIXEL QUALITY WEIGHT
// ============================================================
struct DepthFrame { cv::Mat z, w; };          // filtered depth (m, 0 = invalid), quality weight

static float depthSigma(float z){ const float d=z-0.4f; return DEPTH_NOISE_A+DEPTH_NOISE_B*d*d; }

static DepthFrame prepareDepth(const cv::Mat& depth16,const cv::Mat& bgr,const Intrinsics& K){
    const int W=depth16.cols, H=depth16.rows;
    const float sx=(float)bgr.cols/W, sy=(float)bgr.rows/H;
    // 1) Valid range, no depth under blown-out colour.
    cv::Mat raw(H,W,CV_32F,cv::Scalar(0));
    #pragma omp parallel for
    for(int v=0;v<H;++v) for(int u=0;u<W;++u){
        const float z=depth16.at<uint16_t>(v,u)*DEPTH_SCALE;
        if(z<MIN_DEPTH||z>MAX_DEPTH) continue;
        const cv::Vec3b c=bgr.at<cv::Vec3b>(std::min(bgr.rows-1,(int)((v+0.5f)*sy)),std::min(bgr.cols-1,(int)((u+0.5f)*sx)));
        if(c[0]>=COLOR_SATURATED&&c[1]>=COLOR_SATURATED&&c[2]>=COLOR_SATURATED) continue;
        raw.at<float>(v,u)=z;
    }
    // 2) Drop depth edges (flying pixels) and the rim of holes.
    cv::Mat clean(H,W,CV_32F,cv::Scalar(0));
    #pragma omp parallel for
    for(int v=1;v<H-1;++v) for(int u=1;u<W-1;++u){
        const float z=raw.at<float>(v,u);
        if(z<=0) continue;
        bool ok=true;
        for(int dy=-1;dy<=1&&ok;++dy) for(int dx=-1;dx<=1;++dx){
            const float zn=raw.at<float>(v+dy,u+dx);
            if(zn<=0||std::abs(zn-z)>DEPTH_EDGE_JUMP*z){ ok=false; break; }
        }
        if(ok) clean.at<float>(v,u)=z;
    }
    // 3) Edge-preserving smoothing (bilateral in depth, scaled to the noise at that depth).
    DepthFrame f; f.z=cv::Mat(H,W,CV_32F,cv::Scalar(0)); f.w=cv::Mat(H,W,CV_32F,cv::Scalar(0));
    const int r=DEPTH_FILTER_RADIUS; const float sp2=2*DEPTH_FILTER_SIGMA_PX*DEPTH_FILTER_SIGMA_PX;
    #pragma omp parallel for
    for(int v=0;v<H;++v) for(int u=0;u<W;++u){
        const float zc=clean.at<float>(v,u);
        if(zc<=0) continue;
        const float s=depthSigma(zc), sr2=2*s*s;
        float sum=0, ws=0;
        for(int dy=-r;dy<=r;++dy) for(int dx=-r;dx<=r;++dx){
            const int y=v+dy, x=u+dx;
            if(y<0||x<0||y>=H||x>=W) continue;
            const float zn=clean.at<float>(y,x), dz=zn-zc;
            if(zn<=0||std::abs(dz)>3*s) continue;
            const float w=std::exp(-(dx*dx+dy*dy)/sp2-dz*dz/sr2);
            sum+=w*zn; ws+=w;
        }
        f.z.at<float>(v,u)=sum/ws;
    }
    // 4) Quality weight: distance noise x viewing angle (normal from the filtered depth).
    const float s1=depthSigma(1.f);
    auto P=[&](int u,int v){ const float z=f.z.at<float>(v,u); return Eigen::Vector3f((u-K.cx)*z/K.fx,(v-K.cy)*z/K.fy,z); };
    #pragma omp parallel for
    for(int v=0;v<H;++v) for(int u=0;u<W;++u){
        const float z=f.z.at<float>(v,u);
        if(z<=0) continue;
        float cosI=0.5f;
        if(u>0&&v>0&&u<W-1&&v<H-1&&f.z.at<float>(v,u-1)>0&&f.z.at<float>(v,u+1)>0&&f.z.at<float>(v-1,u)>0&&f.z.at<float>(v+1,u)>0){
            const Eigen::Vector3f n=(P(u+1,v)-P(u-1,v)).cross(P(u,v+1)-P(u,v-1)), p=P(u,v);
            if(n.norm()>0) cosI=std::abs(n.dot(p))/(n.norm()*p.norm());
        }
        const float wd=(s1/depthSigma(z))*(s1/depthSigma(z));
        f.w.at<float>(v,u)=std::min(4.f,std::max(0.02f,wd*std::max(MIN_INCIDENCE_COS,cosI)));
    }
    return f;
}

// Sharpness of a frame (variance of the Laplacian); low = motion blur.
static double sharpness(const cv::Mat& bgr){
    cv::Mat g, lap; cv::cvtColor(bgr,g,cv::COLOR_BGR2GRAY);
    cv::Laplacian(g,lap,CV_32F);
    cv::Scalar mean, sd; cv::meanStdDev(lap,mean,sd);
    return sd[0]*sd[0];
}

// ============================================================
// TSDF VOLUME (voxel hashing, 8^3 blocks)
// tsdf > 0 in front of the surface (free), < 0 behind it.
// ============================================================
class TSDFVolume {
public:
    static constexpr int B=8, N=B*B*B;
    // w = summed depth quality, cw = summed colour weight, n = observations,
    // rgb in 1/256 steps so colour keeps converging after many views.
    struct Block { float tsdf[N]; float w[N]; float cw[N]; uint16_t n[N]; uint16_t rgb[N][3]; };

    explicit TSDFVolume(double voxel):voxel_(voxel),trunc_(TRUNC_VOXELS*voxel){}
    size_t blockCount() const { return blocks_.size(); }
    double voxel() const { return voxel_; }

    static int64_t key(int x,int y,int z){
        return ((int64_t)(x+(1<<20))<<42)|((int64_t)(y+(1<<20))<<21)|(int64_t)(z+(1<<20));
    }
    static Eigen::Vector3i unkey(int64_t k){
        return {(int)((k>>42)&((1<<21)-1))-(1<<20),(int)((k>>21)&((1<<21)-1))-(1<<20),(int)(k&((1<<21)-1))-(1<<20)};
    }

    void integrate(const DepthFrame& df,const cv::Mat& bgr,const Intrinsics& K,const Pose& T_wc,float colourWeight){
        const double bs=B*voxel_;
        // 1) Blocks along the truncation band of every valid depth pixel.
        std::vector<int64_t> touched;
        std::unordered_set<int64_t> seen;
        for(int v=0;v<df.z.rows;v+=2) for(int u=0;u<df.z.cols;u+=2){
            const float z=df.z.at<float>(v,u);
            if(z<=0) continue;
            const Eigen::Vector3d ray((u-K.cx)/K.fx,(v-K.cy)/K.fy,1.0);
            for(double d=-trunc_;d<=trunc_+1e-9;d+=bs*0.5){
                const Eigen::Vector3d pw=T_wc.R*(ray*(z+d))+T_wc.t;
                const int64_t k=key((int)std::floor(pw.x()/bs),(int)std::floor(pw.y()/bs),(int)std::floor(pw.z()/bs));
                if(seen.insert(k).second) touched.push_back(k);
            }
        }
        std::vector<Block*> blk(touched.size());
        for(size_t i=0;i<touched.size();++i){
            auto& b=blocks_[touched[i]];
            if(!b){
                b.reset(new Block());
                std::fill(b->tsdf,b->tsdf+N,1.f); std::fill(b->w,b->w+N,0.f); std::fill(b->cw,b->cw+N,0.f);
                std::fill(b->n,b->n+N,(uint16_t)0); std::memset(b->rgb,0,sizeof(b->rgb));
            }
            blk[i]=b.get();
        }
        // 2) Quality-weighted update of every voxel of the touched blocks (projective TSDF).
        const Pose T_cw=T_wc.inverse();
        const Eigen::Matrix3f R=T_cw.R.cast<float>(); const Eigen::Vector3f t=T_cw.t.cast<float>();
        const float vx=(float)voxel_, tr=(float)trunc_;
        const float sx=(float)bgr.cols/df.z.cols, sy=(float)bgr.rows/df.z.rows;
        #pragma omp parallel for schedule(dynamic,16)
        for(int i=0;i<(int)touched.size();++i){
            const Eigen::Vector3i b=unkey(touched[i]);
            Block& B_=*blk[i];
            for(int z=0;z<B;++z) for(int y=0;y<B;++y) for(int x=0;x<B;++x){
                const Eigen::Vector3f pw(((b.x()*B+x)+0.5f)*vx,((b.y()*B+y)+0.5f)*vx,((b.z()*B+z)+0.5f)*vx);
                const Eigen::Vector3f pc=R*pw+t;
                if(pc.z()<MIN_DEPTH) continue;
                const float uf=K.fx*pc.x()/pc.z()+K.cx, vf=K.fy*pc.y()/pc.z()+K.cy;
                const int u=(int)std::lround(uf), v=(int)std::lround(vf);
                if(u<0||v<0||u>=df.z.cols||v>=df.z.rows) continue;
                const float zm=df.z.at<float>(v,u);
                if(zm<=0) continue;
                const float sdf=zm-pc.z();
                if(sdf<-tr) continue;
                const float wp=df.w.at<float>(v,u);
                const int idx=(z*B+y)*B+x;
                const float w0=B_.w[idx];
                B_.tsdf[idx]=(B_.tsdf[idx]*w0+std::min(1.f,sdf/tr)*wp)/(w0+wp);
                B_.w[idx]=w0+wp;
                if(B_.n[idx]<65535) ++B_.n[idx];
                if(std::abs(sdf)<tr){                                         // colour only near the surface
                    const float cwp=wp*colourWeight, c0=B_.cw[idx];
                    const int uc=std::min(bgr.cols-1,std::max(0,(int)std::lround(uf*sx))), vc=std::min(bgr.rows-1,std::max(0,(int)std::lround(vf*sy)));
                    const cv::Vec3b c=bgr.at<cv::Vec3b>(vc,uc);
                    for(int ch=0;ch<3;++ch) B_.rgb[idx][ch]=(uint16_t)std::lround((B_.rgb[idx][ch]*c0+256.f*c[2-ch]*cwp)/(c0+cwp));
                    B_.cw[idx]=c0+cwp;
                }
            }
        }
    }

    // Marching cubes over all blocks. A cube is used only if all 8 corners
    // were observed >= minObservations times with summed quality >= minWeight. Vertices are identified by their
    // voxel edge, so neighbouring cubes and blocks share them.
    Mesh extractMesh(int minObservations,float minWeight) const {
        static const int corner[8][3]={{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
        static const int edgeCorners[12][2]={{0,1},{1,2},{3,2},{0,3},{4,5},{5,6},{7,6},{4,7},{0,4},{1,5},{3,7},{2,6}};
        const vtkMarchingCubesTriangleCases* cases=vtkMarchingCubesTriangleCases::GetCases();
        std::vector<int64_t> keys; keys.reserve(blocks_.size());
        for(const auto& kv:blocks_) keys.push_back(kv.first);
        std::sort(keys.begin(),keys.end());                               // deterministic output

        struct LocalVertex { uint64_t key; Eigen::Vector3f p; std::array<uint8_t,3> c; };
        struct Local { std::vector<LocalVertex> v; std::vector<std::array<int,3>> f; };
        std::vector<Local> local(keys.size());
        const int L=B+1;
        #pragma omp parallel for schedule(dynamic,64)
        for(int bi=0;bi<(int)keys.size();++bi){
            const Eigen::Vector3i b=unkey(keys[bi]);
            // Gather the block plus one voxel layer of its +x/+y/+z neighbours.
            std::vector<float> T(L*L*L,1.f); std::vector<char> ok_(L*L*L,0); std::vector<std::array<uint8_t,3>> C(L*L*L,{0,0,0});
            for(int dz=0;dz<2;++dz) for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx){
                auto it=blocks_.find(key(b.x()+dx,b.y()+dy,b.z()+dz));
                if(it==blocks_.end()) continue;
                const Block& src=*it->second;
                for(int z=0;z<(dz?1:B);++z) for(int y=0;y<(dy?1:B);++y) for(int x=0;x<(dx?1:B);++x){
                    const int s=(z*B+y)*B+x, d=((z+dz*B)*L+(y+dy*B))*L+(x+dx*B);
                    T[d]=src.tsdf[s]; ok_[d]=src.n[s]>=minObservations&&src.w[s]>=minWeight;
                    C[d]={(uint8_t)(src.rgb[s][0]>>8),(uint8_t)(src.rgb[s][1]>>8),(uint8_t)(src.rgb[s][2]>>8)};
                }
            }
            Local& out=local[bi];
            std::unordered_map<uint64_t,int> idx;
            for(int z=0;z<B;++z) for(int y=0;y<B;++y) for(int x=0;x<B;++x){
                int g[8]; bool ok=true; int cubeIndex=0;
                for(int c=0;c<8;++c){
                    g[c]=((z+corner[c][2])*L+(y+corner[c][1]))*L+(x+corner[c][0]);
                    if(!ok_[g[c]]){ ok=false; break; }
                    if(T[g[c]]>=0.f) cubeIndex|=1<<c;
                }
                if(!ok||cubeIndex==0||cubeIndex==255) continue;
                const int* edges=cases[cubeIndex].edges;
                for(int e=0;edges[e]>=0;e+=3){
                    std::array<int,3> tri;
                    for(int k=0;k<3;++k){
                        const int ei=edges[e+k], a=edgeCorners[ei][0], c=edgeCorners[ei][1];
                        int axis=0; while(corner[a][axis]==corner[c][axis]) ++axis;
                        const int gx=b.x()*B+x+corner[a][0], gy=b.y()*B+y+corner[a][1], gz=b.z()*B+z+corner[a][2];
                        const uint64_t ek=((uint64_t)(gx+(1<<19))<<42)|((uint64_t)(gy+(1<<19))<<22)|((uint64_t)(gz+(1<<19))<<2)|(uint64_t)axis;
                        auto it=idx.find(ek);
                        if(it==idx.end()){
                            const float ta=T[g[a]], tc=T[g[c]], s=ta/(ta-tc);
                            Eigen::Vector3f p((gx+0.5f)*(float)voxel_,(gy+0.5f)*(float)voxel_,(gz+0.5f)*(float)voxel_);
                            p[axis]+=s*(float)voxel_;
                            std::array<uint8_t,3> col;
                            for(int ch=0;ch<3;++ch) col[ch]=(uint8_t)std::lround((1-s)*C[g[a]][ch]+s*C[g[c]][ch]);
                            it=idx.emplace(ek,(int)out.v.size()).first;
                            out.v.push_back({ek,p,col});
                        }
                        tri[k]=it->second;
                    }
                    out.f.push_back(tri);
                }
            }
        }
        // Merge: one vertex per voxel edge.
        Mesh m;
        std::unordered_map<uint64_t,int> global;
        size_t nv=0; for(const auto& l:local) nv+=l.v.size();
        global.reserve(nv/2);
        for(auto& l:local){
            std::vector<int> map(l.v.size());
            for(size_t i=0;i<l.v.size();++i){
                auto it=global.find(l.v[i].key);
                if(it==global.end()){
                    it=global.emplace(l.v[i].key,(int)m.v.size()).first;
                    m.v.push_back(l.v[i].p); m.rgb.push_back(l.v[i].c);
                }
                map[i]=it->second;
            }
            // Reversed winding: with the VTK case table and tsdf > 0 = free space this
            // makes the normals point out of the objects, towards the free space
            // (checked on an analytic sphere: watertight, all triangles outward).
            for(const auto& t:l.f) m.f.push_back({map[t[0]],map[t[2]],map[t[1]]});
            Local().v.swap(l.v); Local().f.swap(l.f);
        }
        return m;
    }

private:
    double voxel_, trunc_;
    std::unordered_map<int64_t,std::unique_ptr<Block>> blocks_;
};

// ============================================================
// MESH UTILITIES
// ============================================================
static void computeNormals(Mesh& m){
    m.n.assign(m.v.size(),Eigen::Vector3f::Zero());
    for(const auto& t:m.f){
        const Eigen::Vector3f fn=(m.v[t[1]]-m.v[t[0]]).cross(m.v[t[2]]-m.v[t[0]]);   // area weighted
        for(int k=0;k<3;++k) m.n[t[k]]+=fn;
    }
    for(auto& n:m.n){ const float l=n.norm(); n=l>0?Eigen::Vector3f(n/l):Eigen::Vector3f(0,0,1); }
}

// Keep vertices still referenced by faces.
static void compact(Mesh& m){
    std::vector<int> map(m.v.size(),-1);
    Mesh o;
    for(auto& t:m.f) for(int k=0;k<3;++k){
        int& id=map[t[k]];
        if(id<0){
            id=(int)o.v.size(); o.v.push_back(m.v[t[k]]);
            if(!m.rgb.empty()) o.rgb.push_back(m.rgb[t[k]]);
            if(!m.n.empty()) o.n.push_back(m.n[t[k]]);
        }
        t[k]=id;
    }
    o.f.swap(m.f);
    m=std::move(o);
}

// Taubin smoothing (alternating lambda / mu Laplacian steps): removes grain
// without the shrinkage of plain Laplacian smoothing.
static void taubinSmooth(Mesh& m,int iterations,float lambda,float mu){
    const int nv=(int)m.v.size();
    std::vector<int> start(nv+1,0);
    for(const auto& t:m.f) for(int k=0;k<3;++k) start[t[k]+1]+=2;
    for(int i=0;i<nv;++i) start[i+1]+=start[i];
    std::vector<int> nbr(start[nv]), fill(start.begin(),start.end()-1);
    for(const auto& t:m.f) for(int k=0;k<3;++k){ nbr[fill[t[k]]++]=t[(k+1)%3]; nbr[fill[t[k]]++]=t[(k+2)%3]; }
    std::vector<Eigen::Vector3f> tmp(nv);
    for(int it=0;it<2*iterations;++it){
        const float f=(it%2==0)?lambda:mu;
        #pragma omp parallel for
        for(int i=0;i<nv;++i){
            if(start[i+1]==start[i]){ tmp[i]=m.v[i]; continue; }
            Eigen::Vector3f c=Eigen::Vector3f::Zero();
            for(int j=start[i];j<start[i+1];++j) c+=m.v[nbr[j]];
            c/=(float)(start[i+1]-start[i]);
            tmp[i]=m.v[i]+f*(c-m.v[i]);
        }
        m.v.swap(tmp);
    }
}

// Remove connected pieces with fewer than minTriangles triangles.
static size_t removeSmallComponents(Mesh& m,size_t minTriangles){
    std::vector<int> parent(m.v.size());
    std::iota(parent.begin(),parent.end(),0);
    std::function<int(int)> find=[&](int x){ while(parent[x]!=x){ parent[x]=parent[parent[x]]; x=parent[x]; } return x; };
    for(const auto& t:m.f){ int a=find(t[0]); for(int k=1;k<3;++k){ int b=find(t[k]); if(a!=b) parent[b]=a; } }
    std::vector<size_t> count(m.v.size(),0);
    for(const auto& t:m.f) ++count[find(t[0])];
    std::vector<std::array<int,3>> keep; keep.reserve(m.f.size());
    for(const auto& t:m.f) if(count[find(t[0])]>=minTriangles) keep.push_back(t);
    const size_t removed=m.f.size()-keep.size();
    m.f.swap(keep);
    compact(m);
    return removed;
}

// Vertex clustering: one vertex per cell (average position / colour / normal).
static Mesh cluster(const Mesh& m,double cell,bool keepColour){
    struct Acc { Eigen::Vector3d p=Eigen::Vector3d::Zero(), n=Eigen::Vector3d::Zero(), c=Eigen::Vector3d::Zero(); int k=0; };
    std::unordered_map<int64_t,int> cellIndex;
    std::vector<Acc> acc;
    std::vector<int> map(m.v.size());
    for(size_t i=0;i<m.v.size();++i){
        const int64_t k=TSDFVolume::key((int)std::floor(m.v[i].x()/cell),(int)std::floor(m.v[i].y()/cell),(int)std::floor(m.v[i].z()/cell));
        auto it=cellIndex.find(k);
        if(it==cellIndex.end()){ it=cellIndex.emplace(k,(int)acc.size()).first; acc.emplace_back(); }
        Acc& a=acc[it->second];
        a.p+=m.v[i].cast<double>(); ++a.k;
        if(!m.n.empty()) a.n+=m.n[i].cast<double>();
        if(keepColour&&!m.rgb.empty()) a.c+=Eigen::Vector3d(m.rgb[i][0],m.rgb[i][1],m.rgb[i][2]);
        map[i]=it->second;
    }
    Mesh o;
    for(const Acc& a:acc){
        o.v.push_back((a.p/a.k).cast<float>());
        if(keepColour&&!m.rgb.empty()){ const Eigen::Vector3d c=a.c/a.k; o.rgb.push_back({(uint8_t)std::lround(c.x()),(uint8_t)std::lround(c.y()),(uint8_t)std::lround(c.z())}); }
    }
    std::unordered_set<uint64_t> seen;
    for(const auto& t:m.f){
        std::array<int,3> u={map[t[0]],map[t[1]],map[t[2]]};
        if(u[0]==u[1]||u[1]==u[2]||u[0]==u[2]) continue;
        std::array<int,3> s=u; std::sort(s.begin(),s.end());
        const uint64_t h=((uint64_t)s[0]*2654435761ull)^((uint64_t)s[1]*40503ull<<21)^((uint64_t)s[2]<<42);
        if(!seen.insert(h).second) continue;
        o.f.push_back(u);
    }
    computeNormals(o);
    compact(o);
    return o;
}

// Binary little-endian PLY: x y z nx ny nz [red green blue], faces.
static bool writePly(const std::string& path,const Mesh& m){
    std::ofstream o(path,std::ios::binary);
    if(!o) return false;
    const bool col=!m.rgb.empty(), nor=!m.n.empty();
    o<<"ply\nformat binary_little_endian 1.0\nelement vertex "<<m.v.size()<<"\n"
     <<"property float x\nproperty float y\nproperty float z\n";
    if(nor) o<<"property float nx\nproperty float ny\nproperty float nz\n";
    if(col) o<<"property uchar red\nproperty uchar green\nproperty uchar blue\n";
    o<<"element face "<<m.f.size()<<"\nproperty list uchar int vertex_indices\nend_header\n";
    for(size_t i=0;i<m.v.size();++i){
        o.write((const char*)m.v[i].data(),12);
        if(nor) o.write((const char*)m.n[i].data(),12);
        if(col) o.write((const char*)m.rgb[i].data(),3);
    }
    for(const auto& t:m.f){ const uint8_t three=3; o.write((const char*)&three,1); o.write((const char*)t.data(),12); }
    return (bool)o;
}

// ============================================================
// FLOOR PLANE: height above floor = n . p + d
// ============================================================
struct Floor { Eigen::Vector3d n=Eigen::Vector3d::UnitZ(); double d=0; bool detected=false; size_t inliers=0; double rms=0; };

static Floor findFloor(const Mesh& m){
    Floor fl;
    std::vector<Eigen::Vector3d> up;
    for(size_t i=0;i<m.v.size();++i) if(m.n[i].z()>0.9f) up.push_back(m.v[i].cast<double>());
    if(up.size()<1000){
        double zmin=1e9; for(const auto& p:m.v) zmin=std::min(zmin,(double)p.z());
        fl.d=-zmin; return fl;
    }
    // Lowest strong peak of the height histogram of upward-facing surface.
    double zlo=1e9, zhi=-1e9; for(const auto& p:up){ zlo=std::min(zlo,p.z()); zhi=std::max(zhi,p.z()); }
    const double bin=0.02; const int nb=(int)((zhi-zlo)/bin)+1;
    std::vector<int> hist(nb,0); for(const auto& p:up) ++hist[std::min(nb-1,(int)((p.z()-zlo)/bin))];
    const int hmax=*std::max_element(hist.begin(),hist.end());
    int k=0; while(hist[k]<0.25*hmax) ++k;
    std::vector<double> zs; for(const auto& p:up){ const int b=std::min(nb-1,(int)((p.z()-zlo)/bin)); if(b==k) zs.push_back(p.z()); }
    std::nth_element(zs.begin(),zs.begin()+zs.size()/2,zs.end());
    const double z0=zs[zs.size()/2];
    fl.d=-z0;
    std::vector<Eigen::Vector3d> cand;
    for(const auto& p:up) if(std::abs(p.z()-z0)<0.25) cand.push_back(p);
    std::mt19937 rng(0);
    if(cand.size()>50000){ std::shuffle(cand.begin(),cand.end(),rng); cand.resize(50000); }
    // RANSAC
    const double thr=0.012; size_t best=0; Eigen::Vector3d bn=Eigen::Vector3d::UnitZ(); double bd=-z0;
    std::uniform_int_distribution<size_t> pick(0,cand.size()-1);
    for(int it=0;it<3000;++it){
        const Eigen::Vector3d a=cand[pick(rng)], b=cand[pick(rng)], c=cand[pick(rng)];
        Eigen::Vector3d n=(b-a).cross(c-a); const double l=n.norm(); if(l<1e-9) continue;
        n/=l; if(n.z()<0) n=-n;
        const double d=-n.dot(a);
        size_t cnt=0; for(const auto& p:cand) if(std::abs(n.dot(p)+d)<thr) ++cnt;
        if(cnt>best){ best=cnt; bn=n; bd=d; }
    }
    // Least-squares refinement on the inliers.
    std::vector<Eigen::Vector3d> inl;
    for(const auto& p:cand) if(std::abs(bn.dot(p)+bd)<thr) inl.push_back(p);
    if(inl.size()<1000||bn.z()<std::cos(10*M_PI/180)){ fl.detected=true; return fl; }   // implausible: keep horizontal
    Eigen::Vector3d mean=Eigen::Vector3d::Zero(); for(const auto& p:inl) mean+=p; mean/=inl.size();
    Eigen::Matrix3d C=Eigen::Matrix3d::Zero(); for(const auto& p:inl) C+=(p-mean)*(p-mean).transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(C);
    Eigen::Vector3d n=es.eigenvectors().col(0); if(n.z()<0) n=-n;
    fl.n=n; fl.d=-n.dot(mean); fl.detected=true; fl.inliers=inl.size();
    double ss=0; for(const auto& p:inl){ const double e=n.dot(p)+fl.d; ss+=e*e; }
    fl.rms=std::sqrt(ss/inl.size());                                         // floor flatness
    return fl;
}

// ============================================================
// TOP VIEW
// ============================================================
static void topView(const Mesh& m,const Floor& fl,const std::string& dir){
    std::vector<int> ids;
    Eigen::Vector2d lo(1e9,1e9), hi(-1e9,-1e9);
    std::vector<double> h(m.v.size());
    for(size_t i=0;i<m.v.size();++i){
        h[i]=fl.n.dot(m.v[i].cast<double>())+fl.d;
        if(h[i]>=TOP_VIEW_MAX_HEIGHT) continue;                              // look under the ceiling
        ids.push_back((int)i);
        lo=lo.cwiseMin(m.v[i].head<2>().cast<double>()); hi=hi.cwiseMax(m.v[i].head<2>().cast<double>());
    }
    if(ids.empty()) return;
    std::sort(ids.begin(),ids.end(),[&](int a,int b){ return h[a]<h[b]; }); // higher surfaces drawn last
    const int W=(int)((hi.x()-lo.x())/TOP_VIEW_PX)+1, H=(int)((hi.y()-lo.y())/TOP_VIEW_PX)+1;
    cv::Mat img(H,W,CV_8UC3,cv::Scalar(30,30,30));
    for(int i:ids){
        const int x=(int)((m.v[i].x()-lo.x())/TOP_VIEW_PX), y=H-1-(int)((m.v[i].y()-lo.y())/TOP_VIEW_PX);
        const auto& c=m.rgb[i];
        img.at<cv::Vec3b>(y,x)=cv::Vec3b(c[2],c[1],c[0]);
    }
    cv::imwrite(dir+"/top_view.png",img);
}

// ============================================================
// MAIN
// ============================================================
int main(int argc,char** argv){
    std::string rec=DATASET_DIR;
    double voxel=DEFAULT_VOXEL; int every=1; bool allFrames=false;
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        if(a=="--voxel"&&i+1<argc) voxel=std::atof(argv[++i]);
        else if(a=="--every"&&i+1<argc) every=std::max(1,std::atoi(argv[++i]));
        else if(a=="--all-frames") allFrames=true;
        else if(!a.empty()&&a[0]!='-') rec=a;
        else{
            std::cerr<<"Usage: map_builder [recording_dir] [--voxel M] [--every N] [--all-frames]\n";
            return 1;
        }
    }
    const std::string recArg=rec;
    rec=resolveRecording(recArg);
    if(rec.empty()){ std::cerr<<"No recording (data.mov) found at '"<<recArg<<"' or "<<fs::path(DATASET_DIR).parent_path().string()<<"/"<<recArg<<"\n"; return 1; }
    const std::string res=rec+"/slam_results_vio", out=rec+"/reconstruction";
    if(!fs::exists(res+"/trajectory_all_frames.txt")){ std::cerr<<"No vio_slam results in "<<res<<" - run vio_slam first.\n"; return 1; }
    fs::create_directories(out);
    const auto start=std::chrono::steady_clock::now();
    auto secs=[&]{ return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(); };

    std::cout<<"\n============================================================\n"
               " MAP BUILDER (TSDF 3D reconstruction)\n"
               "============================================================\n";
    std::vector<Pose> pose; std::vector<char> tracked;
    loadPoses(res+"/trajectory_all_frames.txt",pose,tracked);
    const auto quality=loadQuality(res+"/frame_quality.txt");
    std::vector<int> frames;
    int posed=0;
    for(int i=0;i<(int)pose.size();++i){
        if(!tracked[i]) continue;
        ++posed;
        auto q=quality.find(i);
        if(i%every==0&&(allFrames||q==quality.end()||q->second=="PASS")) frames.push_back(i);
    }
    std::cout<<"[INPUT] "<<rec<<"\n[INPUT] "<<posed<<" posed frames, "<<frames.size()<<" used ("
             <<(allFrames?"all":"pose check PASS only")<<", every "<<every<<"), voxel "<<voxel*1000<<" mm, "
             <<omp_get_max_threads()<<" threads\n";

    Mesh mesh, light; int fused=0; size_t blocks=0, rawTriangles=0, removedTriangles=0;
    {
        const auto Ks=loadIntrinsics(rec);
        const auto depthFiles=getDepthFiles(rec);
        struct Item { int index; cv::Mat depth, bgr; };
        BlockingQueue<Item> queue(16);
        std::thread reader([&]{
            cv::VideoCapture cap(rec+"/data.mov");
            std::unordered_set<int> want(frames.begin(),frames.end());
            const int last=frames.empty()?-1:frames.back();
            for(int i=0;i<=last;++i){
                cv::Mat raw;
                if(!cap.read(raw)||raw.empty()) break;
                if(!want.count(i)||i>=(int)depthFiles.size()) continue;
                Item it; it.index=i;
                it.depth=cv::imread(depthFiles[i],cv::IMREAD_UNCHANGED);
                if(it.depth.type()!=CV_16UC1||it.depth.cols!=DEPTH_W||it.depth.rows!=DEPTH_H) continue;
                cv::resize(raw,it.bgr,cv::Size(COLOR_W,COLOR_H),0,0,cv::INTER_AREA);
                queue.push(std::move(it));
            }
            queue.finish();
        });
        TSDFVolume tsdf(voxel);
        std::deque<double> recentSharpness;
        Item it;
        while(queue.pop(it)){
            if(it.index>=(int)Ks.size()||!Ks[it.index].valid) continue;
            // Colour weight from sharpness relative to the recent frames (motion blur counts less).
            const double sh=sharpness(it.bgr);
            recentSharpness.push_back(sh); if((int)recentSharpness.size()>BLUR_WINDOW) recentSharpness.pop_front();
            std::vector<double> sorted(recentSharpness.begin(),recentSharpness.end());
            std::nth_element(sorted.begin(),sorted.begin()+sorted.size()/2,sorted.end());
            const float colourWeight=(float)std::min(1.0,std::max(0.25,sh/std::max(1e-6,sorted[sorted.size()/2])));
            tsdf.integrate(prepareDepth(it.depth,it.bgr,Ks[it.index]),it.bgr,Ks[it.index],pose[it.index],colourWeight);
            if(++fused%500==0) std::cout<<"[FUSE] "<<fused<<"/"<<frames.size()<<" frames, "<<tsdf.blockCount()<<" blocks ("
                                        <<std::fixed<<std::setprecision(0)<<secs()<<" s)"<<std::endl;
        }
        reader.join();
        blocks=tsdf.blockCount();
        std::cout<<"[FUSE] "<<fused<<" frames fused, "<<blocks<<" voxel blocks ("<<std::fixed<<std::setprecision(0)<<secs()<<" s)\n";

        mesh=tsdf.extractMesh(MIN_OBSERVATIONS,MIN_SURFACE_WEIGHT);
        rawTriangles=mesh.f.size();
        { TSDFVolume freed(voxel); std::swap(tsdf,freed); }                  // release the volume
        removedTriangles=removeSmallComponents(mesh,MIN_COMPONENT_TRIANGLES);
        taubinSmooth(mesh,TAUBIN_ITERATIONS,TAUBIN_LAMBDA,TAUBIN_MU);
        computeNormals(mesh);
        std::cout<<"[MESH] "<<rawTriangles<<" triangles -> "<<mesh.f.size()<<" after removing small fragments ("
                 <<secs()<<" s)\n";
        writePly(out+"/room_mesh.ply",mesh);
        light=cluster(mesh,LIGHT_CELL,true);
        writePly(out+"/room_mesh_light.ply",light);
        std::cout<<"[MESH] light "<<light.f.size()<<" triangles ("<<secs()<<" s)\n";
    }

    const Floor fl=findFloor(mesh);
    topView(mesh,fl,out);

    Eigen::Vector3f bmin=Eigen::Vector3f::Constant(1e9f), bmax=Eigen::Vector3f::Constant(-1e9f);
    for(const auto& p:mesh.v){ bmin=bmin.cwiseMin(p); bmax=bmax.cwiseMax(p); }
    const double tilt=std::acos(std::min(1.0,fl.n.z()))*180/M_PI;
    std::ostringstream r; r<<std::fixed<<std::setprecision(3);
    r<<"recording            "<<rec<<"\n";
    r<<"frames fused         "<<fused<<" of "<<posed<<" ("<<(allFrames?"all":"pose check PASS only")<<", every "<<every<<")\n"
          <<"TSDF                 voxel "<<std::setprecision(1)<<voxel*1000<<" mm, truncation "<<TRUNC_VOXELS*voxel*100<<" cm, depth "
          <<std::setprecision(2)<<MIN_DEPTH<<"-"<<MAX_DEPTH<<" m, surface seen >= "<<MIN_OBSERVATIONS<<" times with quality >= "
          <<MIN_SURFACE_WEIGHT<<", "<<blocks<<" blocks\n"
          <<"depth clean-up       edges + blown-out colour dropped, edge-preserving filter, distance + viewing-angle weights\n"
          <<"smoothing            Taubin, "<<TAUBIN_ITERATIONS<<" iterations\n"
          <<"fragments removed    "<<removedTriangles<<" of "<<rawTriangles<<" triangles (pieces < "<<MIN_COMPONENT_TRIANGLES<<" triangles)\n";
    r<<std::setprecision(3)
     <<"room_mesh.ply        "<<mesh.v.size()<<" vertices, "<<mesh.f.size()<<" triangles (coloured)\n"
     <<"room_mesh_light.ply  "<<light.f.size()<<" triangles ("<<LIGHT_CELL*100<<" cm vertex spacing, coloured)\n"
     <<"model extent         "<<(bmax-bmin).x()<<" x "<<(bmax-bmin).y()<<" x "<<(bmax-bmin).z()<<" m (x, y, z)\n"
     <<"bounding box         min ["<<bmin.x()<<", "<<bmin.y()<<", "<<bmin.z()<<"]  max ["<<bmax.x()<<", "<<bmax.y()<<", "<<bmax.z()<<"]\n"
     <<"floor plane          n = ["<<std::setprecision(4)<<fl.n.x()<<", "<<fl.n.y()<<", "<<fl.n.z()<<"], d = "<<std::setprecision(3)<<fl.d
     <<" (tilt "<<std::setprecision(2)<<tilt<<" deg from the VIO z axis, "<<fl.inliers<<" inlier points; "
     <<(fl.detected?"detected":"NOT detected - lowest surface used")<<")\n"
     <<"floor flatness       "<<std::setprecision(1)<<fl.rms*1000<<" mm RMS (floor points vs fitted plane)\n"
     <<"frame                vio_slam world frame, metres, z up (gravity -z)\n"
     <<"time                 "<<secs()<<" s\n";
    std::ofstream(out+"/reconstruction_report.txt")<<r.str();
    std::cout<<"\n"<<r.str()<<"Output: "<<out<<"\n";
    return 0;
}
