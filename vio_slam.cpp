// ============================================================
// VISUAL-INERTIAL RGB-D SLAM  (dense tracking + IMU + PGO + BA + TSDF)
//
// Input (aligned RGB-D + IMU, phone recording):
//   data.mov          RGB video (1920x1440)
//   frames2/*.png     16-bit depth (mm), 256x192, aligned to RGB
//   data.jsonl        frame timestamps + per-frame intrinsics,
//                     gyroscope + accelerometer (~100 Hz)
//   calibration.json  image size + IMU->camera rotation
//
// Pipeline
//   1. FRONT-END (every frame)
//      Dense direct RGB-D alignment of the frame against the current
//      keyframe: photometric + depth residuals over all valid pixels,
//      3-level pyramid, Huber weights, Gauss-Newton on SE(3).
//      Initial guess = IMU-propagated pose. Fallbacks:
//        a) ORB + 3D-3D RANSAC initialization, dense again
//        b) promote the last good frame to a keyframe, retry
//        c) IMU-only prediction (frame flagged)
//   2. KEYFRAMES
//      New keyframe if translation > KF_TRANSLATION, rotation >
//      KF_ROTATION_DEG, dense overlap < KF_MIN_INLIER_RATIO or
//      time since last keyframe > KF_MAX_DT.
//   3. VISUAL-INERTIAL SLIDING WINDOW (every keyframe, Ceres)
//      States per keyframe: pose, velocity, gyro bias, accel bias.
//      Factors: dense relative-pose factors (information from the
//      dense Hessian), IMU preintegration factors (bias Jacobians,
//      re-integration on bias change), ORB landmark reprojection +
//      depth factors (local RGB-D bundle adjustment).
//      IMU initialization from the first keyframes: gyro bias from
//      visual/gyro rotation alignment, gravity from the accelerometer,
//      velocities from linear least squares.
//   4. LOOP CLOSURE
//      Old keyframes near the current one but far back in time,
//      verified by ORB matching -> 3D-3D RANSAC -> dense alignment.
//      Accepted loops trigger a global pose-graph optimization.
//   5. FINAL GLOBAL BUNDLE ADJUSTMENT
//      All keyframes + landmarks + IMU factors + loop factors.
//   6. 3D SCENE
//      TSDF fusion (voxel hashing) with the final poses, surface
//      extraction to a colored point cloud.
//   7. VERIFICATION
//      Every keyframe pose is checked by dense re-alignment against its
//      neighbours, loops and held-out revisits, plus landmark reprojection.
//
// Usage  (recording_dir: full path or folder name next to DATASET_DIR;
//         default DATASET_DIR)
//   ./vio_slam [recording_dir] optimized pose for every frame, dense
//                              reprojection check of EVERY frame pose with
//                              automatic refinement where it fails,
//                              3D scene; all results saved
//   ./vio_slam --live          same, while showing the RGB frames and the 3D
//                              scene being built (blue trajectory, red camera)
//   options: --max-frames N   --no-imu (camera-only, separate output dir)
//   Replay saved results at a fixed 30 FPS with ./scene_player
//
// Pose convention: p_world = R * p_camera + t (camera = body frame;
// IMU measurements are rotated into the camera frame, lever arm 0).
// ============================================================

#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

// ============================================================
// DATASET
// ============================================================
static const std::string DATASET_DIR =
    "/home/mohit/3D/recording_2026-09-17_13-13-55";
// Paths of the recording being processed (DATASET_DIR, or the recording
// folder given on the command line).
static std::string RECORDING_DIR, VIDEO_PATH, DEPTH_DIR, DATA_JSONL_PATH, CALIBRATION_PATH;
static void setRecording(const std::string& dir){
    RECORDING_DIR=dir;
    VIDEO_PATH=dir+"/data.mov";
    DEPTH_DIR=dir+"/frames2";
    DATA_JSONL_PATH=dir+"/data.jsonl";
    CALIBRATION_PATH=dir+"/calibration.json";
}

// Recording folder from the command line: a path (absolute or relative to
// the current directory) or just the folder name, looked up next to DATASET_DIR.
static std::string resolveRecording(const std::string& arg){
    std::error_code ec;
    for(const fs::path& p:{fs::path(arg),fs::path(DATASET_DIR).parent_path()/arg})
        if(fs::exists(p/"data.mov",ec)) return fs::canonical(p,ec).string();
    return "";
}


// ============================================================
// IMAGE / CAMERA
// ============================================================
static constexpr int DEPTH_WIDTH  = 256;
static constexpr int DEPTH_HEIGHT = 192;
static constexpr int RGB_WIDTH    = 512;
static constexpr int RGB_HEIGHT   = 384;
static constexpr float RGB_TO_DEPTH_SCALE = 0.5f;
static constexpr float DEPTH_SCALE = 0.001f;
static constexpr float MIN_DEPTH = 0.15f;
static constexpr float MAX_DEPTH = 3.0f;
// Fallback intrinsics at depth resolution (if data.jsonl has none)
static constexpr float FX = 179.424f, FY = 179.424f, CX = 128.852f, CY = 95.357f;

// ============================================================
// DENSE RGB-D ALIGNMENT
// ============================================================
static constexpr int DENSE_LEVELS = 3;
static constexpr int DENSE_ITERATIONS[DENSE_LEVELS] = {6, 10, 14};   // level 0 (fine) .. 2 (coarse)
static constexpr int DENSE_STRIDE_L0 = 1;
static constexpr int DENSE_MIN_POINTS = 300;
static constexpr float DENSE_HUBER = 1.345f;          // in sigma units
static constexpr float DENSE_PHOTO_WEIGHT = 1.0f;
static constexpr float DENSE_DEPTH_WEIGHT = 1.0f;
static constexpr float DEPTH_SIGMA_A = 0.004f;        // sigma_z = A + B z^2  [m]
static constexpr float DEPTH_SIGMA_B = 0.006f;
static constexpr float DENSE_OCCLUSION = 0.10f;       // |dz| above this = occlusion
static constexpr float DENSE_MIN_INLIER_RATIO = 0.30f;
// Dense Hessians assume independent pixels and are far too confident.
// Calibrated on this recording: with sigma ~2 mm per keyframe the chain
// predicted ~6x less drift than the loop closures measured.
static constexpr double DENSE_COV_INFLATION = 3600.0;
static constexpr double DENSE_MIN_ROT_SIGMA = 0.003;    // rad
static constexpr double DENSE_MIN_TRANS_SIGMA = 0.004;  // m
// Robust kernel scale (in sigmas) for loop / revisit edges.
static constexpr double LOOP_CAUCHY_SCALE = 3.0;
// Rotation consistency with the IMU prediction:  base + scale * |rotation|
static constexpr double GATE_ROT_BASE_DEG = 3.0;
static constexpr double GATE_ROT_SCALE = 0.3;
static constexpr double GATE_MAX_TRANS = 0.30;

// ============================================================
// FEATURES (ORB: fallback init, landmarks for BA, loop closure)
// ============================================================
static constexpr int ORB_FEATURES = 4000;
static constexpr int GRID_X = 20;
static constexpr int GRID_Y = 15;
static constexpr int FEATURES_PER_CELL = 2;
static constexpr int MAX_SELECTED_FEATURES = 600;
static constexpr float LOWE_RATIO = 0.75f;
static constexpr int RANSAC_ITERATIONS = 1500;
static constexpr double RANSAC_THRESHOLD = 0.025;
static constexpr int MIN_RANSAC_INLIERS = 12;

// ============================================================
// KEYFRAMES
// ============================================================
static constexpr double KF_TRANSLATION = 0.10;
static constexpr double KF_ROTATION_DEG = 10.0;
static constexpr float KF_MIN_INLIER_RATIO = 0.55f;
static constexpr double KF_MAX_DT = 0.5;
static constexpr int FORCE_KF_AFTER_FAILURES = 5;

// ============================================================
// IMU (continuous-time noise densities, inflated phone values)
// ============================================================
static constexpr double GRAVITY = 9.81;
static constexpr double GYRO_NOISE = 0.010;       // rad/s/sqrt(Hz)
static constexpr double ACC_NOISE = 0.10;         // m/s^2/sqrt(Hz)
static constexpr double GYRO_BIAS_RW = 2.0e-4;    // rad/s^2/sqrt(Hz)
static constexpr double ACC_BIAS_RW = 3.0e-3;     // m/s^3/sqrt(Hz)
static constexpr double REINTEGRATE_BG = 0.002;   // re-integrate if bias moved more
static constexpr double REINTEGRATE_BA = 0.05;
static constexpr int IMU_INIT_KEYFRAMES = 8;
static constexpr double IMU_INIT_MIN_SPAN = 1.5;  // s

// ============================================================
// SLIDING WINDOW / BA / LOOPS
// ============================================================
static constexpr int WINDOW_KEYFRAMES = 10;
static constexpr int WINDOW_ITERATIONS = 10;
static constexpr int LANDMARK_MATCH_KEYFRAMES = 4;
static constexpr double LANDMARK_MAX_DIST = 0.10;   // m, association gate
static constexpr double REPROJ_SIGMA_PX = 1.0;      // pixels at depth resolution
static constexpr double PRIOR_V_SIGMA = 0.10;
static constexpr double PRIOR_BG_SIGMA = 0.01;
static constexpr double PRIOR_BA_SIGMA = 0.10;
static constexpr int LOOP_MIN_FRAME_GAP = 150;
static constexpr double LOOP_SEARCH_RADIUS = 2.0;
static constexpr double LOOP_MAX_VIEW_ANGLE_DEG = 60.0;
static constexpr int LOOP_MAX_CANDIDATES = 8;
static constexpr int LOOP_MIN_MATCHES = 30;
static constexpr int LOOP_MIN_INLIERS = 25;
static constexpr float LOOP_MIN_DENSE_INLIER_RATIO = 0.40f;
static constexpr int LOOP_COOLDOWN_KEYFRAMES = 3;
static constexpr int PGO_ITERATIONS = 30;
static constexpr int FINAL_BA_ITERATIONS = 40;
static constexpr double FINAL_BA_LANDMARK_SIGMA_SCALE = 3.0;
static constexpr double PGO_SEQ_ROT_SIGMA = 0.005;
static constexpr double PGO_SEQ_TRANS_SIGMA = 0.01;
// Revisit constraints (final stage): old keyframes near a keyframe but far
// back in time, verified by dense alignment. Keyframes with odd id are held
// out so the verification stage can still test them independently.
static constexpr double REVISIT_MAX_DISTANCE = 0.40;     // m
static constexpr double REVISIT_MAX_ANGLE_DEG = 30.0;
static constexpr float REVISIT_MIN_INLIER_RATIO = 0.50f;
static constexpr double REVISIT_MAX_CORRECTION_T = 0.15; // m, gross-consistency gate
static constexpr double REVISIT_MAX_CORRECTION_R = 8.0;  // deg
static constexpr int REVISIT_ROUNDS = 2;
static bool isHeldOut(int kfId){ return kfId%2==1; }

// ============================================================
// TSDF / MAP / VIEWER
// ============================================================
static constexpr float TSDF_VOXEL = 0.01f;
static constexpr float TSDF_TRUNCATION = 0.04f;
static constexpr int TSDF_FRAME_INTERVAL = 3;
static constexpr float TSDF_MIN_WEIGHT = 2.0f;
static constexpr float TSDF_MAX_WEIGHT = 100.0f;
static constexpr int LIVE_KF_STRIDE = 2;            // depth pixels per keyframe cloud point
static constexpr int LIVE_FRAME_STRIDE = 4;         // depth pixels per current-frame point
static constexpr float LIVE_VOXEL = 0.010f;          // keyframe cloud voxel size
static constexpr float LIVE_EDGE_JUMP = 0.05f;       // relative depth jump = flying pixel
static constexpr double LIVE_FOLLOW_DISTANCE = 1.0;  // viewer camera behind the phone (m)

// ============================================================
// SE(3)
// ============================================================
struct Pose {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Vector3d transform(const Eigen::Vector3d& p) const { return R*p+t; }
    Pose inverse() const { Pose o; o.R=R.transpose(); o.t=-o.R*t; return o; }
    Pose operator*(const Pose& b) const { Pose o; o.R=R*b.R; o.t=R*b.t+t; return o; }
};

template<typename S>
static Eigen::Matrix<S,3,3> skew(const Eigen::Matrix<S,3,1>& v){
    Eigen::Matrix<S,3,3> m;
    m << S(0),-v.z(),v.y(), v.z(),S(0),-v.x(), -v.y(),v.x(),S(0);
    return m;
}

static Eigen::Matrix3d expSO3(const Eigen::Vector3d& w){
    double th=w.norm();
    if(th<1e-10) return Eigen::Matrix3d::Identity()+skew(w);
    return Eigen::AngleAxisd(th,w/th).toRotationMatrix();
}

static Eigen::Vector3d logSO3(const Eigen::Matrix3d& R){
    Eigen::AngleAxisd aa(Eigen::Quaterniond(R).normalized());
    Eigen::Vector3d w=aa.axis()*aa.angle();
    if(aa.angle()>M_PI) w=aa.axis()*(aa.angle()-2*M_PI);
    return w.allFinite()?w:Eigen::Vector3d::Zero();
}

static Eigen::Matrix3d rightJacobianSO3(const Eigen::Vector3d& w){
    double th=w.norm();
    Eigen::Matrix3d W=skew(w);
    if(th<1e-6) return Eigen::Matrix3d::Identity()-0.5*W;
    return Eigen::Matrix3d::Identity()-(1-std::cos(th))/(th*th)*W+(th-std::sin(th))/(th*th*th)*W*W;
}

// Left-multiplied SE(3) increment with xi = (omega, v).
template<typename S>
static void applyLeftSE3(Eigen::Matrix<S,3,3>& R,Eigen::Matrix<S,3,1>& t,const Eigen::Matrix<S,6,1>& xi){
    Eigen::Matrix<S,3,1> w=xi.template head<3>(), v=xi.template tail<3>();
    S th=w.norm();
    Eigen::Matrix<S,3,3> W=skew(w), dR, V;
    if(th<S(1e-8)){ dR=Eigen::Matrix<S,3,3>::Identity()+W; V=Eigen::Matrix<S,3,3>::Identity()+S(0.5)*W; }
    else{
        S a=std::sin(th)/th, b=(S(1)-std::cos(th))/(th*th), c=(S(1)-a)/(th*th);
        dR=Eigen::Matrix<S,3,3>::Identity()+a*W+b*W*W;
        V=Eigen::Matrix<S,3,3>::Identity()+b*W+c*W*W;
    }
    R=dR*R; t=dR*t+V*v;
    // Re-orthonormalize.
    Eigen::JacobiSVD<Eigen::Matrix<S,3,3>> svd(R,Eigen::ComputeFullU|Eigen::ComputeFullV);
    R=svd.matrixU()*svd.matrixV().transpose();
}

static double rotationDeg(const Eigen::Matrix3d& a,const Eigen::Matrix3d& b){
    return logSO3(a.transpose()*b).norm()*180.0/M_PI;
}

static Pose interpolatePose(const Pose& a,const Pose& b,double alpha){
    Pose o;
    Eigen::Quaterniond qa(a.R), qb(b.R);
    o.R=qa.slerp(alpha,qb).normalized().toRotationMatrix();
    o.t=(1-alpha)*a.t+alpha*b.t;
    return o;
}

// ============================================================
// PHONE DATA
// ============================================================
struct Intrinsics { float fx=FX, fy=FY, cx=CX, cy=CY; };

struct ImuSample { double t=0; Eigen::Vector3d gyro=Eigen::Vector3d::Zero(), acc=Eigen::Vector3d::Zero(); };

struct PhoneData {
    std::vector<double> frameTime;
    std::vector<Intrinsics> frameK;
    std::vector<ImuSample> imu;            // camera frame, time-sorted
    Eigen::Matrix3d R_cam_imu=Eigen::Matrix3d::Identity();
    bool hasImu() const { return imu.size()>100; }
    // Index of the last sample with time <= t (0 if none).
    size_t sampleAt(double t) const {
        auto it=std::upper_bound(imu.begin(),imu.end(),t,[](double x,const ImuSample& s){return x<s.t;});
        return it==imu.begin()?0:(size_t)(it-imu.begin()-1);
    }
};

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

static std::vector<double> jsonNumberArray(const std::string& s,const std::string& key){
    std::vector<double> out;
    size_t p=s.find("\""+key+"\":");
    if(p==std::string::npos) return out;
    size_t a=s.find('[',p), b=matchBracket(s,a);
    if(b==std::string::npos) return out;
    std::string body=s.substr(a,b-a+1);
    for(char& ch:body) if(ch=='['||ch==']'||ch==',') ch=' ';
    std::istringstream ss(body); double v;
    while(ss>>v) out.push_back(v);
    return out;
}

static bool loadPhoneData(PhoneData& pd){
    double imageW=1920.0, imageH=1440.0;
    // Validated on this recording: omega_cam = R_cam_imu * omega_imu.
    pd.R_cam_imu << 0,-1,0, -1,0,0, 0,0,-1;
    {
        std::ifstream f(CALIBRATION_PATH);
        if(f.is_open()){
            std::stringstream ss; ss<<f.rdbuf(); std::string s=ss.str(); double v;
            if(jsonNumber(s,"imageWidth",v)&&v>0) imageW=v;
            if(jsonNumber(s,"imageHeight",v)&&v>0) imageH=v;
            auto m=jsonNumberArray(s,"imuToCamera");
            if(m.size()>=16) for(int r=0;r<3;++r) for(int c=0;c<3;++c) pd.R_cam_imu(r,c)=m[r*4+c];
        }
    }
    const double sx=DEPTH_WIDTH/imageW, sy=DEPTH_HEIGHT/imageH;

    std::ifstream f(DATA_JSONL_PATH);
    if(!f.is_open()){ std::cerr<<"[DATA] Cannot open "<<DATA_JSONL_PATH<<"\n"; return false; }
    std::vector<std::pair<int,double>> times;
    std::vector<std::pair<int,Intrinsics>> ks;
    std::vector<std::pair<double,Eigen::Vector3d>> gyro, acc;
    std::string line;
    while(std::getline(f,line)){
        if(line.empty()) continue;
        size_t fk=line.find("\"frames\":");
        if(fk!=std::string::npos){
            size_t a=line.find('[',fk), b=matchBracket(line,a);
            if(b==std::string::npos) continue;
            const std::string arr=line.substr(a,b-a+1);
            const std::string outer=line.substr(0,fk)+line.substr(b+1);
            double num=-1,t=0;
            if(!jsonNumber(outer,"number",num)||!jsonNumber(outer,"time",t)) continue;
            times.emplace_back((int)num,t);
            Intrinsics best; bool found=false; size_t p=1;
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
                Intrinsics k; k.fx=(float)(fx*sx); k.fy=(float)(fy*sy); k.cx=(float)(cx*sx); k.cy=(float)(cy*sy);
                if(!found||camInd==1){ best=k; found=true; }   // depth camera preferred
            }
            if(found) ks.emplace_back((int)num,best);
            continue;
        }
        const bool isGyro=line.find("\"gyroscope\"")!=std::string::npos;
        const bool isAcc=line.find("\"accelerometer\"")!=std::string::npos;
        if(!isGyro&&!isAcc) continue;
        double t; auto v=jsonNumberArray(line,"values");
        if(v.size()<3||!jsonNumber(line,"time",t)) continue;
        Eigen::Vector3d m=pd.R_cam_imu*Eigen::Vector3d(v[0],v[1],v[2]);
        (isGyro?gyro:acc).emplace_back(t,m);
    }
    auto byTime=[](const auto& a,const auto& b){return a.first<b.first;};
    std::sort(gyro.begin(),gyro.end(),byTime);
    std::sort(acc.begin(),acc.end(),byTime);

    // IMU samples on gyro timestamps, accelerometer linearly interpolated.
    size_t j=0;
    for(const auto& g:gyro){
        if(acc.size()<2) break;
        while(j+2<acc.size()&&acc[j+1].first<g.first) ++j;
        double t0=acc[j].first, t1=acc[j+1].first;
        double a=(t1>t0)?std::max(0.0,std::min(1.0,(g.first-t0)/(t1-t0))):0.0;
        ImuSample s; s.t=g.first; s.gyro=g.second; s.acc=(1-a)*acc[j].second+a*acc[j+1].second;
        pd.imu.push_back(s);
    }

    int maxNum=-1;
    for(const auto& t:times) maxNum=std::max(maxNum,t.first);
    if(maxNum<0) return false;
    pd.frameTime.assign(maxNum+1,std::numeric_limits<double>::quiet_NaN());
    pd.frameK.assign(maxNum+1,Intrinsics());
    for(const auto& t:times) pd.frameTime[t.first]=t.second;
    for(const auto& k:ks) pd.frameK[k.first]=k.second;
    for(size_t i=0;i<pd.frameTime.size();++i)
        if(!std::isfinite(pd.frameTime[i]))
            pd.frameTime[i]=(i>0)?pd.frameTime[i-1]+1.0/30.0:0.0;

    std::cout<<"[DATA] Frames "<<times.size()<<", intrinsics "<<ks.size()
             <<", gyro "<<gyro.size()<<", accel "<<acc.size()<<", IMU samples "<<pd.imu.size()<<"\n";
    std::cout<<"[DATA] Duration "<<std::fixed<<std::setprecision(2)<<pd.frameTime.back()-pd.frameTime.front()<<" s\n";
    return true;
}

static std::vector<std::string> getDepthFiles(){
    std::vector<std::string> files;
    if(!fs::exists(DEPTH_DIR)) return files;
    for(const auto& e:fs::directory_iterator(DEPTH_DIR)){
        if(!e.is_regular_file()) continue;
        std::string ext=e.path().extension().string();
        std::transform(ext.begin(),ext.end(),ext.begin(),::tolower);
        if(ext==".png") files.push_back(e.path().string());
    }
    std::sort(files.begin(),files.end());
    return files;
}

static bool loadDepth(const std::string& path,cv::Mat& depth){
    depth=cv::imread(path,cv::IMREAD_UNCHANGED);
    return !depth.empty()&&depth.type()==CV_16UC1&&depth.cols==DEPTH_WIDTH&&depth.rows==DEPTH_HEIGHT;
}

// ============================================================
// IMU PREINTEGRATION (on-manifold, Forster et al.)
//   R_j = R_i dR
//   v_j = v_i + g dt + R_i dV
//   p_j = p_i + v_i dt + 1/2 g dt^2 + R_i dP
// Bias changes are handled to first order with the Jacobians below,
// and by full re-integration when the bias moves too far.
// ============================================================
class Preintegration {
public:
    double t0=0, t1=0, dt=0;
    Eigen::Vector3d bg=Eigen::Vector3d::Zero(), ba=Eigen::Vector3d::Zero();
    Eigen::Matrix3d dR=Eigen::Matrix3d::Identity();
    Eigen::Vector3d dV=Eigen::Vector3d::Zero(), dP=Eigen::Vector3d::Zero();
    Eigen::Matrix3d JR_bg=Eigen::Matrix3d::Zero(), JV_bg=Eigen::Matrix3d::Zero(), JV_ba=Eigen::Matrix3d::Zero();
    Eigen::Matrix3d JP_bg=Eigen::Matrix3d::Zero(), JP_ba=Eigen::Matrix3d::Zero();
    Eigen::Matrix<double,9,9> cov=Eigen::Matrix<double,9,9>::Zero();

    void integrate(const PhoneData& pd,double from,double to,const Eigen::Vector3d& bgIn,const Eigen::Vector3d& baIn){
        *this=Preintegration(); t0=from; t1=to; bg=bgIn; ba=baIn;
        if(!pd.hasImu()||!(to>from)) return;
        size_t k=pd.sampleAt(from);
        double t=from;
        while(t<to&&k<pd.imu.size()){
            double tn=(k+1<pd.imu.size())?std::min(to,pd.imu[k+1].t):to;
            if(tn>t) step(tn-t,pd.imu[k].gyro,pd.imu[k].acc);
            t=std::max(t,tn); ++k;
        }
    }

    // Prediction of the end state from the start state.
    void predict(const Eigen::Matrix3d& Ri,const Eigen::Vector3d& pi,const Eigen::Vector3d& vi,const Eigen::Vector3d& g,
                 Eigen::Matrix3d& Rj,Eigen::Vector3d& pj,Eigen::Vector3d& vj) const {
        Rj=Ri*dR; vj=vi+g*dt+Ri*dV; pj=pi+vi*dt+0.5*g*dt*dt+Ri*dP;
    }

private:
    void step(double h,const Eigen::Vector3d& gyro,const Eigen::Vector3d& accel){
        const Eigen::Vector3d w=gyro-bg, a=accel-ba;
        const Eigen::Matrix3d dRi=expSO3(w*h), Jr=rightJacobianSO3(w*h), Ax=skew(a);

        Eigen::Matrix<double,9,9> A=Eigen::Matrix<double,9,9>::Identity();
        A.block<3,3>(0,0)=dRi.transpose();
        A.block<3,3>(3,0)=-dR*Ax*h;
        A.block<3,3>(6,0)=-0.5*dR*Ax*h*h;
        A.block<3,3>(6,3)=Eigen::Matrix3d::Identity()*h;
        Eigen::Matrix<double,9,6> B=Eigen::Matrix<double,9,6>::Zero();
        B.block<3,3>(0,0)=Jr*h;
        B.block<3,3>(3,3)=dR*h;
        B.block<3,3>(6,3)=0.5*dR*h*h;
        Eigen::Matrix<double,6,6> Q=Eigen::Matrix<double,6,6>::Zero();
        Q.block<3,3>(0,0)=Eigen::Matrix3d::Identity()*GYRO_NOISE*GYRO_NOISE/h;
        Q.block<3,3>(3,3)=Eigen::Matrix3d::Identity()*ACC_NOISE*ACC_NOISE/h;
        cov=A*cov*A.transpose()+B*Q*B.transpose();

        JP_ba+=JV_ba*h-0.5*dR*h*h;
        JP_bg+=JV_bg*h-0.5*dR*Ax*JR_bg*h*h;
        JV_ba+=-dR*h;
        JV_bg+=-dR*Ax*JR_bg*h;
        JR_bg=dRi.transpose()*JR_bg-Jr*h;

        dP+=dV*h+0.5*dR*a*h*h;
        dV+=dR*a*h;
        dR=dR*dRi;
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(dR,Eigen::ComputeFullU|Eigen::ComputeFullV);
        dR=svd.matrixU()*svd.matrixV().transpose();
        dt+=h;
    }
};

// ============================================================
// DENSE IMAGE PYRAMID
// ============================================================
struct DenseLevel {
    int w=0, h=0;
    float fx=0, fy=0, cx=0, cy=0;
    cv::Mat I, Ix, Iy, Z, Zx, Zy;   // CV_32F; Z = 0 where invalid
};
struct DenseImage { std::vector<DenseLevel> lv; };

static void intensityGradients(const cv::Mat& I,cv::Mat& gx,cv::Mat& gy){
    gx=cv::Mat::zeros(I.size(),CV_32F); gy=cv::Mat::zeros(I.size(),CV_32F);
    for(int v=1;v<I.rows-1;++v){
        const float* r=I.ptr<float>(v); const float* ru=I.ptr<float>(v-1); const float* rd=I.ptr<float>(v+1);
        float* px=gx.ptr<float>(v); float* py=gy.ptr<float>(v);
        for(int u=1;u<I.cols-1;++u){ px[u]=0.5f*(r[u+1]-r[u-1]); py[u]=0.5f*(rd[u]-ru[u]); }
    }
}

static void depthGradients(const cv::Mat& Z,cv::Mat& gx,cv::Mat& gy){
    gx=cv::Mat::zeros(Z.size(),CV_32F); gy=cv::Mat::zeros(Z.size(),CV_32F);
    for(int v=1;v<Z.rows-1;++v){
        const float* r=Z.ptr<float>(v); const float* ru=Z.ptr<float>(v-1); const float* rd=Z.ptr<float>(v+1);
        float* px=gx.ptr<float>(v); float* py=gy.ptr<float>(v);
        for(int u=1;u<Z.cols-1;++u){
            if(r[u]<=0) continue;
            if(r[u-1]>0&&r[u+1]>0&&std::abs(r[u+1]-r[u-1])<0.1f) px[u]=0.5f*(r[u+1]-r[u-1]);
            if(ru[u]>0&&rd[u]>0&&std::abs(rd[u]-ru[u])<0.1f) py[u]=0.5f*(rd[u]-ru[u]);
        }
    }
}

// 2x2 downsampling that averages only mutually consistent valid depths.
static cv::Mat downsampleDepth(const cv::Mat& Z){
    cv::Mat out=cv::Mat::zeros(Z.rows/2,Z.cols/2,CV_32F);
    for(int v=0;v<out.rows;++v) for(int u=0;u<out.cols;++u){
        float s=0, ref=0; int n=0;
        for(int dv=0;dv<2;++dv) for(int du=0;du<2;++du){
            float z=Z.at<float>(2*v+dv,2*u+du);
            if(z<=0) continue;
            if(n==0) ref=z;
            if(std::abs(z-ref)<0.05f){ s+=z; ++n; }
        }
        if(n>=2) out.at<float>(v,u)=s/n;
    }
    return out;
}

static DenseImage buildDenseImage(const cv::Mat& gray8,const cv::Mat& depth16,const Intrinsics& K){
    DenseImage d;
    cv::Mat I0; gray8.convertTo(I0,CV_32F,1.0/255.0);
    cv::Mat Z0(depth16.size(),CV_32F);
    for(int v=0;v<depth16.rows;++v){
        const uint16_t* s=depth16.ptr<uint16_t>(v); float* o=Z0.ptr<float>(v);
        for(int u=0;u<depth16.cols;++u){ float z=s[u]*DEPTH_SCALE; o[u]=(z>=MIN_DEPTH&&z<=MAX_DEPTH)?z:0.f; }
    }
    d.lv.resize(DENSE_LEVELS);
    for(int L=0;L<DENSE_LEVELS;++L){
        DenseLevel& l=d.lv[L];
        if(L==0){ l.I=I0; l.Z=Z0; l.fx=K.fx; l.fy=K.fy; l.cx=K.cx; l.cy=K.cy; }
        else{
            const DenseLevel& p=d.lv[L-1];
            cv::pyrDown(p.I,l.I,cv::Size(p.I.cols/2,p.I.rows/2));
            l.Z=downsampleDepth(p.Z);
            l.fx=p.fx*0.5f; l.fy=p.fy*0.5f; l.cx=(p.cx+0.5f)*0.5f-0.5f; l.cy=(p.cy+0.5f)*0.5f-0.5f;
        }
        l.w=l.I.cols; l.h=l.I.rows;
        intensityGradients(l.I,l.Ix,l.Iy);
        depthGradients(l.Z,l.Zx,l.Zy);
    }
    return d;
}

static inline float bilinear(const cv::Mat& m,float x,float y){
    int x0=(int)x, y0=(int)y; float ax=x-x0, ay=y-y0;
    const float* r0=m.ptr<float>(y0); const float* r1=m.ptr<float>(y0+1);
    return (1-ay)*((1-ax)*r0[x0]+ax*r0[x0+1])+ay*((1-ax)*r1[x0]+ax*r1[x0+1]);
}

static inline bool bilinearDepth(const cv::Mat& Z,float x,float y,float& z){
    int x0=(int)x, y0=(int)y;
    const float* r0=Z.ptr<float>(y0); const float* r1=Z.ptr<float>(y0+1);
    float a=r0[x0], b=r0[x0+1], c=r1[x0], d=r1[x0+1];
    if(a<=0||b<=0||c<=0||d<=0) return false;
    float mn=std::min(std::min(a,b),std::min(c,d)), mx=std::max(std::max(a,b),std::max(c,d));
    if(mx-mn>0.1f) return false;
    float ax=x-x0, ay=y-y0;
    z=(1-ay)*((1-ax)*a+ax*b)+ay*((1-ax)*c+ax*d);
    return true;
}

// ============================================================
// DENSE RGB-D ALIGNMENT
// Estimates T_cr (reference camera -> current camera) by minimizing
//   photometric:  I_cur(pi(T p)) - I_ref(u)
//   geometric:    Z_cur(pi(T p)) - [T p]_z
// over every reference pixel with valid depth.
// ============================================================
struct DenseResult {
    bool ok=false;
    float inlierRatio=0, validRatio=0;
    int iterations=0;
    Eigen::Matrix<double,6,6> H=Eigen::Matrix<double,6,6>::Zero();   // (omega, v), level 0
};

struct DenseAccum {
    Eigen::Matrix<double,6,6> H=Eigen::Matrix<double,6,6>::Zero();
    Eigen::Matrix<double,6,1> b=Eigen::Matrix<double,6,1>::Zero();
    double cost=0; int valid=0, inliers=0;
    void add(const DenseAccum& o){ H+=o.H; b+=o.b; cost+=o.cost; valid+=o.valid; inliers+=o.inliers; }
};

static inline float huberWeight(float e){ float a=std::abs(e); return a<=DENSE_HUBER?1.f:DENSE_HUBER/a; }
static inline float huberCost(float e){ float a=std::abs(e); return a<=DENSE_HUBER?0.5f*e*e:DENSE_HUBER*(a-0.5f*DENSE_HUBER); }

static DenseAccum denseEvaluate(const std::vector<Eigen::Vector4f>& pts,const DenseLevel& B,
                                const Eigen::Matrix3f& R,const Eigen::Vector3f& t,float sigmaI){
    const int nt=omp_get_max_threads();
    std::vector<DenseAccum> part(nt);
    #pragma omp parallel
    {
        DenseAccum& acc=part[omp_get_thread_num()];
        #pragma omp for schedule(static)
        for(int i=0;i<(int)pts.size();++i){
            const Eigen::Vector3f p=pts[i].head<3>();
            const Eigen::Vector3f q=R*p+t;
            if(q.z()<0.05f) continue;
            const float inv=1.f/q.z();
            const float x=B.fx*q.x()*inv+B.cx, y=B.fy*q.y()*inv+B.cy;
            if(x<1.f||y<1.f||x>B.w-2.f||y>B.h-2.f) continue;
            float Zc=0; const bool hasZ=bilinearDepth(B.Z,x,y,Zc);
            const float rZ=hasZ?Zc-q.z():0.f;
            if(hasZ&&std::abs(rZ)>DENSE_OCCLUSION) continue;
            ++acc.valid;

            const Eigen::Vector3f dPx(B.fx*inv,0.f,-B.fx*q.x()*inv*inv);
            const Eigen::Vector3f dPy(0.f,B.fy*inv,-B.fy*q.y()*inv*inv);
            bool inlier=true;

            // Photometric term.
            {
                const float eI=(bilinear(B.I,x,y)-pts[i][3])/sigmaI;
                const Eigen::Vector3f g=bilinear(B.Ix,x,y)*dPx+bilinear(B.Iy,x,y)*dPy;
                Eigen::Matrix<double,6,1> J;
                J.head<3>()=q.cross(g).cast<double>()/sigmaI;
                J.tail<3>()=g.cast<double>()/sigmaI;
                const double w=DENSE_PHOTO_WEIGHT*huberWeight(eI);
                acc.H.selfadjointView<Eigen::Upper>().rankUpdate(J,w);
                acc.b+=w*J*eI;
                acc.cost+=DENSE_PHOTO_WEIGHT*huberCost(eI);
                inlier=inlier&&std::abs(eI)<3.f;
            }
            // Geometric term.
            if(hasZ){
                const float sZ=DEPTH_SIGMA_A+DEPTH_SIGMA_B*q.z()*q.z();
                const float eZ=rZ/sZ;
                const Eigen::Vector3f g=bilinear(B.Zx,x,y)*dPx+bilinear(B.Zy,x,y)*dPy-Eigen::Vector3f(0,0,1);
                Eigen::Matrix<double,6,1> J;
                J.head<3>()=q.cross(g).cast<double>()/sZ;
                J.tail<3>()=g.cast<double>()/sZ;
                const double w=DENSE_DEPTH_WEIGHT*huberWeight(eZ);
                acc.H.selfadjointView<Eigen::Upper>().rankUpdate(J,w);
                acc.b+=w*J*eZ;
                acc.cost+=DENSE_DEPTH_WEIGHT*huberCost(eZ);
                inlier=inlier&&std::abs(eZ)<3.f;
            }
            if(inlier) ++acc.inliers;
        }
    }
    DenseAccum total;
    for(const auto& p:part) total.add(p);
    total.H=total.H.selfadjointView<Eigen::Upper>();
    return total;
}

static DenseResult denseAlign(const DenseImage& ref,const DenseImage& cur,Pose& T_cr){
    DenseResult res;
    Eigen::Matrix3f R=T_cr.R.cast<float>();
    Eigen::Vector3f t=T_cr.t.cast<float>();
    for(int L=DENSE_LEVELS-1;L>=0;--L){
        const DenseLevel& A=ref.lv[L];
        const DenseLevel& B=cur.lv[L];
        const int stride=(L==0)?DENSE_STRIDE_L0:1;
        std::vector<Eigen::Vector4f> pts;
        pts.reserve((A.w/stride)*(A.h/stride));
        for(int v=0;v<A.h;v+=stride){
            const float* zr=A.Z.ptr<float>(v); const float* ir=A.I.ptr<float>(v);
            for(int u=0;u<A.w;u+=stride){
                const float z=zr[u];
                if(z<=0) continue;
                pts.emplace_back((u-A.cx)*z/A.fx,(v-A.cy)*z/A.fy,z,ir[u]);
            }
        }
        if((int)pts.size()<DENSE_MIN_POINTS/(1<<(2*L))) return res;

        // Robust photometric scale from the residuals at the initial pose of this level.
        float sigmaI=0.05f;
        {
            std::vector<float> r; r.reserve(pts.size());
            for(const auto& pt:pts){
                Eigen::Vector3f q=R*pt.head<3>()+t;
                if(q.z()<0.05f) continue;
                float x=B.fx*q.x()/q.z()+B.cx, y=B.fy*q.y()/q.z()+B.cy;
                if(x<1.f||y<1.f||x>B.w-2.f||y>B.h-2.f) continue;
                r.push_back(std::abs(bilinear(B.I,x,y)-pt[3]));
            }
            if(r.size()>50){
                auto mid=r.begin()+r.size()/2; std::nth_element(r.begin(),mid,r.end());
                sigmaI=std::max(0.01f,std::min(0.2f,1.4826f*(*mid)));
            }
        }

        double lastCost=std::numeric_limits<double>::max();
        Eigen::Matrix3f lastR=R; Eigen::Vector3f lastT=t;
        DenseAccum lastAcc;
        for(int it=0;it<DENSE_ITERATIONS[L];++it){
            DenseAccum acc=denseEvaluate(pts,B,R,t,sigmaI);
            if(acc.valid<(int)(0.15*pts.size())){ R=lastR; t=lastT; break; }
            const double cost=acc.cost/acc.valid;
            if(cost>lastCost){ R=lastR; t=lastT; break; }
            lastCost=cost; lastR=R; lastT=t; lastAcc=acc;
            ++res.iterations;
            Eigen::Matrix<double,6,6> Hd=acc.H;
            Hd.diagonal().array()+=1e-6*Hd.diagonal().maxCoeff()+1e-9;
            Eigen::Matrix<double,6,1> dx=Hd.ldlt().solve(-acc.b);
            if(!dx.allFinite()) break;
            Eigen::Matrix<float,6,1> dxf=dx.cast<float>();
            applyLeftSE3<float>(R,t,dxf);
            if(dx.norm()<1e-6) break;
        }
        if(L==0){
            res.H=lastAcc.H;
            res.validRatio=(float)lastAcc.valid/pts.size();
            res.inlierRatio=(float)lastAcc.inliers/pts.size();
        }
    }
    T_cr.R=R.cast<double>(); T_cr.t=t.cast<double>();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> es(res.H);
    res.ok=res.inlierRatio>=DENSE_MIN_INLIER_RATIO&&es.eigenvalues()(0)>1e-3&&T_cr.R.allFinite()&&T_cr.t.allFinite();
    return res;
}

// Square-root information of a relative-pose measurement from the
// dense Hessian (inflated + floored), ordered (rotation, translation).
static Eigen::Matrix<double,6,6> denseSqrtInfo(const Eigen::Matrix<double,6,6>& H,double floorScale=1.0){
    Eigen::Matrix<double,6,6> Hr=H; Hr.diagonal().array()+=1e-6;
    Eigen::Matrix<double,6,6> cov=DENSE_COV_INFLATION*Hr.inverse();
    for(int i=0;i<3;++i) cov(i,i)+=std::pow(DENSE_MIN_ROT_SIGMA*floorScale,2);
    for(int i=3;i<6;++i) cov(i,i)+=std::pow(DENSE_MIN_TRANS_SIGMA*floorScale,2);
    Eigen::Matrix<double,6,6> info=cov.inverse();
    info=0.5*(info+info.transpose());
    return Eigen::LLT<Eigen::Matrix<double,6,6>>(info).matrixU();
}

static Eigen::Matrix<double,6,6> diagonalSqrtInfo(double rotSigma,double transSigma){
    Eigen::Matrix<double,6,6> m=Eigen::Matrix<double,6,6>::Zero();
    for(int i=0;i<3;++i) m(i,i)=1.0/rotSigma;
    for(int i=3;i<6;++i) m(i,i)=1.0/transSigma;
    return m;
}

// ============================================================
// CUDA ORB + MATCHER
// ============================================================
class CUDAORB {
public:
    cv::Ptr<cv::cuda::ORB> orb;
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher;
    cv::cuda::Stream stream;
    CUDAORB(){
        orb=cv::cuda::ORB::create(ORB_FEATURES,1.2f,8,31,0,2,cv::ORB::HARRIS_SCORE,31,20);
        matcher=cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    }
    bool compute(const cv::Mat& gray,std::vector<cv::KeyPoint>& kp,cv::Mat& desc){
        kp.clear(); desc.release();
        if(gray.empty()) return false;
        cv::cuda::GpuMat g(gray),gkp,gdesc;
        orb->detectAndComputeAsync(g,cv::cuda::GpuMat(),gkp,gdesc,false,stream);
        stream.waitForCompletion();
        orb->convert(gkp,kp);
        gdesc.download(desc,stream);
        stream.waitForCompletion();
        return !kp.empty()&&!desc.empty();
    }
    // Lowe ratio test + one-to-one filtering.
    std::vector<cv::DMatch> match(const cv::Mat& query,const cv::Mat& train) const {
        std::vector<cv::DMatch> good;
        if(query.rows<2||train.rows<2) return good;
        cv::cuda::GpuMat gq(query),gt(train);
        std::vector<std::vector<cv::DMatch>> knn;
        matcher->knnMatch(gq,gt,knn,2);
        std::unordered_map<int,cv::DMatch> best;
        for(const auto& k:knn){
            if(k.size()<2||!(k[0].distance<LOWE_RATIO*k[1].distance)) continue;
            auto it=best.find(k[0].trainIdx);
            if(it==best.end()||k[0].distance<it->second.distance) best[k[0].trainIdx]=k[0];
        }
        for(const auto& p:best) good.push_back(p.second);
        return good;
    }
};

struct Feature {
    Eigen::Vector3d pc;     // camera coordinates
    float u=0, v=0;         // pixel at depth resolution
    int landmark=-1;
};

static bool filteredDepthPoint(const cv::Mat& depth16,const Intrinsics& K,float uf,float vf,Eigen::Vector3d& p){
    int u=(int)std::lround(uf), v=(int)std::lround(vf);
    if(u<1||v<1||u>=depth16.cols-1||v>=depth16.rows-1) return false;
    float z[9]; int n=0;
    for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx){
        float d=depth16.at<uint16_t>(v+dy,u+dx)*DEPTH_SCALE;
        if(d>=MIN_DEPTH&&d<=MAX_DEPTH) z[n++]=d;
    }
    float c=depth16.at<uint16_t>(v,u)*DEPTH_SCALE;
    if(n<5||c<MIN_DEPTH||c>MAX_DEPTH) return false;
    std::sort(z,z+n); float med=z[n/2];
    if(std::abs(c-med)>0.05f) return false;     // depth discontinuity
    p=Eigen::Vector3d((uf-K.cx)*med/K.fx,(vf-K.cy)*med/K.fy,med);
    return true;
}

static void selectFeatures(const std::vector<cv::KeyPoint>& kps,const cv::Mat& desc,const cv::Mat& depth16,const Intrinsics& K,
                           std::vector<Feature>& features,cv::Mat& selDesc){
    struct C { int idx; float resp; Feature f; };
    std::vector<std::vector<C>> cells(GRID_X*GRID_Y);
    const float cw=(float)RGB_WIDTH/GRID_X, ch=(float)RGB_HEIGHT/GRID_Y;
    for(int i=0;i<(int)kps.size();++i){
        const auto& kp=kps[i];
        Feature f; f.u=kp.pt.x*RGB_TO_DEPTH_SCALE; f.v=kp.pt.y*RGB_TO_DEPTH_SCALE;
        if(!filteredDepthPoint(depth16,K,f.u,f.v,f.pc)) continue;
        int gx=std::min(GRID_X-1,(int)(kp.pt.x/cw)), gy=std::min(GRID_Y-1,(int)(kp.pt.y/ch));
        cells[gy*GRID_X+gx].push_back({i,kp.response,f});
    }
    std::vector<C> all;
    for(auto& c:cells){
        int n=std::min((int)c.size(),FEATURES_PER_CELL);
        std::partial_sort(c.begin(),c.begin()+n,c.end(),[](const C&a,const C&b){return a.resp>b.resp;});
        all.insert(all.end(),c.begin(),c.begin()+n);
    }
    std::sort(all.begin(),all.end(),[](const C&a,const C&b){return a.resp>b.resp;});
    if((int)all.size()>MAX_SELECTED_FEATURES) all.resize(MAX_SELECTED_FEATURES);
    features.clear(); selDesc.release();
    if(all.empty()) return;
    selDesc.create((int)all.size(),desc.cols,desc.type());
    for(int i=0;i<(int)all.size();++i){ features.push_back(all[i].f); desc.row(all[i].idx).copyTo(selDesc.row(i)); }
}

// ============================================================
// 3D-3D RANSAC (dst = T * src)
// ============================================================
static bool rigidTransform(const std::vector<Eigen::Vector3d>& src,const std::vector<Eigen::Vector3d>& dst,Pose& T){
    if(src.size()<3) return false;
    Eigen::Vector3d cs=Eigen::Vector3d::Zero(), cd=Eigen::Vector3d::Zero();
    for(size_t i=0;i<src.size();++i){ cs+=src[i]; cd+=dst[i]; }
    cs/=src.size(); cd/=dst.size();
    Eigen::Matrix3d H=Eigen::Matrix3d::Zero();
    for(size_t i=0;i<src.size();++i) H+=(src[i]-cs)*(dst[i]-cd).transpose();
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H,Eigen::ComputeFullU|Eigen::ComputeFullV);
    if(svd.singularValues()(1)<1e-8) return false;
    Eigen::Matrix3d D=Eigen::Matrix3d::Identity();
    if((svd.matrixV()*svd.matrixU().transpose()).determinant()<0) D(2,2)=-1;
    T.R=svd.matrixV()*D*svd.matrixU().transpose(); T.t=cd-T.R*cs;
    return true;
}

static bool ransacRigid(const std::vector<Eigen::Vector3d>& src,const std::vector<Eigen::Vector3d>& dst,Pose& best,std::vector<int>& inliers){
    inliers.clear();
    if(src.size()<(size_t)MIN_RANSAC_INLIERS) return false;
    static thread_local std::mt19937 rng(12345);
    std::uniform_int_distribution<int> pick(0,(int)src.size()-1);
    const double th2=RANSAC_THRESHOLD*RANSAC_THRESHOLD;
    for(int it=0;it<RANSAC_ITERATIONS;++it){
        int a=pick(rng),b=pick(rng),c=pick(rng);
        if(a==b||a==c||b==c) continue;
        if((src[b]-src[a]).cross(src[c]-src[a]).norm()<0.003) continue;
        Pose T;
        if(!rigidTransform({src[a],src[b],src[c]},{dst[a],dst[b],dst[c]},T)) continue;
        std::vector<int> in;
        for(int i=0;i<(int)src.size();++i) if((T.transform(src[i])-dst[i]).squaredNorm()<th2) in.push_back(i);
        if(in.size()>inliers.size()){ inliers=std::move(in); best=T; }
    }
    if((int)inliers.size()<MIN_RANSAC_INLIERS) return false;
    for(int round=0;round<2;++round){
        std::vector<Eigen::Vector3d> s,d;
        for(int i:inliers){ s.push_back(src[i]); d.push_back(dst[i]); }
        if(!rigidTransform(s,d,best)) return false;
        inliers.clear();
        for(int i=0;i<(int)src.size();++i) if((best.transform(src[i])-dst[i]).squaredNorm()<th2) inliers.push_back(i);
    }
    return (int)inliers.size()>=MIN_RANSAC_INLIERS;
}

// ============================================================
// CERES FACTORS
// Quaternion parameter blocks use Eigen storage order (x, y, z, w).
// ============================================================
template<typename T>
static Eigen::Quaternion<T> quatExp(const Eigen::Matrix<T,3,1>& w){
    T aa[3]={w[0],w[1],w[2]}, q[4];
    ceres::AngleAxisToQuaternion(aa,q);
    return Eigen::Quaternion<T>(q[0],q[1],q[2],q[3]);
}
template<typename T>
static Eigen::Matrix<T,3,1> quatLog(const Eigen::Quaternion<T>& q){
    T qq[4]={q.w(),q.x(),q.y(),q.z()}, aa[3];
    ceres::QuaternionToAngleAxis(qq,aa);
    return Eigen::Matrix<T,3,1>(aa[0],aa[1],aa[2]);
}

// Relative pose T_ij = T_i^-1 T_j; residual (rotation, translation) is a
// right perturbation of the measured relative pose.
struct RelPoseFactor {
    Eigen::Quaterniond qm; Eigen::Vector3d tm; Eigen::Matrix<double,6,6> sqrtInfo;
    RelPoseFactor(const Pose& m,const Eigen::Matrix<double,6,6>& s):qm(m.R),tm(m.t),sqrtInfo(s){}
    template<typename T>
    bool operator()(const T* qi_,const T* pi_,const T* qj_,const T* pj_,T* res) const {
        Eigen::Map<const Eigen::Quaternion<T>> qi(qi_), qj(qj_);
        Eigen::Map<const Eigen::Matrix<T,3,1>> pi(pi_), pj(pj_);
        const Eigen::Quaternion<T> qmT=qm.cast<T>();
        const Eigen::Quaternion<T> qij=qi.conjugate()*qj;
        const Eigen::Matrix<T,3,1> tij=qi.conjugate()*(pj-pi);
        Eigen::Matrix<T,6,1> r;
        r.template head<3>()=quatLog<T>(qmT.conjugate()*qij);
        r.template tail<3>()=qmT.conjugate()*(tij-tm.cast<T>());
        Eigen::Map<Eigen::Matrix<T,6,1>> out(res);
        out=sqrtInfo.cast<T>()*r;
        return true;
    }
    static ceres::CostFunction* create(const Pose& m,const Eigen::Matrix<double,6,6>& s){
        return new ceres::AutoDiffCostFunction<RelPoseFactor,6,4,3,4,3>(new RelPoseFactor(m,s));
    }
};

struct ImuFactor {
    const Preintegration* pre; Eigen::Vector3d g; Eigen::Matrix<double,15,15> sqrtInfo;
    ImuFactor(const Preintegration* p,const Eigen::Vector3d& gw):pre(p),g(gw){
        Eigen::Matrix<double,15,15> cov=Eigen::Matrix<double,15,15>::Zero();
        cov.block<9,9>(0,0)=p->cov;
        cov.block<3,3>(9,9)=Eigen::Matrix3d::Identity()*GYRO_BIAS_RW*GYRO_BIAS_RW*std::max(p->dt,1e-3);
        cov.block<3,3>(12,12)=Eigen::Matrix3d::Identity()*ACC_BIAS_RW*ACC_BIAS_RW*std::max(p->dt,1e-3);
        cov.diagonal().array()+=1e-10;
        Eigen::Matrix<double,15,15> info=cov.inverse();
        info=0.5*(info+info.transpose());
        sqrtInfo=Eigen::LLT<Eigen::Matrix<double,15,15>>(info).matrixU();
    }
    template<typename T>
    bool operator()(const T* qi_,const T* pi_,const T* vi_,const T* bgi_,const T* bai_,
                    const T* qj_,const T* pj_,const T* vj_,const T* bgj_,const T* baj_,T* res) const {
        using V3=Eigen::Matrix<T,3,1>;
        Eigen::Map<const Eigen::Quaternion<T>> qi(qi_), qj(qj_);
        Eigen::Map<const V3> pi(pi_), vi(vi_), bgi(bgi_), bai(bai_), pj(pj_), vj(vj_), bgj(bgj_), baj(baj_);
        const V3 dbg=bgi-pre->bg.cast<T>(), dba=bai-pre->ba.cast<T>();
        const Eigen::Quaternion<T> dq=Eigen::Quaterniond(pre->dR).cast<T>()*quatExp<T>(V3(pre->JR_bg.cast<T>()*dbg));
        const V3 dv=pre->dV.cast<T>()+pre->JV_bg.cast<T>()*dbg+pre->JV_ba.cast<T>()*dba;
        const V3 dp=pre->dP.cast<T>()+pre->JP_bg.cast<T>()*dbg+pre->JP_ba.cast<T>()*dba;
        const T dt=T(pre->dt);
        const V3 gT=g.cast<T>();
        Eigen::Matrix<T,15,1> r;
        r.template segment<3>(0)=quatLog<T>(dq.conjugate()*qi.conjugate()*qj);
        r.template segment<3>(3)=qi.conjugate()*(vj-vi-gT*dt)-dv;
        r.template segment<3>(6)=qi.conjugate()*(pj-pi-vi*dt-T(0.5)*gT*dt*dt)-dp;
        r.template segment<3>(9)=bgj-bgi;
        r.template segment<3>(12)=baj-bai;
        Eigen::Map<Eigen::Matrix<T,15,1>> out(res);
        out=sqrtInfo.cast<T>()*r;
        return true;
    }
    static ceres::CostFunction* create(const Preintegration* p,const Eigen::Vector3d& g){
        return new ceres::AutoDiffCostFunction<ImuFactor,15,4,3,3,3,3,4,3,3,3,3>(new ImuFactor(p,g));
    }
};

// RGB-D bundle adjustment: pixel reprojection + measured depth.
struct ReprojDepthFactor {
    double u,v,z,fx,fy,cx,cy,sz;
    ReprojDepthFactor(const Feature& f,const Intrinsics& K)
        :u(f.u),v(f.v),z(f.pc.z()),fx(K.fx),fy(K.fy),cx(K.cx),cy(K.cy),sz(DEPTH_SIGMA_A+DEPTH_SIGMA_B*f.pc.z()*f.pc.z()){}
    template<typename T>
    bool operator()(const T* q_,const T* p_,const T* X_,T* res) const {
        Eigen::Map<const Eigen::Quaternion<T>> q(q_);
        Eigen::Map<const Eigen::Matrix<T,3,1>> p(p_), X(X_);
        Eigen::Matrix<T,3,1> xc=q.conjugate()*(X-p);
        T zc=xc.z();
        if(zc<T(0.05)) zc=T(0.05);
        res[0]=(T(fx)*xc.x()/zc+T(cx)-T(u))/T(REPROJ_SIGMA_PX);
        res[1]=(T(fy)*xc.y()/zc+T(cy)-T(v))/T(REPROJ_SIGMA_PX);
        res[2]=(xc.z()-T(z))/T(sz);
        return true;
    }
    static ceres::CostFunction* create(const Feature& f,const Intrinsics& K){
        return new ceres::AutoDiffCostFunction<ReprojDepthFactor,3,4,3,3>(new ReprojDepthFactor(f,K));
    }
};

struct VecPrior {
    Eigen::Vector3d m; double s;
    VecPrior(const Eigen::Vector3d& mm,double ss):m(mm),s(ss){}
    template<typename T> bool operator()(const T* x,T* r) const {
        for(int i=0;i<3;++i) r[i]=(x[i]-T(m[i]))/T(s);
        return true;
    }
    static ceres::CostFunction* create(const Eigen::Vector3d& m,double s){
        return new ceres::AutoDiffCostFunction<VecPrior,3,3>(new VecPrior(m,s));
    }
};

// ============================================================
// MAP DATA
// ============================================================
struct KeyFrame {
    int id=-1, frameIndex=-1;
    double t=0;
    Intrinsics K;
    double q[4]={0,0,0,1}, p[3]={0,0,0}, v[3]={0,0,0}, bg[3]={0,0,0}, ba[3]={0,0,0};
    bool imuActive=false;            // velocity/bias states are estimated
    Pose odometryPose;               // chained front-end measurements only (no window/loop/BA corrections)
    cv::Mat gray8, depth16;          // for dense loop verification
    std::vector<Feature> features;
    cv::Mat desc;

    Pose pose() const {
        Pose T; T.R=Eigen::Quaterniond(q[3],q[0],q[1],q[2]).normalized().toRotationMatrix();
        T.t=Eigen::Vector3d(p[0],p[1],p[2]); return T;
    }
    void setPose(const Pose& T){
        Eigen::Quaterniond qq(T.R); qq.normalize();
        q[0]=qq.x(); q[1]=qq.y(); q[2]=qq.z(); q[3]=qq.w();
        p[0]=T.t.x(); p[1]=T.t.y(); p[2]=T.t.z();
    }
    Eigen::Vector3d vel() const { return {v[0],v[1],v[2]}; }
    Eigen::Vector3d gyroBias() const { return {bg[0],bg[1],bg[2]}; }
    Eigen::Vector3d accBias() const { return {ba[0],ba[1],ba[2]}; }
    void setVel(const Eigen::Vector3d& x){ v[0]=x.x(); v[1]=x.y(); v[2]=x.z(); }
    void setGyroBias(const Eigen::Vector3d& x){ bg[0]=x.x(); bg[1]=x.y(); bg[2]=x.z(); }
    void setAccBias(const Eigen::Vector3d& x){ ba[0]=x.x(); ba[1]=x.y(); ba[2]=x.z(); }
};

struct Landmark {
    double X[3]={0,0,0};
    std::vector<std::pair<int,int>> obs;   // (keyframe id, feature index)
    bool hasObsIn(int kf) const { for(const auto& o:obs) if(o.first==kf) return true; return false; }
};

enum EdgeType { EDGE_SEQUENTIAL=0, EDGE_LOOP=1, EDGE_REVISIT=2, EDGE_LOCAL=3 };   // LOCAL = added by refinement
struct VisualEdge { int from=-1, to=-1; Pose meas; Eigen::Matrix<double,6,6> sqrtInfo; EdgeType type=EDGE_SEQUENTIAL; };

// status: 0 none, 1 dense-tracked, 2 IMU-only prediction
struct FrameRecord {
    int status=0, refKF=-1;
    int checkKF=-1;                 // keyframe this pose is checked against (-1 = next keyframe)
    Pose T_kf_c;
    bool isKeyframe=false, loop=false, sparseInit=false;
    float inlierRatio=0;
    int iterations=0;
    double ms=0;
    Pose finalPose;
};

// Everything the tracker keeps about one processed frame.
struct FrameData {
    int index=-1; double t=0; Intrinsics K;
    cv::Mat gray8, depth16;
    DenseImage dense;
    std::vector<Feature> features; cv::Mat desc;
};

// ============================================================
// VISUAL-INERTIAL SLAM
// ============================================================
class VioSlam {
public:
    VioSlam(const PhoneData& pd,bool useImu):phone(pd),useImu(useImu&&pd.hasImu()){}

    const PhoneData& phone;
    const bool useImu;
    CUDAORB orb;
    std::vector<KeyFrame> kfs;
    std::vector<Landmark> lms;
    std::vector<VisualEdge> edges;
    std::vector<std::unique_ptr<Preintegration>> imuEdge;   // imuEdge[k]: kf k-1 -> k
    std::vector<FrameRecord> frames;
    Eigen::Vector3d g=Eigen::Vector3d(0,0,-GRAVITY);
    bool imuInitialized=false;
    int loops=0, revisits=0, localEdges=0, pgoRuns=0, imuOnlyFrames=0, sparseInits=0, promotions=0;

    void process(FrameData&& f){
        auto start=std::chrono::steady_clock::now();
        if((int)frames.size()<=f.index) frames.resize(f.index+1);
        FrameRecord& rec=frames[f.index];

        if(kfs.empty()){
            createKeyframe(f,Pose(),nullptr,nullptr);
            rec.status=1; rec.refKF=0; rec.isKeyframe=true;
            activeDense=std::move(f.dense);
            lastGood=FrameData(); lastGoodValid=false;
            prevFramePose=Pose(); prevFrameTime=f.t; havePrevFrame=true;
            return;
        }

        const Pose predicted=predictPose(f.t);
        Pose T_cr; DenseResult dr;
        bool ok=trackAgainstActive(f,predicted,T_cr,dr,rec);

        // Fallback: promote the last good frame to a keyframe and retry.
        if(!ok&&lastGoodValid&&lastGood.index!=kfs.back().frameIndex){
            promoteLastGood();
            ++promotions;
            ok=trackAgainstActive(f,predicted,T_cr,dr,rec);
        }

        const KeyFrame& act=kfs.back();
        Pose T_w_c;
        if(ok){
            T_w_c=act.pose()*T_cr.inverse();
            rec.status=1; consecutiveFailures=0;
        }else{
            T_w_c=predicted;
            rec.status=2; ++consecutiveFailures; ++imuOnlyFrames;
        }
        rec.refKF=act.id;
        rec.T_kf_c=act.pose().inverse()*T_w_c;
        rec.inlierRatio=dr.inlierRatio;
        rec.iterations=dr.iterations;

        // ---------------- keyframe decision ----------------
        const double frameTime=f.t;
        bool newKF=false;
        if(ok){
            const Pose rel=rec.T_kf_c;
            newKF=rel.t.norm()>KF_TRANSLATION||rotationDeg(Eigen::Matrix3d::Identity(),rel.R)>KF_ROTATION_DEG||
                  dr.inlierRatio<KF_MIN_INLIER_RATIO||f.t-act.t>KF_MAX_DT;
            if(newKF){
                VisualEdge e; e.from=act.id; e.meas=rec.T_kf_c; e.sqrtInfo=denseSqrtInfo(dr.H);
                createKeyframe(f,T_w_c,&e,&rec);
            }
        }else if(consecutiveFailures>=FORCE_KF_AFTER_FAILURES){
            // Re-anchor tracking on an IMU-predicted keyframe; the IMU factor
            // (and later loops) connect it to the map.
            createKeyframe(f,T_w_c,nullptr,&rec);
            consecutiveFailures=0; newKF=true;
        }
        if(newKF){
            rec.isKeyframe=true; rec.refKF=kfs.back().id; rec.T_kf_c=Pose();
            activeDense=std::move(f.dense);
            lastGoodValid=false;
        }else if(ok){
            lastGood=std::move(f); lastGoodRel=rec.T_kf_c; lastGoodH=dr.H; lastGoodValid=true;
        }
        prevPrevFramePose=prevFramePose; prevPrevFrameTime=prevFrameTime;
        prevFramePose=T_w_c; prevFrameTime=frameTime;
        rec.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    }

    void finish(){
        addRevisitConstraints();
        std::cout<<"\n[FINAL] Global bundle adjustment over "<<kfs.size()<<" keyframes, "
                 <<lms.size()<<" landmarks, "<<edges.size()<<" visual edges...\n";
        globalBundleAdjustment();
        updateFinalPoses();
    }

    // Used by the automatic refinement after it added constraints.
    void reoptimize(){
        poseGraphOptimization();
        globalBundleAdjustment();
        updateFinalPoses();
    }

    void updateFinalPoses(){
        for(auto& r:frames)
            if(r.refKF>=0) r.finalPose=kfs[r.refKF].pose()*r.T_kf_c;
    }

    Pose framePose(const FrameRecord& r) const { return kfs[r.refKF].pose()*r.T_kf_c; }

private:
    DenseImage activeDense;
    FrameData lastGood; bool lastGoodValid=false; Pose lastGoodRel; Eigen::Matrix<double,6,6> lastGoodH;
    Pose prevFramePose, prevPrevFramePose; double prevFrameTime=0, prevPrevFrameTime=0; bool havePrevFrame=false;
    int consecutiveFailures=0, lastLoopKF=-1000;

    // ---------------- prediction ----------------
    Pose predictPose(double t) const {
        const KeyFrame& k=kfs.back();
        if(imuInitialized&&k.imuActive){
            Preintegration pre; pre.integrate(phone,k.t,t,k.gyroBias(),k.accBias());
            Pose T; Eigen::Vector3d v;
            pre.predict(k.pose().R,k.pose().t,k.vel(),g,T.R,T.t,v);
            return T;
        }
        // Before IMU initialization: gyro rotation + constant velocity.
        Pose T=prevFramePose;
        if(useImu){
            Preintegration pre; pre.integrate(phone,prevFrameTime,t,k.gyroBias(),Eigen::Vector3d::Zero());
            T.R=prevFramePose.R*pre.dR;
        }
        if(prevFrameTime-prevPrevFrameTime>1e-3){
            Eigen::Vector3d vel=(prevFramePose.t-prevPrevFramePose.t)/(prevFrameTime-prevPrevFrameTime);
            if(vel.norm()>1.5) vel*=1.5/vel.norm();
            T.t=prevFramePose.t+0.9*vel*(t-prevFrameTime);
        }
        return T;
    }

    // ---------------- dense tracking against the active keyframe ----------------
    bool trackAgainstActive(FrameData& f,const Pose& predicted,Pose& T_cr,DenseResult& dr,FrameRecord& rec){
        const KeyFrame& act=kfs.back();
        const Pose T_w_kf=act.pose();
        const double predRot=rotationDeg(T_w_kf.R,predicted.R);
        auto consistent=[&](const Pose& Tcr){
            const Pose T_w_c=T_w_kf*Tcr.inverse();
            const double gate=GATE_ROT_BASE_DEG+GATE_ROT_SCALE*predRot;
            return (!useImu||rotationDeg(T_w_c.R,predicted.R)<=gate)&&(T_w_c.t-predicted.t).norm()<=GATE_MAX_TRANS;
        };

        T_cr=(T_w_kf.inverse()*predicted).inverse();
        dr=denseAlign(activeDense,f.dense,T_cr);
        if(dr.ok&&consistent(T_cr)) return true;

        // Sparse re-initialization: ORB + 3D-3D RANSAC, then dense again.
        std::vector<Eigen::Vector3d> src,dst;
        for(const auto& m:orb.match(act.desc,f.desc)){
            src.push_back(act.features[m.queryIdx].pc);
            dst.push_back(f.features[m.trainIdx].pc);
        }
        Pose Tr; std::vector<int> in;
        if(ransacRigid(src,dst,Tr,in)){
            Pose T2=Tr;
            DenseResult d2=denseAlign(activeDense,f.dense,T2);
            if(d2.ok&&consistent(T2)){ T_cr=T2; dr=d2; rec.sparseInit=true; ++sparseInits; return true; }
        }
        return false;
    }

    void promoteLastGood(){
        VisualEdge e; e.from=kfs.back().id; e.meas=lastGoodRel; e.sqrtInfo=denseSqrtInfo(lastGoodH);
        const Pose T_w=kfs.back().pose()*lastGoodRel;
        FrameRecord& r=frames[lastGood.index];
        createKeyframe(lastGood,T_w,&e,&r);
        r.isKeyframe=true; r.refKF=kfs.back().id; r.T_kf_c=Pose();
        activeDense=std::move(lastGood.dense);
        lastGoodValid=false;
    }

    // ---------------- keyframe creation ----------------
    void createKeyframe(const FrameData& f,const Pose& T_w_c,VisualEdge* edge,FrameRecord* rec){
        KeyFrame k;
        k.id=(int)kfs.size(); k.frameIndex=f.index; k.t=f.t; k.K=f.K;
        k.setPose(T_w_c);
        k.gray8=f.gray8; k.depth16=f.depth16;
        k.features=f.features; k.desc=f.desc.clone();
        for(auto& ft:k.features) ft.landmark=-1;
        std::unique_ptr<Preintegration> pre;
        if(!kfs.empty()){
            const KeyFrame& prev=kfs.back();
            k.setGyroBias(prev.gyroBias()); k.setAccBias(prev.accBias()); k.setVel(prev.vel());
            if(useImu){
                pre.reset(new Preintegration());
                pre->integrate(phone,prev.t,k.t,prev.gyroBias(),prev.accBias());
                if(imuInitialized&&prev.imuActive){
                    Pose Tp; Eigen::Vector3d vj;
                    pre->predict(prev.pose().R,prev.pose().t,prev.vel(),g,Tp.R,Tp.t,vj);
                    k.setVel(vj); k.imuActive=true;
                }
            }
        }
        // T_w_c = active.pose() * T_kf_c, so this chains the raw relative measurement.
        k.odometryPose=kfs.empty()?T_w_c:kfs.back().odometryPose*(kfs.back().pose().inverse()*T_w_c);
        kfs.push_back(std::move(k));
        imuEdge.push_back(std::move(pre));
        if(edge){ edge->to=kfs.back().id; edges.push_back(*edge); }

        associateLandmarks(kfs.back().id);

        if(useImu&&!imuInitialized) tryInitializeImu();
        optimizeWindow();
        if(detectLoop(kfs.back().id)&&rec) rec->loop=true;
    }

    // ---------------- landmarks ----------------
    void associateLandmarks(int n){
        KeyFrame& kn=kfs[n];
        const Pose Tn=kn.pose();
        for(int k=n-1;k>=std::max(0,n-LANDMARK_MATCH_KEYFRAMES);--k){
            KeyFrame& kk=kfs[k];
            const Pose Tk=kk.pose();
            for(const auto& m:orb.match(kn.desc,kk.desc)){
                Feature& fn=kn.features[m.queryIdx];
                Feature& fk=kk.features[m.trainIdx];
                if((Tn.transform(fn.pc)-Tk.transform(fk.pc)).norm()>LANDMARK_MAX_DIST) continue;
                linkFeatures(n,m.queryIdx,k,m.trainIdx);
            }
        }
    }

    void linkFeatures(int a,int fa,int b,int fb){
        Feature& A=kfs[a].features[fa];
        Feature& B=kfs[b].features[fb];
        if(A.landmark>=0&&B.landmark>=0){
            if(A.landmark!=B.landmark) mergeLandmarks(A.landmark,B.landmark);
            return;
        }
        if(B.landmark>=0){
            Landmark& L=lms[B.landmark];
            if(L.hasObsIn(a)) return;
            L.obs.push_back({a,fa}); A.landmark=B.landmark; return;
        }
        if(A.landmark>=0){
            Landmark& L=lms[A.landmark];
            if(L.hasObsIn(b)) return;
            L.obs.push_back({b,fb}); B.landmark=A.landmark; return;
        }
        Landmark L;
        Eigen::Vector3d X=0.5*(kfs[a].pose().transform(A.pc)+kfs[b].pose().transform(B.pc));
        L.X[0]=X.x(); L.X[1]=X.y(); L.X[2]=X.z();
        L.obs={{b,fb},{a,fa}};
        A.landmark=B.landmark=(int)lms.size();
        lms.push_back(L);
    }

    void mergeLandmarks(int keep,int drop){
        Landmark& K=lms[keep]; Landmark& D=lms[drop];
        for(const auto& o:D.obs){
            if(K.hasObsIn(o.first)){ kfs[o.first].features[o.second].landmark=-1; continue; }
            K.obs.push_back(o); kfs[o.first].features[o.second].landmark=keep;
        }
        D.obs.clear();
    }

    // ---------------- IMU initialization ----------------
    void tryInitializeImu(){
        const int n=(int)kfs.size();
        if(n<IMU_INIT_KEYFRAMES||kfs.back().t-kfs.front().t<IMU_INIT_MIN_SPAN) return;

        // 1) Gyro bias from visual vs. preintegrated rotations.
        Eigen::Matrix3d A=Eigen::Matrix3d::Zero(); Eigen::Vector3d b=Eigen::Vector3d::Zero();
        for(int k=1;k<n;++k){
            const Preintegration& pre=*imuEdge[k];
            const Eigen::Matrix3d Rij=kfs[k-1].pose().R.transpose()*kfs[k].pose().R;
            const Eigen::Vector3d e=logSO3(pre.dR.transpose()*Rij);
            A+=pre.JR_bg.transpose()*pre.JR_bg; b+=pre.JR_bg.transpose()*e;
        }
        const Eigen::Vector3d bg=A.ldlt().solve(b);
        for(auto& k:kfs) k.setGyroBias(bg);
        for(int k=1;k<n;++k) imuEdge[k]->integrate(phone,kfs[k-1].t,kfs[k].t,bg,Eigen::Vector3d::Zero());

        // 2) Gravity direction: mean specific force rotated into the world.
        Eigen::Vector3d fsum=Eigen::Vector3d::Zero(); int cnt=0;
        for(int k=1;k<n;++k){
            const Pose Ta=kfs[k-1].pose(), Tb=kfs[k].pose();
            for(size_t s=phone.sampleAt(kfs[k-1].t);s<phone.imu.size()&&phone.imu[s].t<kfs[k].t;++s){
                double a=(phone.imu[s].t-kfs[k-1].t)/std::max(1e-6,kfs[k].t-kfs[k-1].t);
                a=std::max(0.0,std::min(1.0,a));
                fsum+=interpolatePose(Ta,Tb,a).R*phone.imu[s].acc; ++cnt;
            }
        }
        if(cnt<50) return;
        const Eigen::Vector3d gdir=-(fsum/cnt).normalized();
        std::cout<<"[IMU-INIT] gyro bias "<<bg.transpose()<<"  |mean specific force| "<<(fsum/cnt).norm()<<" m/s^2\n";

        // 3) Rotate the world so gravity is (0,0,-g).
        const Eigen::Matrix3d Ralign=Eigen::Quaterniond::FromTwoVectors(gdir,Eigen::Vector3d(0,0,-1)).toRotationMatrix();
        Pose C; C.R=Ralign;
        applyWorldTransform(C);
        g=Eigen::Vector3d(0,0,-GRAVITY);

        // 4) Velocities by linear least squares with known gravity.
        Eigen::MatrixXd M=Eigen::MatrixXd::Zero(6*(n-1),3*n);
        Eigen::VectorXd r=Eigen::VectorXd::Zero(6*(n-1));
        for(int k=1;k<n;++k){
            const Preintegration& pre=*imuEdge[k];
            const Eigen::Matrix3d RiT=kfs[k-1].pose().R.transpose();
            const double dt=pre.dt;
            const int row=6*(k-1);
            M.block<3,3>(row,3*k)=RiT; M.block<3,3>(row,3*(k-1))=-RiT;
            r.segment<3>(row)=pre.dV+RiT*g*dt;
            M.block<3,3>(row+3,3*(k-1))=-RiT*dt;
            r.segment<3>(row+3)=pre.dP-RiT*(kfs[k].pose().t-kfs[k-1].pose().t)+0.5*RiT*g*dt*dt;
        }
        Eigen::VectorXd vs=M.colPivHouseholderQr().solve(r);
        for(int k=0;k<n;++k){ kfs[k].setVel(vs.segment<3>(3*k)); kfs[k].setAccBias(Eigen::Vector3d::Zero()); kfs[k].imuActive=true; }
        imuInitialized=true;
        std::cout<<"[IMU-INIT] initialized over "<<n<<" keyframes ("<<std::setprecision(2)<<kfs.back().t-kfs.front().t
                 <<" s), |v| last "<<kfs.back().vel().norm()<<" m/s\n";
        optimizeRange(0,n-1,true);
    }

    // T_new = C * T_old for every world-frame quantity.
    void applyWorldTransform(const Pose& C){
        for(auto& k:kfs){ k.setPose(C*k.pose()); k.setVel(C.R*k.vel()); k.odometryPose=C*k.odometryPose; }
        for(auto& l:lms){ Eigen::Vector3d X(l.X[0],l.X[1],l.X[2]); X=C.transform(X); l.X[0]=X.x(); l.X[1]=X.y(); l.X[2]=X.z(); }
        prevFramePose=C*prevFramePose; prevPrevFramePose=C*prevPrevFramePose;
    }

    void reintegrateIfNeeded(int k){
        if(k<=0||!imuEdge[k]) return;
        Preintegration& pre=*imuEdge[k];
        const KeyFrame& a=kfs[k-1];
        if((a.gyroBias()-pre.bg).norm()>REINTEGRATE_BG||(a.accBias()-pre.ba).norm()>REINTEGRATE_BA)
            pre.integrate(phone,a.t,kfs[k].t,a.gyroBias(),a.accBias());
    }

    // ---------------- optimization ----------------
    void optimizeWindow(){
        const int n=(int)kfs.size();
        if(n<2) return;
        optimizeRange(std::max(0,n-WINDOW_KEYFRAMES),n-1,false);
    }

    // Visual-inertial bundle adjustment over keyframes [first,last].
    // Keyframes outside the range that observe the same landmarks or are
    // connected by edges enter as constants (anchoring the window).
    void optimizeRange(int first,int last,bool global){
        ceres::Problem problem;
        auto* quatParam=new ceres::EigenQuaternionParameterization();
        std::unordered_set<int> added;
        auto addPose=[&](int id,bool constant){
            KeyFrame& k=kfs[id];
            if(added.insert(id).second){
                problem.AddParameterBlock(k.q,4,quatParam);
                problem.AddParameterBlock(k.p,3);
            }
            if(constant){ problem.SetParameterBlockConstant(k.q); problem.SetParameterBlockConstant(k.p); }
        };
        auto inRange=[&](int id){ return id>=first&&id<=last; };
        for(int id=first;id<=last;++id) addPose(id,false);
        addPose(first,true);   // gauge

        // Dense relative-pose factors.
        for(const auto& e:edges){
            const bool a=inRange(e.from), b=inRange(e.to);
            if(!a&&!b) continue;
            if(!a) addPose(e.from,true);
            if(!b) addPose(e.to,true);
            // Sequential edges passed dense alignment + the gyro gate; a robust
            // loss would let them give way to loop edges and bend the path.
            ceres::LossFunction* loss=(e.type==EDGE_SEQUENTIAL)?nullptr:new ceres::CauchyLoss(LOOP_CAUCHY_SCALE);
            problem.AddResidualBlock(RelPoseFactor::create(e.meas,e.sqrtInfo),loss,kfs[e.from].q,kfs[e.from].p,kfs[e.to].q,kfs[e.to].p);
        }

        // IMU factors between consecutive keyframes.
        if(imuInitialized){
            for(int k=std::max(first+1,1);k<=last;++k){
                KeyFrame& a=kfs[k-1]; KeyFrame& b=kfs[k];
                if(!imuEdge[k]||!a.imuActive||!b.imuActive||imuEdge[k]->dt<=0) continue;
                reintegrateIfNeeded(k);
                problem.AddResidualBlock(ImuFactor::create(imuEdge[k].get(),g),nullptr,
                                         a.q,a.p,a.v,a.bg,a.ba,b.q,b.p,b.v,b.bg,b.ba);
            }
            KeyFrame& f0=kfs[first];
            if(f0.imuActive&&!global){
                problem.AddResidualBlock(VecPrior::create(f0.vel(),PRIOR_V_SIGMA),nullptr,f0.v);
                problem.AddResidualBlock(VecPrior::create(f0.gyroBias(),PRIOR_BG_SIGMA),nullptr,f0.bg);
                problem.AddResidualBlock(VecPrior::create(f0.accBias(),PRIOR_BA_SIGMA),nullptr,f0.ba);
            }
            if(f0.imuActive&&global) problem.AddResidualBlock(VecPrior::create(Eigen::Vector3d::Zero(),1.0),nullptr,f0.ba);
        }

        // Landmark factors (RGB-D bundle adjustment).
        std::unordered_set<int> lmIds;
        for(int id=first;id<=last;++id)
            for(const auto& f:kfs[id].features)
                if(f.landmark>=0&&lms[f.landmark].obs.size()>=2) lmIds.insert(f.landmark);
        // In the global BA the ~40k short-track observations act like an
        // overconfident set of local constraints and pull the map back toward
        // the drifted shape (measured by the verification stage), so they are
        // down-weighted there.
        const double lmScale=global?FINAL_BA_LANDMARK_SIGMA_SCALE:1.0;
        auto lmLoss=[&]()->ceres::LossFunction*{ return new ceres::ScaledLoss(new ceres::HuberLoss(2.0*lmScale),1.0/(lmScale*lmScale),ceres::TAKE_OWNERSHIP); };
        for(int l:lmIds){
            Landmark& L=lms[l];
            for(const auto& o:L.obs){
                if(!inRange(o.first)) addPose(o.first,true);
                const KeyFrame& k=kfs[o.first];
                problem.AddResidualBlock(ReprojDepthFactor::create(k.features[o.second],k.K),lmLoss(),
                                         kfs[o.first].q,kfs[o.first].p,L.X);
            }
        }

        ceres::Solver::Options opt;
        opt.linear_solver_type=ceres::SPARSE_SCHUR;
        opt.max_num_iterations=global?FINAL_BA_ITERATIONS:WINDOW_ITERATIONS;
        opt.num_threads=std::max(1,omp_get_max_threads());
        opt.minimizer_progress_to_stdout=global;
        ceres::Solver::Summary sum;
        ceres::Solve(opt,&problem,&sum);
        if(global) std::cout<<sum.BriefReport()<<"\n";
        for(auto& k:kfs){ Eigen::Map<Eigen::Quaterniond> q(k.q); q.normalize(); }
    }

    // ---------------- loop closure ----------------
    bool detectLoop(int n){
        if(n-lastLoopKF<LOOP_COOLDOWN_KEYFRAMES) return false;
        const KeyFrame& kn=kfs[n];
        const Pose Tn=kn.pose();
        std::vector<std::pair<double,int>> cands;
        for(int k=0;k<n;++k){
            const KeyFrame& kk=kfs[k];
            if(kn.frameIndex-kk.frameIndex<LOOP_MIN_FRAME_GAP) continue;
            const Pose Tk=kk.pose();
            const double d=(Tk.t-Tn.t).norm();
            if(d>LOOP_SEARCH_RADIUS) continue;
            const double ang=std::acos(std::max(-1.0,std::min(1.0,(Tk.R.col(2)).dot(Tn.R.col(2)))))*180/M_PI;
            if(ang>LOOP_MAX_VIEW_ANGLE_DEG) continue;
            cands.push_back({d+0.01*ang,k});
        }
        if(cands.empty()) return false;
        std::sort(cands.begin(),cands.end());
        if((int)cands.size()>LOOP_MAX_CANDIDATES) cands.resize(LOOP_MAX_CANDIDATES);

        // Rank by appearance.
        struct Cand { int k; std::vector<cv::DMatch> m; };
        std::vector<Cand> ranked;
        for(const auto& c:cands){
            auto m=orb.match(kn.desc,kfs[c.second].desc);
            if((int)m.size()>=LOOP_MIN_MATCHES) ranked.push_back({c.second,std::move(m)});
        }
        std::sort(ranked.begin(),ranked.end(),[](const Cand&a,const Cand&b){return a.m.size()>b.m.size();});

        DenseImage curDense;
        for(size_t ci=0;ci<std::min<size_t>(3,ranked.size());++ci){
            const KeyFrame& kk=kfs[ranked[ci].k];
            std::vector<Eigen::Vector3d> src,dst;
            for(const auto& m:ranked[ci].m){ src.push_back(kk.features[m.trainIdx].pc); dst.push_back(kn.features[m.queryIdx].pc); }
            Pose Tr; std::vector<int> in;
            if(!ransacRigid(src,dst,Tr,in)||(int)in.size()<LOOP_MIN_INLIERS) continue;

            // Dense verification and refinement (reference = old keyframe).
            if(curDense.lv.empty()) curDense=buildDenseImage(kn.gray8,kn.depth16,kn.K);
            DenseImage oldDense=buildDenseImage(kk.gray8,kk.depth16,kk.K);
            Pose Td=Tr;
            DenseResult dr=denseAlign(oldDense,curDense,Td);
            if(!dr.ok||dr.inlierRatio<LOOP_MIN_DENSE_INLIER_RATIO) continue;
            if((Td.t-Tr.t).norm()>0.05||rotationDeg(Td.R,Tr.R)>3.0) continue;

            VisualEdge e; e.from=kk.id; e.to=kn.id; e.meas=Td.inverse(); e.type=EDGE_LOOP;
            e.sqrtInfo=denseSqrtInfo(dr.H,2.0);
            edges.push_back(e);
            for(int i:in){ const auto& m=ranked[ci].m[i]; linkFeatures(kn.id,m.queryIdx,kk.id,m.trainIdx); }
            ++loops; lastLoopKF=n;
            const Pose Tbefore=kn.pose();
            std::cout<<"[LOOP] keyframe "<<kn.id<<" (frame "<<kn.frameIndex<<") <-> keyframe "<<kk.id
                     <<" (frame "<<kk.frameIndex<<"), RANSAC inliers "<<in.size()<<", dense inliers "
                     <<std::fixed<<std::setprecision(2)<<dr.inlierRatio<<", drift "
                     <<std::setprecision(3)<<((kk.pose()*e.meas).t-Tbefore.t).norm()<<" m\n";
            poseGraphOptimization();
            return true;
        }
        return false;
    }

    // Global pose graph: sequential edges from the current estimates plus
    // loop edges. Corrections are propagated to velocities and landmarks.
    void poseGraphOptimization(){
        const int n=(int)kfs.size();
        std::vector<Pose> before(n);
        for(int k=0;k<n;++k) before[k]=kfs[k].pose();
        ceres::Problem problem;
        auto* quatParam=new ceres::EigenQuaternionParameterization();
        for(auto& k:kfs){ problem.AddParameterBlock(k.q,4,quatParam); problem.AddParameterBlock(k.p,3); }
        problem.SetParameterBlockConstant(kfs[0].q); problem.SetParameterBlockConstant(kfs[0].p);
        // Sequential constraints keep the current relative poses, weighted by
        // the dense information of that keyframe pair, so loop corrections
        // are absorbed where the tracking was least certain.
        const Eigen::Matrix<double,6,6> looseInfo=diagonalSqrtInfo(PGO_SEQ_ROT_SIGMA,PGO_SEQ_TRANS_SIGMA);
        std::vector<const Eigen::Matrix<double,6,6>*> seqInfo(n,&looseInfo);
        for(const auto& e:edges) if(e.type==EDGE_SEQUENTIAL&&e.to==e.from+1) seqInfo[e.to]=&e.sqrtInfo;
        for(int k=1;k<n;++k)
            problem.AddResidualBlock(RelPoseFactor::create(before[k-1].inverse()*before[k],*seqInfo[k]),nullptr,
                                     kfs[k-1].q,kfs[k-1].p,kfs[k].q,kfs[k].p);
        for(const auto& e:edges)
            if(e.type!=EDGE_SEQUENTIAL)
                problem.AddResidualBlock(RelPoseFactor::create(e.meas,e.sqrtInfo),new ceres::CauchyLoss(LOOP_CAUCHY_SCALE),
                                         kfs[e.from].q,kfs[e.from].p,kfs[e.to].q,kfs[e.to].p);
        ceres::Solver::Options opt;
        opt.linear_solver_type=ceres::SPARSE_NORMAL_CHOLESKY;
        opt.max_num_iterations=PGO_ITERATIONS;
        opt.num_threads=std::max(1,omp_get_max_threads());
        ceres::Solver::Summary sum;
        ceres::Solve(opt,&problem,&sum);
        for(auto& k:kfs){ Eigen::Map<Eigen::Quaterniond> q(k.q); q.normalize(); }

        std::vector<Pose> corr(n);
        for(int k=0;k<n;++k){
            corr[k]=kfs[k].pose()*before[k].inverse();
            kfs[k].setVel(corr[k].R*kfs[k].vel());
        }
        for(auto& l:lms){
            if(l.obs.empty()) continue;
            Eigen::Vector3d X(l.X[0],l.X[1],l.X[2]);
            X=corr[l.obs.front().first].transform(X);
            l.X[0]=X.x(); l.X[1]=X.y(); l.X[2]=X.z();
        }
        const Pose c=corr.back();
        prevFramePose=c*prevFramePose; prevPrevFramePose=c*prevPrevFramePose;
        ++pgoRuns;
    }

    // Dense-verified constraints between keyframes that see the same place
    // at very different times (in addition to the online loop closures).
    void addRevisitConstraints(){
        std::set<std::pair<int,int>> have;
        for(const auto& e:edges) have.insert({std::min(e.from,e.to),std::max(e.from,e.to)});
        const int n=(int)kfs.size();
        for(int round=0;round<REVISIT_ROUNDS;++round){
            int added=0, tried=0;
            for(int j=0;j<n;++j){
                if(isHeldOut(j)) continue;
                const Pose Tj=kfs[j].pose();
                int best=-1; double bestD=1e9;
                for(int i=0;i<j;++i){
                    if(kfs[j].frameIndex-kfs[i].frameIndex<LOOP_MIN_FRAME_GAP||have.count({i,j})) continue;
                    const Pose Ti=kfs[i].pose();
                    const double d=(Ti.t-Tj.t).norm();
                    if(d>REVISIT_MAX_DISTANCE||rotationDeg(Ti.R,Tj.R)>REVISIT_MAX_ANGLE_DEG) continue;
                    if(d<bestD){ bestD=d; best=i; }
                }
                if(best<0) continue;
                ++tried;
                const KeyFrame& ki=kfs[best]; const KeyFrame& kj=kfs[j];
                const DenseImage di=buildDenseImage(ki.gray8,ki.depth16,ki.K), dj=buildDenseImage(kj.gray8,kj.depth16,kj.K);
                const Pose init=(ki.pose().inverse()*kj.pose()).inverse();
                Pose T=init;
                DenseResult dr=denseAlign(di,dj,T);
                if(!dr.ok||dr.inlierRatio<REVISIT_MIN_INLIER_RATIO) continue;
                const Pose a=init.inverse(), b=T.inverse();
                if((a.t-b.t).norm()>REVISIT_MAX_CORRECTION_T||rotationDeg(a.R,b.R)>REVISIT_MAX_CORRECTION_R) continue;
                VisualEdge e; e.from=best; e.to=j; e.meas=b; e.type=EDGE_REVISIT;
                e.sqrtInfo=denseSqrtInfo(dr.H,2.0);
                edges.push_back(e); have.insert({best,j}); ++added; ++revisits;
            }
            std::cout<<"[REVISIT] round "<<round+1<<": "<<added<<" constraints added ("<<tried<<" candidates)\n";
            if(!added) break;
            poseGraphOptimization();
        }
    }

    void globalBundleAdjustment(){
        if(kfs.size()<2) return;
        if(imuInitialized) for(int k=1;k<(int)kfs.size();++k) reintegrateIfNeeded(k);
        optimizeRange(0,(int)kfs.size()-1,true);
    }
};

// ============================================================
// KEYFRAME POSE VERIFICATION
// Independent check of the estimated keyframe poses against the image
// data (not against the optimizer's own factors):
//   * Dense check per keyframe pair: warp keyframe i into keyframe j with
//     the estimated relative pose and measure depth / photometric
//     residuals; then re-run dense alignment STARTING from that pose and
//     measure how far it moves ("dense offset"). A correct pose has an
//     offset close to zero.
//   * Reprojection check per keyframe: landmark pixel + depth error.
// Pairs: consecutive keyframes (local accuracy), loop pairs and revisit
// pairs (same place, far apart in time, NOT used as loops) for global
// accuracy. Each pair is scored with the pure front-end odometry poses
// and with the final optimized poses.
// ============================================================
static constexpr double VERIFY_MAX_OFFSET_T = 0.010;     // m
static constexpr double VERIFY_MAX_OFFSET_R = 0.5;       // deg
static constexpr float VERIFY_MIN_INLIER_RATIO = 0.50f;
static constexpr double VERIFY_MAX_REPROJ_PX = 3.0;
struct PairCheck {
    bool aligned=false;            // dense re-alignment converged
    float inlierRatio=0;           // at the evaluated pose (before re-alignment)
    double depthMedianMM=0;        // median |depth residual|
    double photoRms=0;             // intensity RMS (0..255), brightness offset removed
    double offsetT=0, offsetR=0;   // re-alignment correction (m, deg)
    Pose realigned;                // pose of cur in ref after re-alignment
    bool pass() const { return aligned&&offsetT<=VERIFY_MAX_OFFSET_T&&offsetR<=VERIFY_MAX_OFFSET_R&&inlierRatio>=VERIFY_MIN_INLIER_RATIO; }
};

// T_ij = pose of keyframe j expressed in keyframe i (reference = i).
static PairCheck checkDensePair(const DenseImage& ref,const DenseImage& cur,const Pose& T_ij){
    PairCheck c;
    const DenseLevel& A=ref.lv[0]; const DenseLevel& B=cur.lv[0];
    const Pose T_cr=T_ij.inverse();
    const Eigen::Matrix3f R=T_cr.R.cast<float>(); const Eigen::Vector3f t=T_cr.t.cast<float>();
    std::vector<float> rI, rZ; std::vector<char> zOK;
    int nRef=0;
    for(int v=0;v<A.h;++v) for(int u=0;u<A.w;++u){
        const float z=A.Z.at<float>(v,u);
        if(z<=0) continue;
        ++nRef;
        const Eigen::Vector3f q=R*Eigen::Vector3f((u-A.cx)*z/A.fx,(v-A.cy)*z/A.fy,z)+t;
        if(q.z()<0.05f) continue;
        const float x=B.fx*q.x()/q.z()+B.cx, y=B.fy*q.y()/q.z()+B.cy;
        if(x<1.f||y<1.f||x>B.w-2.f||y>B.h-2.f) continue;
        float Zc=0; const bool hasZ=bilinearDepth(B.Z,x,y,Zc);
        if(hasZ&&std::abs(Zc-q.z())>DENSE_OCCLUSION) continue;     // occluded / disoccluded
        rI.push_back(255.f*(bilinear(B.I,x,y)-A.I.at<float>(v,u)));
        rZ.push_back(hasZ?(Zc-q.z())/(DEPTH_SIGMA_A+DEPTH_SIGMA_B*q.z()*q.z()):0.f);
        zOK.push_back(hasZ);
    }
    if(nRef==0||rI.size()<500) return c;
    // Exposure differs between keyframes far apart in time: remove the median offset.
    std::vector<float> tmp=rI; std::nth_element(tmp.begin(),tmp.begin()+tmp.size()/2,tmp.end());
    const float bias=tmp[tmp.size()/2];
    double sI=0; int inl=0; std::vector<float> absZ;
    for(size_t i=0;i<rI.size();++i){
        const float e=rI[i]-bias; sI+=e*e;
        if(zOK[i]){
            absZ.push_back(std::abs(rZ[i]));
            if(std::abs(rZ[i])<3.f&&std::abs(e)<25.f) ++inl;
        }
    }
    c.photoRms=std::sqrt(sI/rI.size());
    c.inlierRatio=(float)inl/nRef;
    if(!absZ.empty()){
        std::nth_element(absZ.begin(),absZ.begin()+absZ.size()/2,absZ.end());
        // median normalized residual -> mm at the median scene depth of the reference
        std::vector<float> zs; zs.reserve(nRef);
        for(int v=0;v<A.h;v+=4) for(int u=0;u<A.w;u+=4){ float z=A.Z.at<float>(v,u); if(z>0) zs.push_back(z); }
        float zMed=1.f;
        if(!zs.empty()){ std::nth_element(zs.begin(),zs.begin()+zs.size()/2,zs.end()); zMed=zs[zs.size()/2]; }
        c.depthMedianMM=1000.0*absZ[absZ.size()/2]*(DEPTH_SIGMA_A+DEPTH_SIGMA_B*zMed*zMed);
    }
    // Re-align from the evaluated pose and measure the correction.
    Pose T=T_cr;
    DenseResult dr=denseAlign(ref,cur,T);
    c.aligned=dr.ok;
    const Pose before=T_cr.inverse(), after=T.inverse();
    c.offsetT=(after.t-before.t).norm();
    c.offsetR=rotationDeg(before.R,after.R);
    c.realigned=after;
    return c;
}

static void verifyKeyframePoses(const VioSlam& s,const std::string& dir){
    std::cout<<"\n============================================================\n"
               " KEYFRAME POSE VERIFICATION (dense + reprojection)\n"
               "============================================================\n";
    const int n=(int)s.kfs.size();
    if(n<2) return;
    std::vector<DenseImage> dense(n);
    for(int k=0;k<n;++k) dense[k]=buildDenseImage(s.kfs[k].gray8,s.kfs[k].depth16,s.kfs[k].K);

    // ---------------- pairs ----------------
    struct Pair { int i,j; std::string type; };
    std::vector<Pair> pairs;
    std::set<std::pair<int,int>> loopSet;
    for(int k=1;k<n;++k) pairs.push_back({k-1,k,"consecutive"});
    for(const auto& e:s.edges){
        if(e.type==EDGE_SEQUENTIAL||e.type==EDGE_LOCAL) continue;
        pairs.push_back({e.from,e.to,e.type==EDGE_LOOP?"loop":"revisit-edge"});
        loopSet.insert({std::min(e.from,e.to),std::max(e.from,e.to)});
    }
    // Held-out revisit pairs: never used as constraints.
    for(int j=0;j<n;++j){
        if(!isHeldOut(j)) continue;
        const Pose Tj=s.kfs[j].pose();
        int best=-1; double bestD=1e9;
        for(int i=0;i<j;++i){
            if(s.kfs[j].frameIndex-s.kfs[i].frameIndex<LOOP_MIN_FRAME_GAP||loopSet.count({i,j})) continue;
            const Pose Ti=s.kfs[i].pose();
            const double d=(Ti.t-Tj.t).norm();
            if(d>REVISIT_MAX_DISTANCE||rotationDeg(Ti.R,Tj.R)>REVISIT_MAX_ANGLE_DEG) continue;
            if(d<bestD){ bestD=d; best=i; }
        }
        if(best>=0) pairs.push_back({best,j,"revisit-heldout"});
    }

    struct Row { Pair p; PairCheck odo, fin; };
    std::vector<Row> rows(pairs.size());
    for(size_t q=0;q<pairs.size();++q){
        const Pair& p=pairs[q];
        const KeyFrame& a=s.kfs[p.i]; const KeyFrame& b=s.kfs[p.j];
        rows[q].p=p;
        rows[q].odo=checkDensePair(dense[p.i],dense[p.j],a.odometryPose.inverse()*b.odometryPose);
        rows[q].fin=checkDensePair(dense[p.i],dense[p.j],a.pose().inverse()*b.pose());
        if((q+1)%200==0) std::cout<<"[VERIFY] "<<q+1<<"/"<<pairs.size()<<" pairs\n";
    }

    // ---------------- reprojection per keyframe ----------------
    std::vector<double> kfReprojPx(n,0), kfReprojDepthMM(n,0); std::vector<int> kfObs(n,0);
    std::vector<double> allPx;
    for(int k=0;k<n;++k){
        const KeyFrame& kf=s.kfs[k]; const Pose T=kf.pose();
        double sp=0, sd=0; int m=0;
        for(const auto& f:kf.features){
            if(f.landmark<0||s.lms[f.landmark].obs.size()<2) continue;
            const auto& L=s.lms[f.landmark];
            const Eigen::Vector3d xc=T.R.transpose()*(Eigen::Vector3d(L.X[0],L.X[1],L.X[2])-T.t);
            if(xc.z()<0.05) continue;
            const double du=kf.K.fx*xc.x()/xc.z()+kf.K.cx-f.u, dv=kf.K.fy*xc.y()/xc.z()+kf.K.cy-f.v;
            const double e=std::sqrt(du*du+dv*dv);
            sp+=e*e; sd+=std::pow(xc.z()-f.pc.z(),2); ++m; allPx.push_back(e);
        }
        if(m){ kfReprojPx[k]=std::sqrt(sp/m); kfReprojDepthMM[k]=1000*std::sqrt(sd/m); kfObs[k]=m; }
    }

    // ---------------- reports ----------------
    std::ofstream pf(dir+"/pair_quality.txt");
    pf<<"# type kf_i kf_j frame_i frame_j | odometry: aligned offset_mm offset_deg inlier depth_med_mm photo_rms"
        " | final: aligned offset_mm offset_deg inlier depth_med_mm photo_rms pass\n"<<std::fixed<<std::setprecision(3);
    for(const auto& r:rows){
        pf<<r.p.type<<' '<<r.p.i<<' '<<r.p.j<<' '<<s.kfs[r.p.i].frameIndex<<' '<<s.kfs[r.p.j].frameIndex;
        for(const PairCheck* c:{&r.odo,&r.fin})
            pf<<" | "<<c->aligned<<' '<<1000*c->offsetT<<' '<<c->offsetR<<' '<<c->inlierRatio<<' '<<c->depthMedianMM<<' '<<c->photoRms;
        pf<<' '<<(r.fin.pass()?"PASS":"FAIL")<<'\n';
    }

    // Per keyframe: its consecutive pair (with the previous keyframe) + reprojection.
    std::ofstream kq(dir+"/keyframe_quality.txt");
    kq<<"# kf frame | consecutive(final): offset_mm offset_deg inlier depth_med_mm photo_rms | reproj_rms_px reproj_depth_rms_mm landmark_obs | verdict\n"
      <<std::fixed<<std::setprecision(3);
    int kfBad=0;
    for(int k=0;k<n;++k){
        const PairCheck* c=nullptr;
        for(const auto& r:rows) if(r.p.type=="consecutive"&&r.p.j==k){ c=&r.fin; break; }
        const bool reprojOK=kfObs[k]==0||kfReprojPx[k]<=VERIFY_MAX_REPROJ_PX;
        const bool ok=(k==0||(c&&c->pass()))&&reprojOK;
        if(!ok) ++kfBad;
        kq<<k<<' '<<s.kfs[k].frameIndex<<" | ";
        if(c) kq<<1000*c->offsetT<<' '<<c->offsetR<<' '<<c->inlierRatio<<' '<<c->depthMedianMM<<' '<<c->photoRms;
        else kq<<"- - - - -";
        kq<<" | "<<kfReprojPx[k]<<' '<<kfReprojDepthMM[k]<<' '<<kfObs[k]<<" | "<<(ok?"OK":"CHECK")<<'\n';
    }

    auto median=[](std::vector<double> v){ if(v.empty()) return 0.0; std::nth_element(v.begin(),v.begin()+v.size()/2,v.end()); return v[v.size()/2]; };
    auto pct=[](std::vector<double> v,double p){ if(v.empty()) return 0.0; std::sort(v.begin(),v.end()); return v[std::min(v.size()-1,(size_t)(p*v.size()))]; };
    std::ostringstream ss;
    ss<<std::fixed<<std::setprecision(2);
    ss<<"pair type       poses     count  aligned  offset_mm med/p95   offset_deg med/p95   inlier med  depth_med_mm  PASS\n";
    for(const std::string type:{"consecutive","loop","revisit-edge","revisit-heldout"}){
        for(int which=0;which<2;++which){
            std::vector<double> ot,orr,inl,dz; int cnt=0,al=0,pass=0;
            for(const auto& r:rows){
                if(r.p.type!=type) continue;
                const PairCheck& c=which?r.fin:r.odo;
                ++cnt; if(c.aligned){ ++al; ot.push_back(1000*c.offsetT); orr.push_back(c.offsetR); }
                inl.push_back(c.inlierRatio); dz.push_back(c.depthMedianMM); if(c.pass()) ++pass;
            }
            if(!cnt) continue;
            ss<<std::left<<std::setw(16)<<type<<std::setw(10)<<(which?"final":"odometry")<<std::right
              <<std::setw(5)<<cnt<<std::setw(9)<<al
              <<std::setw(10)<<median(ot)<<" /"<<std::setw(7)<<pct(ot,0.95)
              <<std::setw(12)<<median(orr)<<" /"<<std::setw(6)<<pct(orr,0.95)
              <<std::setw(13)<<median(inl)<<std::setw(14)<<median(dz)
              <<std::setw(7)<<(100*pass/cnt)<<"%\n";
        }
    }
    ss<<"\nlandmark reprojection (final): median "<<median(allPx)<<" px, p95 "<<pct(allPx,0.95)<<" px over "<<allPx.size()<<" observations\n";
    ss<<"keyframes needing attention (consecutive check failed or reprojection > "<<VERIFY_MAX_REPROJ_PX<<" px): "<<kfBad<<" / "<<n<<"\n";
    ss<<"PASS = dense re-alignment converged, moved < "<<VERIFY_MAX_OFFSET_T*1000<<" mm and < "<<VERIFY_MAX_OFFSET_R
      <<" deg, inlier ratio >= "<<VERIFY_MIN_INLIER_RATIO<<"\n";
    std::ofstream(dir+"/verification.txt")<<ss.str();
    std::cout<<ss.str()<<"Details: "<<dir<<"/pair_quality.txt, "<<dir<<"/keyframe_quality.txt\n";
}

// ============================================================
// TSDF FUSION (voxel hashing, 8^3 blocks)
// ============================================================
class TSDFVolume {
public:
    static constexpr int B=8;
    struct Block { float tsdf[B*B*B]; float w[B*B*B]; uint8_t rgb[B*B*B][3]; };
    std::unordered_map<int64_t,std::unique_ptr<Block>> blocks;

    static int64_t key(int x,int y,int z){
        return ((int64_t)(x+(1<<20))<<42)|((int64_t)(y+(1<<20))<<21)|(int64_t)(z+(1<<20));
    }

    void integrate(const cv::Mat& depth16,const cv::Mat& rgb,const Intrinsics& K,const Pose& T_wc){
        const float bs=B*TSDF_VOXEL;
        // 1) Allocate blocks around every observed surface point.
        std::vector<int64_t> touched;
        std::unordered_set<int64_t> seen;
        for(int v=0;v<depth16.rows;v+=2) for(int u=0;u<depth16.cols;u+=2){
            const float z=depth16.at<uint16_t>(v,u)*DEPTH_SCALE;
            if(z<MIN_DEPTH||z>MAX_DEPTH) continue;
            const Eigen::Vector3d ray((u-K.cx)/K.fx,(v-K.cy)/K.fy,1.0);
            for(float d=-TSDF_TRUNCATION;d<=TSDF_TRUNCATION;d+=bs*0.5f){
                const Eigen::Vector3d pw=T_wc.transform(ray*(z+d));
                const int64_t k=key((int)std::floor(pw.x()/bs),(int)std::floor(pw.y()/bs),(int)std::floor(pw.z()/bs));
                if(seen.insert(k).second) touched.push_back(k);
            }
        }
        for(int64_t k:touched){
            auto& b=blocks[k];
            if(!b){ b.reset(new Block()); std::fill(b->tsdf,b->tsdf+B*B*B,1.f); std::fill(b->w,b->w+B*B*B,0.f); std::memset(b->rgb,0,sizeof(b->rgb)); }
        }
        // 2) Update voxels of touched blocks.
        const Pose T_cw=T_wc.inverse();
        const Eigen::Matrix3f Rcw=T_cw.R.cast<float>(); const Eigen::Vector3f tcw=T_cw.t.cast<float>();
        #pragma omp parallel for schedule(dynamic,16)
        for(int i=0;i<(int)touched.size();++i){
            const int64_t k=touched[i];
            const int bx=(int)((k>>42)&((1<<21)-1))-(1<<20), by=(int)((k>>21)&((1<<21)-1))-(1<<20), bz=(int)(k&((1<<21)-1))-(1<<20);
            Block& blk=*blocks.at(k);
            for(int vz=0;vz<B;++vz) for(int vy=0;vy<B;++vy) for(int vx=0;vx<B;++vx){
                const Eigen::Vector3f pw(((bx*B+vx)+0.5f)*TSDF_VOXEL,((by*B+vy)+0.5f)*TSDF_VOXEL,((bz*B+vz)+0.5f)*TSDF_VOXEL);
                const Eigen::Vector3f pc=Rcw*pw+tcw;
                if(pc.z()<MIN_DEPTH) continue;
                const int u=(int)std::lround(K.fx*pc.x()/pc.z()+K.cx), v=(int)std::lround(K.fy*pc.y()/pc.z()+K.cy);
                if(u<0||v<0||u>=depth16.cols||v>=depth16.rows) continue;
                const float z=depth16.at<uint16_t>(v,u)*DEPTH_SCALE;
                if(z<MIN_DEPTH||z>MAX_DEPTH) continue;
                const float sdf=z-pc.z();
                if(sdf<-TSDF_TRUNCATION) continue;
                const float tsdf=std::min(1.f,sdf/TSDF_TRUNCATION);
                const int idx=(vz*B+vy)*B+vx;
                const float w0=blk.w[idx], w1=std::min(TSDF_MAX_WEIGHT,w0+1.f);
                blk.tsdf[idx]=(blk.tsdf[idx]*w0+tsdf)/(w0+1.f);
                const cv::Vec3b c=rgb.at<cv::Vec3b>(std::min(RGB_HEIGHT-1,v*2),std::min(RGB_WIDTH-1,u*2));
                for(int ch=0;ch<3;++ch) blk.rgb[idx][ch]=(uint8_t)std::lround((blk.rgb[idx][ch]*w0+c[2-ch])/(w0+1.f));
                blk.w[idx]=w1;
            }
        }
    }

    bool voxel(int x,int y,int z,float& t,float& w,const uint8_t*& c) const {
        auto fd=[](int a){ return a>=0?a/B:(a-B+1)/B; };
        const int bx=fd(x), by=fd(y), bz=fd(z);
        auto it=blocks.find(key(bx,by,bz));
        if(it==blocks.end()) return false;
        const int idx=((z-bz*B)*B+(y-by*B))*B+(x-bx*B);
        t=it->second->tsdf[idx]; w=it->second->w[idx]; c=it->second->rgb[idx];
        return true;
    }

    // Surface = zero crossings along x, y and z between neighboring voxels.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr extractSurface() const {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        for(const auto& kv:blocks){
            const int64_t k=kv.first;
            const int bx=(int)((k>>42)&((1<<21)-1))-(1<<20), by=(int)((k>>21)&((1<<21)-1))-(1<<20), bz=(int)(k&((1<<21)-1))-(1<<20);
            const Block& blk=*kv.second;
            for(int vz=0;vz<B;++vz) for(int vy=0;vy<B;++vy) for(int vx=0;vx<B;++vx){
                const int idx=(vz*B+vy)*B+vx;
                if(blk.w[idx]<TSDF_MIN_WEIGHT) continue;
                const float t0=blk.tsdf[idx];
                const int x=bx*B+vx, y=by*B+vy, z=bz*B+vz;
                const int nb[3][3]={{1,0,0},{0,1,0},{0,0,1}};
                for(const auto& d:nb){
                    float t1,w1; const uint8_t* c1;
                    if(!voxel(x+d[0],y+d[1],z+d[2],t1,w1,c1)||w1<TSDF_MIN_WEIGHT) continue;
                    if((t0>0)==(t1>0)||std::abs(t0-t1)>1.5f) continue;
                    const float a=t0/(t0-t1);
                    pcl::PointXYZRGB p;
                    p.x=(x+0.5f+a*d[0])*TSDF_VOXEL; p.y=(y+0.5f+a*d[1])*TSDF_VOXEL; p.z=(z+0.5f+a*d[2])*TSDF_VOXEL;
                    p.r=(uint8_t)((1-a)*blk.rgb[idx][0]+a*c1[0]);
                    p.g=(uint8_t)((1-a)*blk.rgb[idx][1]+a*c1[1]);
                    p.b=(uint8_t)((1-a)*blk.rgb[idx][2]+a*c1[2]);
                    cloud->push_back(p);
                }
            }
        }
        cloud->width=(uint32_t)cloud->size(); cloud->height=1; cloud->is_dense=false;
        return cloud;
    }
};

// ============================================================
// SIMPLE VOXEL MAP (progressive playback)
// ============================================================
struct VoxelKey { int x,y,z; bool operator==(const VoxelKey& o)const{return x==o.x&&y==o.y&&z==o.z;} };
struct VoxelHash { size_t operator()(const VoxelKey& k)const{ return ((size_t)(uint32_t)k.x*73856093u)^((size_t)(uint32_t)k.y*19349663u)^((size_t)(uint32_t)k.z*83492791u); } };

// ============================================================
// LIVE VIEW (--live)
// The SLAM thread publishes its state after every frame; the main thread
// renders it. Each keyframe's points stay in that keyframe's camera frame
// and are drawn with the keyframe's CURRENT pose, so loop closures and
// optimization move the existing map instead of leaving stale copies.
// ============================================================
using LiveCloud = pcl::PointCloud<pcl::PointXYZRGB>;

// Points of one RGB-D image in its own camera frame (1 cm voxels, flying pixels removed).
static LiveCloud::Ptr cameraCloud(const cv::Mat& rgb,const cv::Mat& depth16,const Intrinsics& K,int stride){
    LiveCloud::Ptr c(new LiveCloud);
    std::unordered_set<VoxelKey,VoxelHash> seen;
    auto Z=[&](int v,int u){ return depth16.at<uint16_t>(v,u)*DEPTH_SCALE; };
    for(int v=0;v<depth16.rows;v+=stride) for(int u=0;u<depth16.cols;u+=stride){
        const float z=Z(v,u);
        if(z<MIN_DEPTH||z>MAX_DEPTH) continue;
        if(u+stride<depth16.cols&&std::abs(Z(v,u+stride)-z)>LIVE_EDGE_JUMP*z) continue;
        if(v+stride<depth16.rows&&std::abs(Z(v+stride,u)-z)>LIVE_EDGE_JUMP*z) continue;
        const float x=(u-K.cx)*z/K.fx, y=(v-K.cy)*z/K.fy;
        if(!seen.insert({(int)std::floor(x/LIVE_VOXEL),(int)std::floor(y/LIVE_VOXEL),(int)std::floor(z/LIVE_VOXEL)}).second) continue;
        const cv::Vec3b col=rgb.at<cv::Vec3b>(std::min(rgb.rows-1,v*rgb.rows/depth16.rows),std::min(rgb.cols-1,u*rgb.cols/depth16.cols));
        pcl::PointXYZRGB p; p.x=x; p.y=y; p.z=z; p.b=col[0]; p.g=col[1]; p.r=col[2];
        c->points.push_back(p);
    }
    c->width=(uint32_t)c->size(); c->height=1; c->is_dense=true;
    return c;
}

struct LiveState {
    int frame=-1, numFrames=0, status=0, keyframes=0, loops=0;
    float inlier=0; bool isKeyframe=false, imu=false, finished=false;
    std::string stage="tracking";
    cv::Mat rgb; Pose pose; double fps=0;
    std::vector<Pose> kfPoses;
    std::vector<Eigen::Vector3d> trajectory;
    LiveCloud::Ptr current;                       // current frame, camera frame
};
struct LiveKeyframe { int id; LiveCloud::Ptr cloud; };

class LiveFeed {
public:
    std::atomic<bool> paused{false}, viewerOpen{true}, slamDone{false};
    void publish(const LiveState& s,std::vector<LiveKeyframe> kfs={}){
        std::lock_guard<std::mutex> l(m_);
        latest_=s; fresh_=true;
        for(auto& k:kfs) pending_.push_back(std::move(k));
    }
    bool take(LiveState& s,std::vector<LiveKeyframe>& kfs){
        std::lock_guard<std::mutex> l(m_);
        if(!fresh_) return false;
        s=latest_; fresh_=false; kfs.clear(); kfs.swap(pending_);
        return true;
    }
private:
    std::mutex m_; LiveState latest_; bool fresh_=false; std::vector<LiveKeyframe> pending_;
};

class LiveViewer {
public:
    pcl::visualization::PCLVisualizer::Ptr viewer;
    std::atomic<bool> pauseKey{false}, followKey{false};

    LiveViewer(){
        viewer.reset(new pcl::visualization::PCLVisualizer("VIO SLAM - live 3D scene"));
        viewer->setBackgroundColor(0.05,0.05,0.05); viewer->setSize(1400,900); viewer->initCameraParameters();
        viewer->setCameraPosition(0,0,-2,0,0,0,0,-1,0);
        viewer->registerKeyboardCallback([this](const pcl::visualization::KeyboardEvent& e){
            if(!e.keyDown()) return;
            if(e.getKeySym()=="space") pauseKey=true;
            else if(e.getKeySym()=="v"||e.getKeySym()=="V") followKey=true;
        });
        trajPts=vtkSmartPointer<vtkPoints>::New(); trajLines=vtkSmartPointer<vtkCellArray>::New();
        traj=vtkSmartPointer<vtkPolyData>::New(); traj->SetPoints(trajPts); traj->SetLines(trajLines);
        viewer->addModelFromPolyData(traj,"trajectory");
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,0.1,0.4,1.0,"trajectory");
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,3,"trajectory");
        frustumPts=vtkSmartPointer<vtkPoints>::New(); frustumPts->SetNumberOfPoints(5);
        for(int i=0;i<5;++i) frustumPts->SetPoint(i,0,0,0);
        auto lines=vtkSmartPointer<vtkCellArray>::New();
        const int e[8][2]={{0,1},{0,2},{0,3},{0,4},{1,2},{2,3},{3,4},{4,1}};
        for(const auto& s:e){ vtkIdType ids[2]={s[0],s[1]}; lines->InsertNextCell(2,ids); }
        frustum=vtkSmartPointer<vtkPolyData>::New(); frustum->SetPoints(frustumPts); frustum->SetLines(lines);
        viewer->addModelFromPolyData(frustum,"frustum");
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,1.0,0.0,0.0,"frustum");
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH,3,"frustum");
        viewer->addText("",10,10,14,1,1,1,"hud");
        cv::namedWindow("VIO SLAM - live RGB",cv::WINDOW_NORMAL); cv::resizeWindow("VIO SLAM - live RGB",768,576);
    }
    ~LiveViewer(){ cv::destroyWindow("VIO SLAM - live RGB"); }

    void apply(const LiveState& s,const std::vector<LiveKeyframe>& newKfs,bool follow){
        for(const auto& k:newKfs){
            const std::string id="kf_"+std::to_string(k.id);
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(k.cloud);
            viewer->addPointCloud<pcl::PointXYZRGB>(k.cloud,rgb,id);
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,2,id);
            if((int)shown.size()<=k.id) shown.resize(k.id+1);
            shown[k.id].added=true; shown[k.id].placed=false;
        }
        // Move keyframe clouds whose pose changed (new, optimized or loop-corrected).
        for(int k=0;k<(int)s.kfPoses.size()&&k<(int)shown.size();++k){
            Shown& sh=shown[k];
            if(!sh.added) continue;
            const Pose& T=s.kfPoses[k];
            if(sh.placed&&(T.t-sh.pose.t).norm()<1e-4&&(T.R-sh.pose.R).norm()<1e-4) continue;
            Eigen::Affine3f A=Eigen::Affine3f::Identity();
            A.linear()=T.R.cast<float>(); A.translation()=T.t.cast<float>();
            viewer->updatePointCloudPose("kf_"+std::to_string(k),A);
            sh.pose=T; sh.placed=true;
        }
        if(s.current&&!s.current->empty()){
            Eigen::Affine3f A=Eigen::Affine3f::Identity();
            A.linear()=s.pose.R.cast<float>(); A.translation()=s.pose.t.cast<float>();
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(s.current);
            if(!viewer->updatePointCloud<pcl::PointXYZRGB>(s.current,rgb,"current")){
                viewer->addPointCloud<pcl::PointXYZRGB>(s.current,rgb,"current");
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,3,"current");
            }
            viewer->updatePointCloudPose("current",A);
        }else if(s.finished) viewer->removePointCloud("current");

        trajPts->SetNumberOfPoints((vtkIdType)s.trajectory.size());
        for(size_t i=0;i<s.trajectory.size();++i) trajPts->SetPoint((vtkIdType)i,s.trajectory[i].x(),s.trajectory[i].y(),s.trajectory[i].z());
        trajLines->Reset();
        if(s.trajectory.size()>=2){
            trajLines->InsertNextCell((vtkIdType)s.trajectory.size());
            for(size_t i=0;i<s.trajectory.size();++i) trajLines->InsertCellPoint((vtkIdType)i);
        }
        trajPts->Modified(); trajLines->Modified(); traj->Modified();

        const double z=.20,w=.12,h=.09;
        const Eigen::Vector3d c[5]={Eigen::Vector3d::Zero(),{-w,-h,z},{w,-h,z},{w,h,z},{-w,h,z}};
        for(int i=0;i<5;++i){ const Eigen::Vector3d p=s.pose.transform(c[i]); frustumPts->SetPoint(i,p.x(),p.y(),p.z()); }
        frustumPts->Modified(); frustum->Modified();
        if(follow){
            const Eigen::Vector3d fwd=s.pose.R*Eigen::Vector3d(0,0,1), up=s.pose.R*Eigen::Vector3d(0,-1,0);
            const Eigen::Vector3d pos=s.pose.t-fwd*LIVE_FOLLOW_DISTANCE;
            viewer->setCameraPosition(pos.x(),pos.y(),pos.z(),s.pose.t.x(),s.pose.t.y(),s.pose.t.z(),up.x(),up.y(),up.z());
        }

        std::ostringstream hud;
        hud<<"Frame "<<s.frame<<"/"<<s.numFrames-1<<"  KF "<<s.keyframes<<"  loops "<<s.loops
           <<"  "<<std::fixed<<std::setprecision(1)<<s.fps<<" FPS  "<<(s.imu?"VIO":"VO")<<"  ["<<s.stage<<"]"
           <<(follow?"":"  [free view]");
        viewer->updateText(hud.str(),10,10,14,1,1,1,"hud");
        if(!s.rgb.empty()){
            cv::Mat d=s.rgb.clone();
            cv::putText(d,hud.str(),cv::Point(10,22),cv::FONT_HERSHEY_SIMPLEX,.45,cv::Scalar(0,255,0),1,cv::LINE_AA);
            std::string st=s.status==1?"DENSE TRACKED":(s.status==2?"IMU-ONLY PREDICTION":"NOT TRACKED");
            if(s.isKeyframe) st+="  KEYFRAME";
            std::ostringstream inl; inl<<"  inliers "<<std::fixed<<std::setprecision(2)<<s.inlier;
            cv::putText(d,st+inl.str(),cv::Point(10,44),cv::FONT_HERSHEY_SIMPLEX,.45,s.status==1?cv::Scalar(0,255,0):cv::Scalar(0,165,255),1,cv::LINE_AA);
            cv::imshow("VIO SLAM - live RGB",d);
        }
    }
    void spin(){ viewer->spinOnce(1,true); }
    bool stopped(){ return viewer->wasStopped(); }

private:
    struct Shown { bool added=false, placed=false; Pose pose; };
    std::vector<Shown> shown;
    vtkSmartPointer<vtkPoints> trajPts, frustumPts;
    vtkSmartPointer<vtkCellArray> trajLines;
    vtkSmartPointer<vtkPolyData> traj, frustum;
};

// Main-thread loop: renders whatever the SLAM thread publishes until the
// window is closed. Closing the window early lets SLAM finish headless.
static void runLiveViewer(LiveFeed& feed){
    LiveViewer view;
    LiveState s; std::vector<LiveKeyframe> kfs;
    bool follow=true;
    while(true){
        if(feed.take(s,kfs)) view.apply(s,kfs,follow);
        view.spin();
        const int k=cv::waitKey(feed.slamDone?15:1);
        if(view.stopped()||k==27||k=='q'||k=='Q') break;
        if(k==' '||view.pauseKey.exchange(false)) feed.paused=!feed.paused;
        if(k=='v'||k=='V'||view.followKey.exchange(false)){ follow=!follow; view.apply(s,{},follow); }
    }
    feed.viewerOpen=false; feed.paused=false;
    if(!feed.slamDone) std::cout<<"[LIVE] Viewer closed - SLAM continues without display.\n";
}

// ============================================================
// FRAME LOADING (capture thread)
// ============================================================
template<typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t n):max_(n){}
    void push(T x){ std::unique_lock<std::mutex> l(m_); full_.wait(l,[&]{return q_.size()<max_||stop_;}); if(stop_) return; q_.push_back(std::move(x)); empty_.notify_one(); }
    bool pop(T& x){ std::unique_lock<std::mutex> l(m_); empty_.wait(l,[&]{return !q_.empty()||stop_;}); if(q_.empty()) return false; x=std::move(q_.front()); q_.pop_front(); full_.notify_one(); return true; }
    void stop(){ { std::lock_guard<std::mutex> l(m_); stop_=true; } empty_.notify_all(); full_.notify_all(); }
private:
    std::deque<T> q_; size_t max_; std::mutex m_; std::condition_variable empty_,full_; bool stop_=false;
};

struct RawFrame { int index=-1; cv::Mat rgb, depth16; };

// ============================================================
// OUTPUT
// ============================================================
static void writePose(std::ostream& o,const Pose& p){
    Eigen::Quaterniond q(p.R); q.normalize();
    o<<p.t.x()<<' '<<p.t.y()<<' '<<p.t.z()<<' '<<q.x()<<' '<<q.y()<<' '<<q.z()<<' '<<q.w();
}

struct SavedResults {
    std::vector<int> status;            // per frame
    std::vector<Pose> poses;            // per frame
    std::vector<bool> isKeyframe, loop;
    std::vector<float> inlierRatio;
    std::vector<std::pair<int,Eigen::Vector3d>> landmarks;   // (first keyframe frame index, X)
};

static SavedResults collectResults(const VioSlam& s,int numFrames){
    SavedResults r;
    r.status.assign(numFrames,0); r.poses.assign(numFrames,Pose());
    r.isKeyframe.assign(numFrames,false); r.loop.assign(numFrames,false); r.inlierRatio.assign(numFrames,0.f);
    int last=-1;
    for(int i=0;i<numFrames&&i<(int)s.frames.size();++i){
        const FrameRecord& f=s.frames[i];
        if(f.refKF<0){ if(last>=0) r.poses[i]=r.poses[last]; continue; }
        r.status[i]=f.status; r.poses[i]=f.finalPose; r.isKeyframe[i]=f.isKeyframe; r.loop[i]=f.loop; r.inlierRatio[i]=f.inlierRatio;
        last=i;
    }
    for(const auto& l:s.lms){
        if(l.obs.size()<2) continue;
        int first=std::numeric_limits<int>::max();
        for(const auto& o:l.obs) first=std::min(first,s.kfs[o.first].frameIndex);
        r.landmarks.emplace_back(first,Eigen::Vector3d(l.X[0],l.X[1],l.X[2]));
    }
    std::sort(r.landmarks.begin(),r.landmarks.end(),[](const auto& a,const auto& b){return a.first<b.first;});
    return r;
}

static void saveResults(const std::string& dir,const VioSlam& s,const SavedResults& r,const PhoneData& pd){
    fs::create_directories(dir);
    const int n=(int)r.status.size();
    const double t0=pd.frameTime.front();
    std::ofstream tum(dir+"/trajectory_tum.txt"), all(dir+"/trajectory_all_frames.txt"), compat(dir+"/optimized_trajectory.txt");
    tum<<"# timestamp tx ty tz qx qy qz qw (phone clock)\n"<<std::fixed<<std::setprecision(9);
    all<<"# frame_id timestamp tx ty tz qx qy qz qw status(1=dense,2=imu-only,0=none) keyframe loop inlier_ratio\n"<<std::fixed<<std::setprecision(9);
    compat<<"# frame_id timestamp tx ty tz qx qy qz qw keyframe_id\n"<<std::fixed<<std::setprecision(9);
    for(int i=0;i<n;++i){
        tum<<pd.frameTime[i]<<' '; writePose(tum,r.poses[i]); tum<<'\n';
        all<<i<<' '<<pd.frameTime[i]-t0<<' '; writePose(all,r.poses[i]);
        all<<' '<<r.status[i]<<' '<<(r.isKeyframe[i]?1:0)<<' '<<(r.loop[i]?1:0)<<' '<<r.inlierRatio[i]<<'\n';
        if(r.status[i]==1&&i<(int)s.frames.size()){ compat<<i<<' '<<pd.frameTime[i]-t0<<' '; writePose(compat,r.poses[i]); compat<<' '<<s.frames[i].refKF<<'\n'; }
    }
    // Front-end odometry only (no loop closure / PGO / BA corrections), for comparison.
    std::ofstream odo(dir+"/trajectory_odometry.txt");
    odo<<"# frame_id timestamp tx ty tz qx qy qz qw keyframe_id\n"<<std::fixed<<std::setprecision(9);
    for(int i=0;i<n&&i<(int)s.frames.size();++i){
        const FrameRecord& f=s.frames[i];
        if(r.status[i]!=1||f.refKF<0) continue;
        odo<<i<<' '<<pd.frameTime[i]-t0<<' '; writePose(odo,s.kfs[f.refKF].odometryPose*f.T_kf_c); odo<<' '<<f.refKF<<'\n';
    }
    std::ofstream kf(dir+"/keyframes.txt");
    kf<<"# kf_id frame_id timestamp tx ty tz qx qy qz qw vx vy vz bgx bgy bgz bax bay baz\n"<<std::fixed<<std::setprecision(6);
    for(const auto& k:s.kfs){
        kf<<k.id<<' '<<k.frameIndex<<' '<<k.t<<' '; writePose(kf,k.pose());
        kf<<' '<<k.vel().transpose()<<' '<<k.gyroBias().transpose()<<' '<<k.accBias().transpose()<<'\n';
    }
    std::ofstream lm(dir+"/landmarks.txt");
    lm<<"# first_frame_id x y z\n"<<std::fixed<<std::setprecision(5);
    for(const auto& l:r.landmarks) lm<<l.first<<' '<<l.second.transpose()<<'\n';

    int dense=0, imuOnly=0, none=0;
    for(int st:r.status){ if(st==1)++dense; else if(st==2)++imuOnly; else ++none; }
    double path=0; for(int i=1;i<n;++i) path+=(r.poses[i].t-r.poses[i-1].t).norm();
    int loopEdges=0, revisitEdges=0;
    for(const auto& e:s.edges){ if(e.type==EDGE_LOOP) ++loopEdges; else if(e.type==EDGE_REVISIT) ++revisitEdges; }
    std::ostringstream ss;
    ss<<"frames="<<n<<"\nduration_s="<<pd.frameTime[n-1]-pd.frameTime[0]
      <<"\ndense_tracked_frames="<<dense<<"\nimu_only_frames="<<imuOnly<<"\nuntracked_frames="<<none
      <<"\nsparse_reinitializations="<<s.sparseInits<<"\nkeyframe_promotions="<<s.promotions
      <<"\nkeyframes="<<s.kfs.size()<<"\nlandmarks="<<r.landmarks.size()
      <<"\nvisual_edges="<<s.edges.size()<<"\nloop_closures="<<loopEdges<<"\nrevisit_constraints="<<revisitEdges
      <<"\nrefinement_constraints="<<s.localEdges<<"\npose_graph_runs="<<s.pgoRuns
      <<"\nimu_used="<<(s.useImu?1:0)<<"\nimu_initialized="<<(s.imuInitialized?1:0)
      <<"\npath_length_m="<<path<<"\nfinal_position="<<r.poses.back().t.transpose()<<"\n";
    if(!s.kfs.empty()) ss<<"final_gyro_bias="<<s.kfs.back().gyroBias().transpose()<<"\nfinal_acc_bias="<<s.kfs.back().accBias().transpose()<<"\n";
    std::ofstream(dir+"/summary.txt")<<ss.str();
    std::cout<<"\n============================================================\n SUMMARY\n============================================================\n"<<ss.str()<<"Results: "<<dir<<"\n";
}

static void buildScene(const std::string& dir,const SavedResults& r,const PhoneData& pd,const std::vector<std::string>& depthFiles){
    std::cout<<"\n[TSDF] Fusing frames (voxel "<<TSDF_VOXEL*100<<" cm, truncation "<<TSDF_TRUNCATION*100<<" cm)...\n";
    cv::VideoCapture cap(VIDEO_PATH);
    TSDFVolume tsdf;
    auto start=std::chrono::steady_clock::now();
    int fused=0;
    for(int i=0;i<(int)r.status.size();++i){
        cv::Mat raw;
        if(!cap.read(raw)||raw.empty()) break;
        if(r.status[i]!=1||(i%TSDF_FRAME_INTERVAL!=0&&!r.isKeyframe[i])) continue;
        cv::Mat rgb,depth;
        if(!loadDepth(depthFiles[i],depth)) continue;
        cv::resize(raw,rgb,cv::Size(RGB_WIDTH,RGB_HEIGHT),0,0,cv::INTER_AREA);
        tsdf.integrate(depth,rgb,pd.frameK[i],r.poses[i]);
        if(++fused%300==0) std::cout<<"[TSDF] frame "<<i<<", blocks "<<tsdf.blocks.size()<<"\n";
    }
    auto cloud=tsdf.extractSurface();
    const double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if(!cloud->empty()) pcl::io::savePLYFileBinary(dir+"/scene_tsdf.ply",*cloud);
    std::cout<<"[TSDF] "<<fused<<" frames fused in "<<std::fixed<<std::setprecision(1)<<secs<<" s, surface points "
             <<cloud->size()<<" -> "<<dir<<"/scene_tsdf.ply\n";
}

// ============================================================
// VISUALIZER (same look as a.cpp)
// ============================================================
// ============================================================
// MAIN
// ============================================================
// ============================================================
// PER-FRAME POSE CHECK + AUTOMATIC REFINEMENT
// Every frame's final pose is checked by dense reprojection against a
// keyframe it was NOT tracked against (by default the next keyframe): the
// frame's depth + intensity are warped with the estimated relative pose,
// residuals are measured, then dense re-alignment shows how far the pose
// is from the best photometric/geometric fit.
//
// Frames that fail drive refinement rounds:
//   1. every keyframe interval with failing frames gets new dense
//      constraints: the two keyframes re-aligned directly, skip links to
//      the neighbouring keyframes, and a link bridged through the best
//      failing frame;
//   2. pose graph + global BA are re-run and every frame is re-checked;
//   3. the round is KEPT only if more frames pass AND the held-out revisit
//      check (keyframe pairs never used as constraints) is not worse -
//      otherwise it is rolled back and refinement stops.
// Finally, frames that still fail are re-tracked against their other
// keyframe; the new pose is kept only if it passes the check against the
// original keyframe (still an image it was not aligned to).
// ============================================================
static constexpr int REFINE_MAX_ROUNDS = 3;
static constexpr int REFINE_MIN_GAIN = 10;                  // frames that must newly pass to keep a round
static constexpr double REFINE_HELDOUT_TOLERANCE = 1.05;    // held-out median offset may grow <= 5%

// Grayscale of every frame kept from tracking; depth is re-read from disk.
struct FrameImages {
    std::vector<cv::Mat> gray8;
    const std::vector<std::string>* depthFiles=nullptr;
    const PhoneData* pd=nullptr;
    bool has(int i) const { return i>=0&&i<(int)gray8.size()&&!gray8[i].empty(); }
    bool dense(int i,DenseImage& out) const {
        cv::Mat depth;
        if(!has(i)||!loadDepth((*depthFiles)[i],depth)) return false;
        out=buildDenseImage(gray8[i],depth,pd->frameK[i]);
        return true;
    }
};

static std::vector<DenseImage> keyframeDense(const VioSlam& s){
    std::vector<DenseImage> kd(s.kfs.size());
    #pragma omp parallel for schedule(dynamic)
    for(int k=0;k<(int)s.kfs.size();++k) kd[k]=buildDenseImage(s.kfs[k].gray8,s.kfs[k].depth16,s.kfs[k].K);
    return kd;
}

static int frameCheckKF(const VioSlam& s,int i){
    const FrameRecord& f=s.frames[i];
    if(f.checkKF>=0) return f.checkKF;
    return f.refKF+1<(int)s.kfs.size()?f.refKF+1:f.refKF-1;
}

struct FrameCheck {
    int frame=-1, kf=-1, status=0; PairCheck c;
    bool lowOverlap() const { return c.aligned&&c.inlierRatio<VERIFY_MIN_INLIER_RATIO; }
    bool failed() const { return !c.pass()&&!lowOverlap(); }
};

static std::vector<FrameCheck> checkFrames(const VioSlam& s,const std::vector<DenseImage>& kd,const FrameImages& img){
    std::vector<int> ids;
    for(int i=0;i<(int)s.frames.size();++i)
        if(s.frames[i].refKF>=0&&img.has(i)&&frameCheckKF(s,i)>=0) ids.push_back(i);
    std::vector<FrameCheck> rows(ids.size());
    #pragma omp parallel for schedule(dynamic,8)
    for(int q=0;q<(int)ids.size();++q){
        const int i=ids[q]; const int k=frameCheckKF(s,i);
        FrameCheck& r=rows[q]; r.frame=i; r.kf=k; r.status=s.frames[i].status;
        DenseImage d;
        if(img.dense(i,d)) r.c=checkDensePair(kd[k],d,s.kfs[k].pose().inverse()*s.frames[i].finalPose);
    }
    return rows;
}

static int countPass(const std::vector<FrameCheck>& rows){ int p=0; for(const auto& r:rows) if(r.c.pass()) ++p; return p; }

// Held-out revisit pairs (odd keyframes, never constraints), fixed before refinement.
static std::vector<std::pair<int,int>> heldOutPairs(const VioSlam& s){
    std::set<std::pair<int,int>> used;
    for(const auto& e:s.edges) used.insert({std::min(e.from,e.to),std::max(e.from,e.to)});
    std::vector<std::pair<int,int>> out;
    const int n=(int)s.kfs.size();
    for(int j=0;j<n;++j){
        if(!isHeldOut(j)) continue;
        const Pose Tj=s.kfs[j].pose();
        int best=-1; double bestD=1e9;
        for(int i=0;i<j;++i){
            if(s.kfs[j].frameIndex-s.kfs[i].frameIndex<LOOP_MIN_FRAME_GAP||used.count({i,j})) continue;
            const Pose Ti=s.kfs[i].pose();
            const double d=(Ti.t-Tj.t).norm();
            if(d>REVISIT_MAX_DISTANCE||rotationDeg(Ti.R,Tj.R)>REVISIT_MAX_ANGLE_DEG) continue;
            if(d<bestD){ bestD=d; best=i; }
        }
        if(best>=0) out.push_back({best,j});
    }
    return out;
}

struct HeldOutScore { int pass=0, n=0; double medianMM=0; };
static HeldOutScore scoreHeldOut(const VioSlam& s,const std::vector<DenseImage>& kd,const std::vector<std::pair<int,int>>& pairs){
    std::vector<PairCheck> c(pairs.size());
    #pragma omp parallel for schedule(dynamic)
    for(int q=0;q<(int)pairs.size();++q){
        const int i=pairs[q].first, j=pairs[q].second;
        c[q]=checkDensePair(kd[i],kd[j],s.kfs[i].pose().inverse()*s.kfs[j].pose());
    }
    HeldOutScore h; std::vector<double> off;
    for(const auto& x:c){ if(!x.aligned) continue; ++h.n; if(x.pass()) ++h.pass; off.push_back(x.offsetT*1000); }
    if(!off.empty()){ std::nth_element(off.begin(),off.begin()+off.size()/2,off.end()); h.medianMM=off[off.size()/2]; }
    return h;
}

struct RefineReport { int checked=0, passBefore=0, passAfter=0, roundsKept=0, edgesAdded=0, framesRetracked=0; std::string log; std::vector<FrameCheck> rows; };

static RefineReport refineTrajectory(VioSlam& s,const std::vector<DenseImage>& kd,const FrameImages& img){
    std::cout<<"\n============================================================\n"
               " AUTOMATIC REFINEMENT (re-run where the per-frame check fails)\n"
               "============================================================\n";
    RefineReport rep;
    std::ostringstream log; log<<std::fixed;
    const auto pairs=heldOutPairs(s);
    auto rows=checkFrames(s,kd,img);
    int pass=countPass(rows); HeldOutScore ho=scoreHeldOut(s,kd,pairs);
    rep.checked=(int)rows.size(); rep.passBefore=pass;
    log<<"start      frames pass "<<pass<<"/"<<rows.size()<<"  held-out revisits pass "<<ho.pass<<"/"<<ho.n
       <<", median "<<std::setprecision(2)<<ho.medianMM<<" mm\n";
    std::cout<<log.str();
    const int n=(int)s.kfs.size();

    for(int round=1;round<=REFINE_MAX_ROUNDS;++round){
        // Keyframe intervals that contain failing frames.
        std::map<std::pair<int,int>,const FrameCheck*> bridge;   // (ref, check) -> best re-aligned failing frame
        std::set<std::pair<int,int>> tasks;
        for(const auto& r:rows){
            if(!r.failed()) continue;
            const int a=s.frames[r.frame].refKF, b=r.kf;
            const int lo=std::min(a,b), hi=std::max(a,b);
            tasks.insert({lo,hi});
            if(lo>0) tasks.insert({lo-1,hi});
            if(hi+1<n) tasks.insert({lo,hi+1});
            auto& best=bridge[{a,b}];
            if(r.c.aligned&&r.c.inlierRatio>=VERIFY_MIN_INLIER_RATIO&&(!best||r.c.inlierRatio>best->c.inlierRatio)) best=&r;
        }
        if(tasks.empty()) break;

        // New dense constraints (computed in parallel, added afterwards).
        std::vector<std::pair<int,int>> kk(tasks.begin(),tasks.end());
        std::vector<const FrameCheck*> bf;
        for(const auto& b:bridge) if(b.second) bf.push_back(b.second);
        std::vector<VisualEdge> found(kk.size()+bf.size());
        std::vector<char> ok(found.size(),0);
        #pragma omp parallel for schedule(dynamic)
        for(int q=0;q<(int)found.size();++q){
            if(q<(int)kk.size()){
                // keyframe <-> keyframe, starting from the current estimate
                const int i=kk[q].first, j=kk[q].second;
                const Pose init=(s.kfs[i].pose().inverse()*s.kfs[j].pose()).inverse();
                Pose T=init;
                const DenseResult dr=denseAlign(kd[i],kd[j],T);
                if(!dr.ok||dr.inlierRatio<REVISIT_MIN_INLIER_RATIO) continue;
                const Pose a=init.inverse(), b=T.inverse();
                if((a.t-b.t).norm()>REVISIT_MAX_CORRECTION_T||rotationDeg(a.R,b.R)>REVISIT_MAX_CORRECTION_R) continue;
                found[q].from=i; found[q].to=j; found[q].meas=b; found[q].sqrtInfo=denseSqrtInfo(dr.H,2.0);
            }else{
                // keyframe r <-> keyframe c bridged through frame F:  T_rc = T_rF * T_cF^-1
                const FrameCheck& f=*bf[q-kk.size()];
                const FrameRecord& fr=s.frames[f.frame];
                DenseImage d;
                if(!img.dense(f.frame,d)) continue;
                Pose T=f.c.realigned.inverse();                   // T_cF^-1 (maps keyframe c into F)
                const DenseResult dr=denseAlign(kd[f.kf],d,T);
                if(!dr.ok||dr.inlierRatio<VERIFY_MIN_INLIER_RATIO) continue;
                found[q].from=fr.refKF; found[q].to=f.kf; found[q].meas=fr.T_kf_c*T; found[q].sqrtInfo=denseSqrtInfo(dr.H,2.0);
            }
            found[q].type=EDGE_LOCAL; ok[q]=1;
        }

        const auto kfsBak=s.kfs; const auto lmsBak=s.lms; const auto edgesBak=s.edges; const auto framesBak=s.frames;
        const int localBak=s.localEdges;
        int added=0;
        for(size_t q=0;q<found.size();++q) if(ok[q]){ s.edges.push_back(found[q]); ++added; }
        s.localEdges+=added;
        std::cout<<"[REFINE] round "<<round<<": "<<tasks.size()<<" keyframe links to re-measure, "<<added
                 <<" constraints added, re-optimizing...\n";
        if(!added) break;
        auto t0=std::chrono::steady_clock::now();
        s.reoptimize();
        auto t1=std::chrono::steady_clock::now();
        auto newRows=checkFrames(s,kd,img);
        auto t2=std::chrono::steady_clock::now();
        std::cout<<"[REFINE] re-optimization "<<std::fixed<<std::setprecision(1)<<std::chrono::duration<double>(t1-t0).count()
                 <<" s, check of all frames "<<std::chrono::duration<double>(t2-t1).count()<<" s\n";
        const int newPass=countPass(newRows);
        const HeldOutScore nh=scoreHeldOut(s,kd,pairs);
        const bool keep=newPass>=pass+REFINE_MIN_GAIN&&nh.pass>=ho.pass&&nh.medianMM<=ho.medianMM*REFINE_HELDOUT_TOLERANCE;
        std::ostringstream line; line<<std::fixed;
        line<<"round "<<round<<"    frames pass "<<pass<<" -> "<<newPass<<"  held-out pass "<<ho.pass<<" -> "<<nh.pass
            <<", median "<<std::setprecision(2)<<ho.medianMM<<" -> "<<nh.medianMM<<" mm  (+"<<added<<" constraints)  "
            <<(keep?"KEPT":"ROLLED BACK")<<"\n";
        std::cout<<line.str(); log<<line.str();
        if(!keep){
            s.kfs=kfsBak; s.lms=lmsBak; s.edges=edgesBak; s.frames=framesBak; s.localEdges=localBak;
            break;
        }
        rows.swap(newRows); pass=newPass; ho=nh;
        ++rep.roundsKept; rep.edgesAdded+=added;
    }

    // Frame level: re-track still-failing frames against their other keyframe.
    struct Switch { int frame, newRef, oldRef; Pose T, W; };
    std::vector<Switch> sw(rows.size()); std::vector<char> swOk(rows.size(),0);
    #pragma omp parallel for schedule(dynamic)
    for(int q=0;q<(int)rows.size();++q){
        const FrameCheck& r=rows[q];
        if(!r.failed()||!r.c.aligned) continue;
        const int a=s.frames[r.frame].refKF, c=r.kf;
        const Pose W=s.kfs[c].pose()*r.c.realigned;           // frame pose from keyframe c
        DenseImage d;
        if(!img.dense(r.frame,d)) continue;
        if(!checkDensePair(kd[a],d,s.kfs[a].pose().inverse()*W).pass()) continue;
        sw[q]={r.frame,c,a,r.c.realigned,W}; swOk[q]=1;
    }
    for(size_t q=0;q<sw.size();++q){
        if(!swOk[q]) continue;
        FrameRecord& f=s.frames[sw[q].frame];
        f.refKF=sw[q].newRef; f.T_kf_c=sw[q].T; f.finalPose=sw[q].W; f.checkKF=sw[q].oldRef;
        ++rep.framesRetracked;
    }
    if(rep.framesRetracked>0) rows=checkFrames(s,kd,img);
    rep.passAfter=countPass(rows);
    rep.rows=std::move(rows);
    std::ostringstream fin; fin<<std::fixed;
    fin<<"re-track   "<<rep.framesRetracked<<" failing frames re-tracked against their other keyframe\n"
       <<"result     frames pass "<<rep.passBefore<<" -> "<<rep.passAfter<<" of "<<rep.checked<<"\n";
    std::cout<<fin.str(); log<<fin.str();
    rep.log=log.str();
    return rep;
}

// Report of the final per-frame check (the last check of the refinement).
static void evaluateFrames(const RefineReport& refine,const std::string& dir){
    std::cout<<"\n============================================================\n"
               " PER-FRAME POSE CHECK (dense reprojection, every frame)\n"
               "============================================================\n";
    const auto& rows=refine.rows;
    auto pct=[](std::vector<double> v,double q){ if(v.empty()) return 0.0; std::sort(v.begin(),v.end()); return v[std::min(v.size()-1,(size_t)(q*v.size()))]; };
    std::vector<double> oT,oR,inl,dz,ph; int pass=0, aligned=0, lowOverlap=0;
    std::ofstream fq(dir+"/frame_quality.txt");
    fq<<"# frame checked_against_kf status inlier depth_med_mm photo_rms aligned offset_mm offset_deg result\n"<<std::fixed<<std::setprecision(3);
    std::vector<std::pair<int,int>> badRuns;      // [first,last] frames of consecutive failures
    for(const FrameCheck& r:rows){
        fq<<r.frame<<' '<<r.kf<<' '<<r.status<<' '<<r.c.inlierRatio<<' '<<r.c.depthMedianMM<<' '<<r.c.photoRms<<' '<<r.c.aligned<<' '
          <<r.c.offsetT*1000<<' '<<r.c.offsetR<<' '<<(r.c.pass()?"PASS":(r.lowOverlap()?"LOW_OVERLAP":"CHECK"))<<'\n';
        inl.push_back(r.c.inlierRatio); dz.push_back(r.c.depthMedianMM); ph.push_back(r.c.photoRms);
        if(r.c.aligned){ ++aligned; oT.push_back(r.c.offsetT*1000); oR.push_back(r.c.offsetR); }
        if(r.c.pass()) ++pass;
        else if(r.lowOverlap()) ++lowOverlap;
        else if(!badRuns.empty()&&badRuns.back().second>=r.frame-3) badRuns.back().second=r.frame;
        else badRuns.push_back({r.frame,r.frame});
    }
    std::sort(badRuns.begin(),badRuns.end(),[](const auto& a,const auto& b){return a.second-a.first>b.second-b.first;});

    std::ostringstream rep;
    rep<<std::fixed<<std::setprecision(2)
       <<"frames checked            "<<rows.size()<<" (each against a keyframe it was not tracked against)\n"
       <<"dense re-alignment ok     "<<aligned<<"\n"
       <<"pose offset median / p90  "<<pct(oT,0.5)<<" / "<<pct(oT,0.9)<<" mm,  "<<pct(oR,0.5)<<" / "<<pct(oR,0.9)<<" deg\n"
       <<"residual at final pose    inlier "<<pct(inl,0.5)<<",  depth "<<pct(dz,0.5)<<" mm,  photo RMS "<<pct(ph,0.5)<<"\n"
       <<"PASS                      "<<pass<<" ("<<std::setprecision(1)<<(rows.empty()?0.0:100.0*pass/rows.size())<<"%)  offset <= "
       <<VERIFY_MAX_OFFSET_T*1000<<" mm and <= "<<VERIFY_MAX_OFFSET_R<<" deg, inlier >= "<<VERIFY_MIN_INLIER_RATIO<<"\n"
       <<"low overlap (not judged)  "<<lowOverlap<<"\n"
       <<"failed                    "<<(int)rows.size()-pass-lowOverlap<<"\n";
    if(!badRuns.empty()){
        rep<<"longest failing stretches (frames):";
        for(size_t i=0;i<badRuns.size()&&i<6;++i) rep<<"  "<<badRuns[i].first<<"-"<<badRuns[i].second;
        rep<<"\n";
    }
    rep<<"\nAutomatic refinement:\n"<<refine.log
       <<"\nPer-frame rows: "<<dir<<"/frame_quality.txt\n";
    std::cout<<rep.str();
    std::ofstream(dir+"/frame_quality_summary.txt")<<rep.str();
}

// ============================================================
// MAIN
// ============================================================
int main(int argc,char** argv){
    bool live=false, useImu=true; int maxFrames=-1; std::string recording=DATASET_DIR;
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        if(a=="--live") live=true;
        else if(a=="--no-imu") useImu=false;
        else if(a=="--max-frames"&&i+1<argc) maxFrames=std::atoi(argv[++i]);
        else if(!a.empty()&&a[0]!='-') recording=a;
        else{ std::cerr<<"Usage: vio_slam [recording_dir] [--live] [--max-frames N] [--no-imu]\n"; return 1; }
    }
    const std::string dir=resolveRecording(recording);
    if(dir.empty()){ std::cerr<<"No recording (data.mov) found at '"<<recording<<"' or "<<fs::path(DATASET_DIR).parent_path().string()<<"/"<<recording<<"\n"; return 1; }
    setRecording(dir);
    // Swap-buffer vsync would make every live render wait for the monitor.
    if(live){ setenv("__GL_SYNC_TO_VBLANK","0",0); setenv("vblank_mode","0",0); }
    const std::string outDir=RECORDING_DIR+(useImu?"/slam_results_vio":"/slam_results_vo");
    google::InitGoogleLogging(argv[0]);

    std::cout<<"\n============================================================\n"
               " VISUAL-INERTIAL RGB-D SLAM ("<<(useImu?"camera + IMU":"camera only")<<")\n"
               "============================================================\n";
    PhoneData pd;
    if(!loadPhoneData(pd)){ std::cerr<<"Cannot load phone data.\n"; return 1; }
    auto depthFiles=getDepthFiles();
    int numFrames=std::min((int)depthFiles.size(),(int)pd.frameTime.size());
    {
        cv::VideoCapture cap(VIDEO_PATH);
        if(!cap.isOpened()){ std::cerr<<"Cannot open "<<VIDEO_PATH<<"\n"; return 1; }
        int vf=(int)cap.get(cv::CAP_PROP_FRAME_COUNT);
        if(vf>0) numFrames=std::min(numFrames,vf);
    }
    if(maxFrames>0) numFrames=std::min(numFrames,maxFrames);
    std::cout<<"[INPUT] "<<numFrames<<" frames, output "<<outDir<<(live?", live view":"")<<"\n";
    if(cv::cuda::getCudaEnabledDeviceCount()<=0){ std::cerr<<"No CUDA device.\n"; return 1; }
    cv::cuda::setDevice(0);

    LiveFeed feed;
    auto stepStart=std::chrono::steady_clock::now();
    std::ostringstream timing; timing<<std::fixed<<std::setprecision(1);
    auto lap=[&](const std::string& what){
        const auto now=std::chrono::steady_clock::now();
        const double sec=std::chrono::duration<double>(now-stepStart).count();
        stepStart=now;
        timing<<"  "<<std::left<<std::setw(34)<<what<<std::right<<std::setw(7)<<sec<<" s\n";
        std::cout<<"[TIME] "<<what<<": "<<std::setprecision(1)<<sec<<" s\n";
    };
    auto pipeline=[&]{
        BlockingQueue<RawFrame> queue(6);
        std::thread capture([&]{
            cv::VideoCapture cap(VIDEO_PATH);
            for(int i=0;i<numFrames;++i){
                cv::Mat raw;
                if(!cap.read(raw)||raw.empty()) break;
                RawFrame f; f.index=i;
                cv::resize(raw,f.rgb,cv::Size(RGB_WIDTH,RGB_HEIGHT),0,0,cv::INTER_AREA);
                if(!loadDepth(depthFiles[i],f.depth16)) continue;
                queue.push(std::move(f));
            }
            queue.stop();
        });

        VioSlam slam(pd,useImu);
        FrameImages images; images.gray8.resize(numFrames); images.depthFiles=&depthFiles; images.pd=&pd;
        auto start=std::chrono::steady_clock::now();
        LiveState ls; ls.numFrames=numFrames;
        std::deque<std::pair<int,cv::Mat>> recentRgb;      // colours for keyframes promoted from earlier frames
        size_t publishedKfs=0; int fpsCount=0; auto fpsStart=start;
        auto liveSnapshot=[&](bool final){
            ls.kfPoses.clear(); for(const auto& k:slam.kfs) ls.kfPoses.push_back(k.pose());
            ls.trajectory.clear();
            for(const auto& r:slam.frames)
                if(r.refKF>=0&&r.status!=0) ls.trajectory.push_back((final?r.finalPose:slam.framePose(r)).t);
            ls.keyframes=(int)slam.kfs.size(); ls.loops=slam.loops; ls.imu=slam.imuInitialized;
        };
        RawFrame raw;
        while(queue.pop(raw)){
            FrameData f;
            f.index=raw.index; f.t=pd.frameTime[raw.index]; f.K=pd.frameK[raw.index];
            cv::Mat gray; cv::cvtColor(raw.rgb,gray,cv::COLOR_BGR2GRAY);
            cv::resize(gray,f.gray8,cv::Size(DEPTH_WIDTH,DEPTH_HEIGHT),0,0,cv::INTER_AREA);
            f.depth16=raw.depth16;
            f.dense=buildDenseImage(f.gray8,f.depth16,f.K);
            std::vector<cv::KeyPoint> kps; cv::Mat desc;
            if(slam.orb.compute(gray,kps,desc)) selectFeatures(kps,desc,f.depth16,f.K,f.features,f.desc);
            images.gray8[raw.index]=f.gray8.clone();
            slam.process(std::move(f));

            const FrameRecord& r=slam.frames[raw.index];
            if(raw.index%30==0||r.isKeyframe||r.status!=1||r.loop){
                const Pose T=slam.framePose(r);
                std::cout<<"Frame "<<std::setw(4)<<raw.index<<" | "<<(r.status==1?"DENSE   ":"IMU-ONLY")
                         <<" | inl "<<std::fixed<<std::setprecision(2)<<r.inlierRatio<<" | it "<<std::setw(2)<<r.iterations
                         <<(r.sparseInit?" | ORB-init":"         ")
                         <<" | KF "<<std::setw(4)<<slam.kfs.size()<<(r.isKeyframe?" NEW":"    ")
                         <<" | LM "<<std::setw(6)<<slam.lms.size()
                         <<" | loops "<<slam.loops<<(slam.imuInitialized?" | VIO":" | VO ")
                         <<" | p "<<std::setprecision(3)<<T.t.transpose()
                         <<" | "<<std::setprecision(1)<<r.ms<<" ms\n";
            }

            if(live&&feed.viewerOpen&&r.refKF>=0){
                recentRgb.emplace_back(raw.index,raw.rgb);
                if(recentRgb.size()>90) recentRgb.pop_front();
                std::vector<LiveKeyframe> newKfs;
                for(;publishedKfs<slam.kfs.size();++publishedKfs){
                    const KeyFrame& k=slam.kfs[publishedKfs];
                    cv::Mat col;
                    for(const auto& c:recentRgb) if(c.first==k.frameIndex) col=c.second;
                    if(col.empty()) cv::cvtColor(k.gray8,col,cv::COLOR_GRAY2BGR);
                    newKfs.push_back({k.id,cameraCloud(col,k.depth16,k.K,LIVE_KF_STRIDE)});
                }
                ++fpsCount;
                const double el=std::chrono::duration<double>(std::chrono::steady_clock::now()-fpsStart).count();
                if(el>=1.0){ ls.fps=fpsCount/el; fpsCount=0; fpsStart=std::chrono::steady_clock::now(); }
                ls.frame=raw.index; ls.status=r.status; ls.inlier=r.inlierRatio; ls.isKeyframe=r.isKeyframe;
                ls.rgb=raw.rgb; ls.pose=slam.framePose(r);
                ls.current=cameraCloud(raw.rgb,raw.depth16,pd.frameK[raw.index],LIVE_FRAME_STRIDE);
                liveSnapshot(false);
                feed.publish(ls,std::move(newKfs));
                while(feed.paused&&feed.viewerOpen) std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
        capture.join();
        lap("tracking");
        const double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cout<<"[SLAM] Tracking done in "<<std::fixed<<std::setprecision(1)<<secs<<" s ("<<numFrames/secs<<" FPS)\n";
        auto stage=[&](const std::string& st){ if(live){ ls.stage=st; feed.publish(ls); } };
        stage("final optimization (revisits + global BA)...");
        slam.finish();
        lap("final optimization (revisits + BA)");
        stage("checking every frame + automatic refinement...");
        const std::vector<DenseImage> kd=keyframeDense(slam);
        const RefineReport refine=refineTrajectory(slam,kd,images);
        lap("per-frame check + refinement");
        ls.current.reset(); ls.finished=true; liveSnapshot(true);
        if(!slam.frames.empty()&&slam.frames.back().refKF>=0) ls.pose=slam.frames.back().finalPose;
        stage("verifying keyframe poses...");
        const SavedResults results=collectResults(slam,numFrames);
        fs::create_directories(outDir);
        verifyKeyframePoses(slam,outDir);
        saveResults(outDir,slam,results,pd);
        lap("keyframe verification + save");
        stage("building TSDF scene...");
        buildScene(outDir,results,pd,depthFiles);
        lap("3D scene (TSDF)");
        stage("final per-frame check...");
        evaluateFrames(refine,outDir);
        std::cout<<"\n[TIME] per step:\n"<<timing.str();
        stage("DONE - final optimized map. Mouse to explore, ESC/Q to close");
        std::cout<<"\n[DONE] Trajectory: "<<outDir<<"/trajectory_tum.txt   Scene: "<<outDir<<"/scene_tsdf.ply\n";
        feed.slamDone=true;
    };

    if(!live){ pipeline(); return 0; }
    std::cout<<"[LIVE] SPACE pause | V follow camera / free view | ESC/Q close viewer\n";
    std::thread worker(pipeline);
    runLiveViewer(feed);
    worker.join();
    return 0;
}
