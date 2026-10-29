// ============================================================
// scene_player.cpp
//
// Fixed-rate (30 FPS) playback of the phone RGB-D recording with the final
// VIO trajectory:
//   - "RGB" window      : the incoming colour frame
//   - "3D Scene" window : the scene growing frame by frame, the blue camera
//                         trajectory and the red camera frustum, with the 3D
//                         view locked to the camera (1 m behind it).
//
// A decode thread reads video + depth and back-projects the new 1 cm voxels
// of each frame; the render thread only uploads that frame's new points, so
// the frame rate stays at 30 FPS as the scene grows.
//
// Mesh mode (--mesh): instead of the growing point cloud, the finished mesh
// from map_builder (reconstruction/room_mesh_light.ply) is shown and the camera
// moves through it.
//
// Run vio_slam first (it writes the trajectory read here); map_builder
// before --mesh.
//
// Usage: scene_player [recording_dir] [--mesh [FILE.ply]] [--fps F] [--start N] [--trajectory FILE]
// Playback starts paused on the first frame (arrange the windows, e.g. for a
// screen recording, then press SPACE).
// Keys (either window): SPACE start / pause, V follow camera / free mouse view,
//                       ESC or Q quit.
// ============================================================

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <vtkActor.h>
#include <vtkCellArray.h>
#include <vtkPLYReader.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>

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
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

// ============================================================
// SETTINGS
// ============================================================
static const std::string DATASET_DIR =
    "/home/mohit/3D/recording_2026-09-17_13-13-55";
// Paths of the recording being played (default DATASET_DIR, or the
// recording folder given on the command line).
static std::string VIDEO_PATH, DEPTH_DIR, DATA_JSONL_PATH, CALIBRATION_PATH, TRAJECTORY_PATH, MESH_PATH;
// Recording folder from the command line: a path (absolute or relative to
// the current directory) or just the folder name, looked up next to DATASET_DIR.
static std::string resolveRecording(const std::string& arg){
    std::error_code ec;
    for(const fs::path& p:{fs::path(arg),fs::path(DATASET_DIR).parent_path()/arg})
        if(fs::exists(p/"data.mov",ec)) return fs::canonical(p,ec).string();
    return "";
}

static void setRecording(std::string dir){
    while(dir.size()>1&&dir.back()=='/') dir.pop_back();
    VIDEO_PATH=dir+"/data.mov";
    DEPTH_DIR=dir+"/frames2";
    DATA_JSONL_PATH=dir+"/data.jsonl";
    CALIBRATION_PATH=dir+"/calibration.json";
    TRAJECTORY_PATH=dir+"/slam_results_vio/trajectory_all_frames.txt";
    MESH_PATH=dir+"/reconstruction/room_mesh_light.ply";
}

static constexpr int DEPTH_WIDTH  = 256;
static constexpr int DEPTH_HEIGHT = 192;
static constexpr int RGB_WIDTH    = 512;
static constexpr int RGB_HEIGHT   = 384;
static constexpr float DEPTH_SCALE = 0.001f;
static constexpr float MIN_DEPTH = 0.15f;
static constexpr float MAX_DEPTH = 3.0f;
static constexpr float FX = 179.424f, FY = 179.424f, CX = 128.852f, CY = 95.357f;

static constexpr int MAP_STRIDE = 2;                 // depth pixels
static constexpr float MAP_VOXEL = 0.010f;           // one point per 1 cm voxel
static constexpr float MAP_EDGE_JUMP = 0.05f;        // relative depth jump = flying pixel
static constexpr int CHUNK_MAX_POINTS = 40000;       // points per GPU cloud before freezing it
static constexpr int QUEUE_FRAMES = 60;              // 2 s of decoded frames
static constexpr int PREBUFFER_FRAMES = 30;
static constexpr double RECORDING_FPS = 30.0;        // phone video rate (time shown in the HUD)
static constexpr double FOLLOW_DISTANCE = 1.0;       // m behind the camera

// ============================================================
// BASIC TYPES
// ============================================================
struct Pose {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Vector3d transform(const Eigen::Vector3d& p) const { return R*p+t; }
};

struct Intrinsics { float fx=FX, fy=FY, cx=CX, cy=CY; };

struct VoxelKey { int x,y,z; bool operator==(const VoxelKey& o)const{return x==o.x&&y==o.y&&z==o.z;} };
struct VoxelHash { size_t operator()(const VoxelKey& k)const{ return ((size_t)(uint32_t)k.x*73856093u)^((size_t)(uint32_t)k.y*19349663u)^((size_t)(uint32_t)k.z*83492791u); } };

template<typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t n):max_(n){}
    void push(T x){ std::unique_lock<std::mutex> l(m_); full_.wait(l,[&]{return q_.size()<max_||stop_;}); if(stop_) return; q_.push_back(std::move(x)); empty_.notify_one(); }
    bool pop(T& x){ std::unique_lock<std::mutex> l(m_); empty_.wait(l,[&]{return !q_.empty()||stop_;}); if(q_.empty()) return false; x=std::move(q_.front()); q_.pop_front(); full_.notify_one(); return true; }
    size_t size(){ std::lock_guard<std::mutex> l(m_); return q_.size(); }
    void finish(){ std::lock_guard<std::mutex> l(m_); done_=true; stop_=true; empty_.notify_all(); full_.notify_all(); }
    void abort(){ { std::lock_guard<std::mutex> l(m_); stop_=true; q_.clear(); } empty_.notify_all(); full_.notify_all(); }
    bool done(){ std::lock_guard<std::mutex> l(m_); return done_; }
private:
    std::deque<T> q_; size_t max_; std::mutex m_; std::condition_variable empty_,full_; bool stop_=false, done_=false;
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
static std::vector<Intrinsics> loadIntrinsics(){
    double imageW=1920.0, imageH=1440.0;
    {
        std::ifstream f(CALIBRATION_PATH);
        if(f.is_open()){
            std::stringstream ss; ss<<f.rdbuf(); std::string s=ss.str(); double v;
            if(jsonNumber(s,"imageWidth",v)&&v>0) imageW=v;
            if(jsonNumber(s,"imageHeight",v)&&v>0) imageH=v;
        }
    }
    const double sx=DEPTH_WIDTH/imageW, sy=DEPTH_HEIGHT/imageH;
    std::vector<Intrinsics> ks;
    std::ifstream f(DATA_JSONL_PATH);
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
            if(!found||camInd==1){ best=k; found=true; }
        }
        if((int)num>=(int)ks.size()) ks.resize((int)num+1);
        if(found) ks[(int)num]=best;
    }
    return ks;
}

struct Trajectory { std::vector<Pose> pose; std::vector<int> status; };

// trajectory_all_frames.txt: frame_id timestamp tx ty tz qx qy qz qw status ...
static bool loadTrajectory(const std::string& path,Trajectory& tr){
    std::ifstream in(path);
    if(!in.is_open()) return false;
    std::string line;
    while(std::getline(in,line)){
        if(line.empty()||line[0]=='#') continue;
        std::istringstream ss(line); int id,st=1; double ts,tx,ty,tz,qx,qy,qz,qw;
        if(!(ss>>id>>ts>>tx>>ty>>tz>>qx>>qy>>qz>>qw)||id<0) continue;
        if(!(ss>>st)) st=1;
        if(id>=(int)tr.pose.size()){ tr.pose.resize(id+1); tr.status.resize(id+1,0); }
        tr.pose[id].R=Eigen::Quaterniond(qw,qx,qy,qz).normalized().toRotationMatrix();
        tr.pose[id].t=Eigen::Vector3d(tx,ty,tz);
        tr.status[id]=st;
    }
    return !tr.pose.empty();
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

// ============================================================
// DECODE THREAD: frame -> RGB image + the new map points it contributes
// ============================================================
struct PlayFrame {
    int index=-1; bool tracked=false;
    cv::Mat rgb;
    Pose pose;
    std::vector<pcl::PointXYZRGB> newPoints;
};

class SceneBuilder {
public:
    // New 1 cm voxels seen by this frame (flying pixels at depth edges skipped).
    void addFrame(const cv::Mat& rgb,const cv::Mat& depth16,const Intrinsics& K,const Pose& T,std::vector<pcl::PointXYZRGB>& out){
        auto Z=[&](int v,int u){ return depth16.at<uint16_t>(v,u)*DEPTH_SCALE; };
        for(int v=0;v<depth16.rows;v+=MAP_STRIDE) for(int u=0;u<depth16.cols;u+=MAP_STRIDE){
            const float z=Z(v,u);
            if(z<MIN_DEPTH||z>MAX_DEPTH) continue;
            if(u+MAP_STRIDE<depth16.cols&&std::abs(Z(v,u+MAP_STRIDE)-z)>MAP_EDGE_JUMP*z) continue;
            if(v+MAP_STRIDE<depth16.rows&&std::abs(Z(v+MAP_STRIDE,u)-z)>MAP_EDGE_JUMP*z) continue;
            const Eigen::Vector3d q=T.transform(Eigen::Vector3d((u-K.cx)*z/K.fx,(v-K.cy)*z/K.fy,z));
            VoxelKey key{(int)std::floor(q.x()/MAP_VOXEL),(int)std::floor(q.y()/MAP_VOXEL),(int)std::floor(q.z()/MAP_VOXEL)};
            if(!occupied.insert(key).second) continue;
            const cv::Vec3b c=rgb.at<cv::Vec3b>(std::min(RGB_HEIGHT-1,v*RGB_HEIGHT/DEPTH_HEIGHT),std::min(RGB_WIDTH-1,u*RGB_WIDTH/DEPTH_WIDTH));
            pcl::PointXYZRGB p; p.x=(float)q.x(); p.y=(float)q.y(); p.z=(float)q.z(); p.b=c[0]; p.g=c[1]; p.r=c[2];
            out.push_back(p);
        }
    }
private:
    std::unordered_set<VoxelKey,VoxelHash> occupied;
};

// ============================================================
// 3D WINDOW
// ============================================================
class SceneWindow {
public:
    pcl::visualization::PCLVisualizer::Ptr viewer;
    std::atomic<bool> pauseToggled{false}, followToggled{false};

    SceneWindow(){
        viewer.reset(new pcl::visualization::PCLVisualizer("3D Scene"));
        viewer->setBackgroundColor(0.05,0.05,0.05); viewer->setSize(1400,900); viewer->initCameraParameters();
        viewer->setShowFPS(false);                         // offline playback: no render-rate counter
        viewer->setCameraPosition(0,0,-2,0,0,0,0,-1,0);
        viewer->registerKeyboardCallback([this](const pcl::visualization::KeyboardEvent& e){
            if(!e.keyDown()) return;
            if(e.getKeySym()=="space") pauseToggled=true;
            else if(e.getKeySym()=="v"||e.getKeySym()=="V") followToggled=true;
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
        newChunk();
    }

    // Only the active chunk is re-uploaded; full chunks stay on the GPU untouched.
    void addPoints(const std::vector<pcl::PointXYZRGB>& pts){
        if(pts.empty()) return;
        if(chunk->size()+pts.size()>CHUNK_MAX_POINTS&&!chunk->empty()) newChunk();
        chunk->points.insert(chunk->points.end(),pts.begin(),pts.end());
        chunk->width=(uint32_t)chunk->size(); chunk->height=1;
        totalPoints+=pts.size();
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(chunk);
        if(!chunkAdded){
            viewer->addPointCloud<pcl::PointXYZRGB>(chunk,rgb,chunkId());
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,2,chunkId());
            chunkAdded=true;
        }else viewer->updatePointCloud<pcl::PointXYZRGB>(chunk,rgb,chunkId());
    }

    // Static coloured mesh as the scene. Back faces are culled, so walls seen
    // from behind (the follow camera can sit behind a wall) do not block the view.
    bool loadMesh(const std::string& path){
        auto reader=vtkSmartPointer<vtkPLYReader>::New();
        reader->SetFileName(path.c_str());
        reader->Update();
        vtkPolyData* poly=reader->GetOutput();
        if(!poly||poly->GetNumberOfPolys()==0) return false;
        meshTriangles=(size_t)poly->GetNumberOfPolys();
        auto mapper=vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputData(poly);
        mapper->SetScalarModeToUsePointData();
        mapper->SetColorModeToDirectScalars();
        auto actor=vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->GetProperty()->BackfaceCullingOn();
        actor->GetProperty()->SetAmbient(0.35);
        actor->GetProperty()->SetDiffuse(0.75);
        viewer->getRendererCollection()->GetFirstRenderer()->AddActor(actor);
        return true;
    }

    void addTrajectoryPoint(const Eigen::Vector3d& p){
        trajPts->InsertNextPoint(p.x(),p.y(),p.z());
        const vtkIdType n=trajPts->GetNumberOfPoints();
        if(n>=2){ vtkIdType ids[2]={n-2,n-1}; trajLines->InsertNextCell(2,ids); }
        trajPts->Modified(); trajLines->Modified(); traj->Modified();
    }

    void setCamera(const Pose& pose,bool follow){
        const double z=.20,w=.12,h=.09;
        const Eigen::Vector3d c[5]={Eigen::Vector3d::Zero(),{-w,-h,z},{w,-h,z},{w,h,z},{-w,h,z}};
        for(int i=0;i<5;++i){ const Eigen::Vector3d p=pose.transform(c[i]); frustumPts->SetPoint(i,p.x(),p.y(),p.z()); }
        frustumPts->Modified(); frustum->Modified();
        if(!follow) return;
        const Eigen::Vector3d fwd=pose.R*Eigen::Vector3d(0,0,1), up=pose.R*Eigen::Vector3d(0,-1,0);
        const Eigen::Vector3d pos=pose.t-fwd*FOLLOW_DISTANCE, look=pose.t;
        viewer->setCameraPosition(pos.x(),pos.y(),pos.z(),look.x(),look.y(),look.z(),up.x(),up.y(),up.z());
    }

    void setHud(const std::string& s){ viewer->updateText(s,10,10,14,1,1,1,"hud"); }
    void render(){ viewer->spinOnce(1,true); }
    bool stopped(){ return viewer->wasStopped(); }
    size_t points() const { return totalPoints; }
    size_t meshTriangles=0;
    std::string sceneInfo() const {
        return meshTriangles?"mesh "+std::to_string(meshTriangles/1000)+"k triangles":"points "+std::to_string(totalPoints);
    }

private:
    vtkSmartPointer<vtkPoints> trajPts, frustumPts;
    vtkSmartPointer<vtkCellArray> trajLines;
    vtkSmartPointer<vtkPolyData> traj, frustum;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr chunk;
    int chunkIndex=-1; bool chunkAdded=false; size_t totalPoints=0;

    std::string chunkId() const { return "scene_"+std::to_string(chunkIndex); }
    void newChunk(){ chunk.reset(new pcl::PointCloud<pcl::PointXYZRGB>); ++chunkIndex; chunkAdded=false; }
};

// ============================================================
// MAIN
// ============================================================
int main(int argc,char** argv){
    double fps=30.0; int start=0; std::string recording=DATASET_DIR, trajPath, meshPath; bool meshMode=false;
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        if(a=="--fps"&&i+1<argc) fps=std::max(1.0,std::atof(argv[++i]));
        else if(a=="--start"&&i+1<argc) start=std::max(0,std::atoi(argv[++i]));
        else if(a=="--trajectory"&&i+1<argc) trajPath=argv[++i];
        else if(a=="--mesh"){ meshMode=true; if(i+1<argc&&std::string(argv[i+1]).size()>4&&std::string(argv[i+1]).substr(std::string(argv[i+1]).size()-4)==".ply") meshPath=argv[++i]; }
        else if(!a.empty()&&a[0]!='-') recording=a;
        else{ std::cerr<<"Usage: scene_player [recording_dir] [--mesh [FILE.ply]] [--fps F] [--start N] [--trajectory FILE]\n"; return 1; }
    }
    const std::string dir=resolveRecording(recording);
    if(dir.empty()){ std::cerr<<"No recording (data.mov) found at '"<<recording<<"' or "<<fs::path(DATASET_DIR).parent_path().string()<<"/"<<recording<<"\n"; return 1; }
    setRecording(dir);
    if(trajPath.empty()) trajPath=TRAJECTORY_PATH;
    if(meshPath.empty()) meshPath=MESH_PATH;
    // Swap-buffer vsync makes each VTK render wait for the monitor (two renders per
    // frame -> 20 FPS); playback is paced by our own 30 FPS clock instead.
    setenv("__GL_SYNC_TO_VBLANK","0",0);
    setenv("vblank_mode","0",0);

    Trajectory tr;
    if(!loadTrajectory(trajPath,tr)){ std::cerr<<"Cannot read "<<trajPath<<" (run vio_slam first)\n"; return 1; }
    const auto Ks=loadIntrinsics();
    const auto depthFiles=getDepthFiles();
    cv::VideoCapture probe(VIDEO_PATH);
    if(!probe.isOpened()){ std::cerr<<"Cannot open "<<VIDEO_PATH<<"\n"; return 1; }
    int numFrames=std::min((int)depthFiles.size(),(int)tr.pose.size());
    const int vf=(int)probe.get(cv::CAP_PROP_FRAME_COUNT);
    if(vf>0) numFrames=std::min(numFrames,vf);
    probe.release();
    std::cout<<"[PLAYER] "<<numFrames<<" frames at "<<fps<<" FPS, trajectory "<<trajPath<<"\n"
             <<"[PLAYER] SPACE pause | V follow camera / free view | ESC/Q quit\n";

    BlockingQueue<PlayFrame> queue(QUEUE_FRAMES);
    std::thread decoder([&]{
        cv::VideoCapture cap(VIDEO_PATH);
        SceneBuilder builder;
        for(int i=0;i<numFrames;++i){
            cv::Mat raw;
            if(!cap.read(raw)||raw.empty()) break;
            if(i<start) continue;
            PlayFrame f; f.index=i; f.pose=tr.pose[i]; f.tracked=tr.status[i]==1;
            cv::resize(raw,f.rgb,cv::Size(RGB_WIDTH,RGB_HEIGHT),0,0,cv::INTER_AREA);
            if(meshMode){ queue.push(std::move(f)); continue; }        // the mesh is the scene
            cv::Mat depth=cv::imread(depthFiles[i],cv::IMREAD_UNCHANGED);
            if(f.tracked&&depth.type()==CV_16UC1&&depth.cols==DEPTH_WIDTH&&depth.rows==DEPTH_HEIGHT)
                builder.addFrame(f.rgb,depth,i<(int)Ks.size()?Ks[i]:Intrinsics(),f.pose,f.newPoints);
            queue.push(std::move(f));
        }
        queue.finish();
    });

    SceneWindow scene;
    if(meshMode){
        std::cout<<"[PLAYER] loading mesh "<<meshPath<<" ..."<<std::endl;
        if(!scene.loadMesh(meshPath)){ std::cerr<<"Cannot read "<<meshPath<<" (run map_builder first)\n"; queue.abort(); decoder.join(); return 1; }
        std::cout<<"[PLAYER] mesh: "<<scene.meshTriangles<<" triangles"<<std::endl;
    }
    cv::namedWindow("RGB",cv::WINDOW_NORMAL); cv::resizeWindow("RGB",768,576);
    while(queue.size()<PREBUFFER_FRAMES&&!queue.done()) std::this_thread::sleep_for(std::chrono::milliseconds(10));

    const auto period=std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0/fps));
    auto deadline=std::chrono::steady_clock::now(), fpsStart=deadline;
    int fpsCount=0, late=0; double measured=0;
    // Start paused on the first frame so the windows can be arranged (e.g. for
    // a screen recording) before playback begins.
    bool paused=true, waitingForStart=true, follow=true, quit=false;
    PlayFrame f;
    auto handleKeys=[&](int k){
        if(scene.stopped()||k==27||k=='q'||k=='Q') quit=true;
        if(k==' '||scene.pauseToggled.exchange(false)) paused=!paused;
        if(k=='v'||k=='V'||scene.followToggled.exchange(false)) follow=!follow;
    };

    while(!quit&&queue.pop(f)){
        scene.addPoints(f.newPoints);
        if(f.tracked) scene.addTrajectoryPoint(f.pose.t);
        scene.setCamera(f.pose,follow);

        ++fpsCount;
        auto now=std::chrono::steady_clock::now();
        const double el=std::chrono::duration<double>(now-fpsStart).count();
        if(el>=1.0){ measured=fpsCount/el; fpsCount=0; fpsStart=now; }
        if(f.index%300==0) std::cout<<"[PLAYER] frame "<<f.index<<"  FPS "<<std::fixed<<std::setprecision(1)<<measured
                                    <<"  "<<scene.sceneInfo()<<"  late frames "<<late<<"  queued "<<queue.size()<<std::endl;

        std::ostringstream hud;
        hud<<"Frame "<<f.index<<"/"<<numFrames-1<<"  t "<<std::fixed<<std::setprecision(1)<<f.index/RECORDING_FPS<<" s  "
           <<scene.sceneInfo()<<(follow?"  [follow]":"  [free view]");
        scene.setHud(hud.str());
        cv::Mat d=f.rgb.clone();
        cv::putText(d,hud.str(),cv::Point(10,22),cv::FONT_HERSHEY_SIMPLEX,.5,cv::Scalar(0,255,0),1,cv::LINE_AA);
        if(!f.tracked) cv::putText(d,"NOT TRACKED",cv::Point(10,44),cv::FONT_HERSHEY_SIMPLEX,.5,cv::Scalar(0,165,255),1,cv::LINE_AA);
        cv::imshow("RGB",d);
        scene.render();
        handleKeys(cv::waitKey(1));

        if(paused){
            if(waitingForStart){
                const std::string msg="Arrange the windows, then press SPACE to start";
                scene.setHud(hud.str()+"   "+msg);
                cv::putText(d,msg,cv::Point(10,d.rows-16),cv::FONT_HERSHEY_SIMPLEX,.6,cv::Scalar(0,255,255),2,cv::LINE_AA);
                cv::imshow("RGB",d);
                std::cout<<"[PLAYER] "<<msg<<std::endl;
                waitingForStart=false;
            }else scene.setHud(hud.str()+"  [PAUSED]");
            while(paused&&!quit){ scene.render(); handleKeys(cv::waitKey(20)); }
            deadline=std::chrono::steady_clock::now();
        }

        deadline+=period; now=std::chrono::steady_clock::now();
        if(now<deadline) std::this_thread::sleep_until(deadline);
        else{ ++late; deadline=now; }
    }
    if(quit) queue.abort();
    decoder.join();
    std::cout<<"[PLAYER] Done. "<<scene.sceneInfo()<<", "<<late<<" frames rendered late.\n";

    if(!quit){
        std::cout<<"[PLAYER] Playback finished: explore with the mouse, ESC/Q or close to exit.\n";
        scene.setHud("Finished - "+scene.sceneInfo()+". Mouse to explore, ESC/Q to exit.");
        while(!quit){ scene.viewer->spinOnce(15,true); int k=cv::waitKey(15); if(scene.stopped()||k==27||k=='q'||k=='Q') quit=true; }
    }
    return 0;
}
