#include "base_kernel_jit_builder.h"

#include <dlfcn.h>
#include <glog/logging.h>
#include <sys/stat.h>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace kernel {

// ============================ JitModule ============================
JitModule::~JitModule() {
    if (handle_) {
        dlclose(handle_);
        handle_ = nullptr;
    }
}

void* JitModule::get_symbol(const std::string& symbol) {
    if (!handle_) return nullptr;
    dlerror();  // clear
    void* sym = dlsym(handle_, symbol.c_str());
    const char* err = dlerror();
    if (err) {
        LOG(ERROR) << "dlsym(" << symbol << ") failed in " << path_ << ": " << err;
        return nullptr;
    }
    return sym;
}

// ============================ JitSpec paths ============================
std::string JitSpec::library_path() const { return cache_dir + "/" + name + ".so"; }
std::string JitSpec::ninja_path() const { return cache_dir + "/build.ninja"; }
std::string JitSpec::lock_path() const { return cache_dir + "/.lock"; }

// ============================ helpers ============================
static void MakeDirs(const std::string& path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if (path[i] == '/' || i + 1 == path.size()) {
            if (!cur.empty() && cur != "/") {
                mkdir(cur.c_str(), 0755);  // ignore EEXIST
            }
        }
    }
}

static bool FileExists(const std::string& path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0;
}

bool WriteIfDifferent(const std::string& path, const std::string& content) {
    if (FileExists(path)) {
        std::ifstream in(path, std::ios::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        if (ss.str() == content) return false;  // unchanged
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << content;
    return true;
}

static std::string Join(const std::vector<std::string>& v, const std::string& sep) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += sep;
        s += v[i];
    }
    return s;
}

// ============================ ninja generation ============================
// 借鉴 flashinfer generate_ninja_build_for_op: 每个源文件一条编译规则,
// 最终链接为共享库。CUDA 源走 nvcc, C++ 源走 cxx。
std::string GenerateNinjaBuild(const JitSpec& spec) {
    const char* cuda_home_env = std::getenv("CUDA_HOME");
    std::string cuda_home = cuda_home_env ? cuda_home_env : "/usr/local/cuda";
    const char* cxx_env = std::getenv("CXX");
    std::string cxx = cxx_env ? cxx_env : "c++";
    const char* nvcc_env = std::getenv("LLMINFER_NVCC");
    std::string nvcc = nvcc_env ? nvcc_env : (cuda_home + "/bin/nvcc");

    std::vector<std::string> include_flags;
    for (const auto& inc : spec.include_dirs) include_flags.push_back("-I" + inc);

    std::string cflags = Join(spec.cxx_flags, " ") + " " + Join(include_flags, " ");
    std::string cuda_cflags =
        Join(spec.cuda_flags, " ") + " " + Join(include_flags, " ") + " -Xcompiler -fPIC";
    std::string ldflags = Join(spec.ld_flags, " ");

    std::ostringstream o;
    o << "ninja_required_version = 1.3\n";
    o << "cuda_home = " << cuda_home << "\n";
    o << "cxx = " << cxx << "\n";
    o << "nvcc = " << nvcc << "\n\n";
    o << "rule compile\n";
    o << "  command = $cxx -MMD -MF $out.d " << cflags << " -fPIC -c $in -o $out\n";
    o << "  depfile = $out.d\n  deps = gcc\n\n";
    o << "rule cuda_compile\n";
    o << "  command = $nvcc --generate-dependencies-with-compile -MF $out.d " << cuda_cflags
      << " -c $in -o $out\n";
    o << "  depfile = $out.d\n  deps = gcc\n\n";
    if (spec.need_device_linking) {
        o << "rule link\n  command = $nvcc -shared $in " << ldflags << " -o $out\n\n";
    } else {
        o << "rule link\n  command = $cxx -shared $in " << ldflags << " -o $out\n\n";
    }

    std::vector<std::string> objects;
    int idx = 0;
    for (const auto& src : spec.sources) {
        bool is_cuda = src.size() >= 3 && src.substr(src.size() - 3) == ".cu";
        std::string obj = spec.cache_dir + "/obj_" + std::to_string(idx++) + ".o";
        objects.push_back(obj);
        o << "build " << obj << ": " << (is_cuda ? "cuda_compile" : "compile") << " " << src
          << "\n";
    }
    o << "\nbuild " << spec.library_path() << ": link " << Join(objects, " ") << "\n";
    return o.str();
}

// ============================ KernelJitCache ============================
KernelJitCache::KernelJitCache() {
    const char* root = std::getenv("LLMINFER_JIT_CACHE");
    cache_root_ = root ? root : (std::string(std::getenv("HOME") ? std::getenv("HOME") : "/tmp") +
                                 "/.llminfer/jit");
}

KernelJitCache& KernelJitCache::instance() {
    static KernelJitCache inst;
    return inst;
}

void KernelJitCache::finalize_paths(JitSpec& spec) const {
    spec.cache_dir = cache_root_ + "/" + spec.name;
}

std::shared_ptr<JitModule> KernelJitCache::load(const std::string& library_path) {
    dlerror();
    void* handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        LOG(ERROR) << "dlopen failed: " << dlerror();
        return nullptr;
    }
    return std::make_shared<JitModule>(handle, library_path);
}

bool KernelJitCache::compile(const JitSpec& spec) {
    MakeDirs(spec.cache_dir);
    std::string ninja = GenerateNinjaBuild(spec);
    WriteIfDifferent(spec.ninja_path(), ninja);

    std::ostringstream cmd;
    cmd << "ninja -f " << spec.ninja_path() << " -C " << spec.cache_dir << " 2>&1";
    LOG(INFO) << "[JIT] compiling " << spec.name << " -> " << spec.library_path();
    int ret = std::system(cmd.str().c_str());
    if (ret != 0) {
        LOG(ERROR) << "[JIT] ninja build failed for " << spec.name << " (code " << ret << ")";
        return false;
    }
    return FileExists(spec.library_path());
}

std::shared_ptr<JitModule> KernelJitCache::build_and_load(JitSpec spec) {
    finalize_paths(spec);
    std::lock_guard<std::mutex> lock(mutex_);

    // 1) 内存缓存
    auto it = loaded_.find(spec.name);
    if (it != loaded_.end()) return it->second;

    // 2) 产物已存在(AOT/上次 JIT)
    if (FileExists(spec.library_path())) {
        auto mod = load(spec.library_path());
        if (mod) {
            loaded_[spec.name] = mod;
            return mod;
        }
    }

    // 3) 在线编译
    if (!compile(spec)) return nullptr;
    auto mod = load(spec.library_path());
    if (mod) loaded_[spec.name] = mod;
    return mod;
}

}  // namespace kernel
