#include "llama-mem-telemetry.h"

#include "ggml-backend.h"

#include <algorithm>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <fstream>
#include <sstream>
#include <string>
#endif

llama_mem_telemetry::llama_mem_telemetry()
    : cfg_{}
    , last_refresh_(std::chrono::steady_clock::now() - std::chrono::hours(1)) {
}

llama_mem_telemetry::llama_mem_telemetry(const config & cfg)
    : cfg_(cfg)
    , last_refresh_(std::chrono::steady_clock::now() - std::chrono::hours(1)) {
}

bool llama_mem_telemetry::refresh(const std::vector<ggml_backend_dev_t> & devices) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto now = std::chrono::steady_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(now - last_refresh_).count();

    // Check if we should refresh based on interval
    if (cache_valid_ && elapsed_us < cfg_.refresh_interval_us) {
        return false;
    }

    // Refresh device stats
    for (ggml_backend_dev_t dev : devices) {
        if (dev == nullptr) {
            continue;
        }

        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free, &total);

        llama_mem_stats stats;
        stats.total_bytes = total;
        stats.free_bytes = free;
        stats.allocated_bytes = total > free ? total - free : 0;
        stats.update_utilization();

        device_stats_[device_key(dev)] = stats;
    }

    // Refresh CPU stats
    refresh_cpu_stats();

    last_refresh_ = now;
    cache_valid_ = true;

    return true;
}

bool llama_mem_telemetry::is_under_pressure(ggml_backend_dev_t dev) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = device_stats_.find(device_key(dev));
    if (it == device_stats_.end()) {
        return false;
    }

    return it->second.utilization_pct >= cfg_.pressure_threshold * 100.0f;
}

bool llama_mem_telemetry::is_critical(ggml_backend_dev_t dev) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = device_stats_.find(device_key(dev));
    if (it == device_stats_.end()) {
        return false;
    }

    return it->second.utilization_pct >= cfg_.critical_threshold * 100.0f;
}

size_t llama_mem_telemetry::available_bytes(ggml_backend_dev_t dev) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = device_stats_.find(device_key(dev));
    if (it == device_stats_.end()) {
        return 0;
    }

    return it->second.free_bytes;
}

size_t llama_mem_telemetry::total_bytes(ggml_backend_dev_t dev) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = device_stats_.find(device_key(dev));
    if (it == device_stats_.end()) {
        return 0;
    }

    return it->second.total_bytes;
}

bool llama_mem_telemetry::get_stats(ggml_backend_dev_t dev, llama_mem_stats & stats) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = device_stats_.find(device_key(dev));
    if (it == device_stats_.end()) {
        return false;
    }

    stats = it->second;
    return true;
}

llama_mem_stats llama_mem_telemetry::get_cpu_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cpu_stats_;
}

void llama_mem_telemetry::set_config(const config & cfg) {
    std::lock_guard<std::mutex> lock(mutex_);
    cfg_ = cfg;
}

const llama_mem_telemetry::config & llama_mem_telemetry::get_config() const {
    return cfg_;
}

void llama_mem_telemetry::invalidate_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_valid_ = false;
}

void llama_mem_telemetry::refresh_cpu_stats() {
    // Get system memory stats based on platform
#if defined(_WIN32)
    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(MEMORYSTATUSEX);
    if (GlobalMemoryStatusEx(&mem_info)) {
        cpu_stats_.total_bytes = static_cast<size_t>(mem_info.ullTotalPhys);
        cpu_stats_.free_bytes = static_cast<size_t>(mem_info.ullAvailPhys);
        cpu_stats_.allocated_bytes = cpu_stats_.total_bytes - cpu_stats_.free_bytes;
        cpu_stats_.update_utilization();
    }
#elif defined(__APPLE__)
    // Get total physical memory
    int64_t total_mem = 0;
    size_t size = sizeof(total_mem);
    if (sysctlbyname("hw.memsize", &total_mem, &size, NULL, 0) == 0) {
        cpu_stats_.total_bytes = static_cast<size_t>(total_mem);
    }

    // Get free memory using vm_statistics
    vm_size_t page_size;
    mach_port_t mach_port = mach_host_self();
    host_page_size(mach_port, &page_size);

    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = sizeof(vm_stats) / sizeof(natural_t);
    if (host_statistics64(mach_port, HOST_VM_INFO64, (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
        // Free memory = free pages + inactive pages (can be reclaimed)
        uint64_t free_pages = vm_stats.free_count + vm_stats.inactive_count;
        cpu_stats_.free_bytes = static_cast<size_t>(free_pages * page_size);
        cpu_stats_.allocated_bytes = cpu_stats_.total_bytes - cpu_stats_.free_bytes;
        cpu_stats_.update_utilization();
    }
#elif defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        size_t mem_total = 0;
        size_t mem_available = 0;
        bool got_total = false;
        bool got_available = false;

        while (std::getline(meminfo, line) && (!got_total || !got_available)) {
            std::istringstream iss(line);
            std::string key;
            size_t value;
            std::string unit;

            if (iss >> key >> value >> unit) {
                if (key == "MemTotal:") {
                    mem_total = value * 1024;  // Convert from KB to bytes
                    got_total = true;
                } else if (key == "MemAvailable:") {
                    mem_available = value * 1024;
                    got_available = true;
                }
            }
        }

        if (got_total && got_available) {
            cpu_stats_.total_bytes = mem_total;
            cpu_stats_.free_bytes = mem_available;
            cpu_stats_.allocated_bytes = mem_total - mem_available;
            cpu_stats_.update_utilization();
        }
    }
#else
    // Fallback: no platform support, leave stats at zero
    cpu_stats_ = llama_mem_stats{};
#endif
}

uintptr_t llama_mem_telemetry::device_key(ggml_backend_dev_t dev) {
    return reinterpret_cast<uintptr_t>(dev);
}
