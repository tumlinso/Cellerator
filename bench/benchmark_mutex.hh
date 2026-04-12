#pragma once

#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace cellerator::bench {

class benchmark_mutex_guard {
public:
    explicit benchmark_mutex_guard(const char *name = "gpu-bench") {
        const char *path = std::getenv("CUDA_V100_BENCHMARK_MUTEX_PATH");
        if (path == nullptr || *path == '\0') path = "/tmp/cuda_v100_benchmark.lock";
        fd_ = ::open(path, O_CREAT | O_RDWR, 0666);
        if (fd_ < 0) {
            throw std::runtime_error(std::string("failed to open benchmark mutex: ") + std::strerror(errno));
        }
        if (::flock(fd_, LOCK_EX) != 0) {
            const int saved_errno = errno;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error(std::string("failed to lock benchmark mutex: ") + std::strerror(saved_errno));
        }
        const char *tag = name != nullptr ? name : "gpu-bench";
        std::fprintf(stderr, "[benchmark-mutex] acquired %s via %s\n", tag, path);
    }

    ~benchmark_mutex_guard() {
        if (fd_ >= 0) {
            ::flock(fd_, LOCK_UN);
            ::close(fd_);
        }
    }

    benchmark_mutex_guard(const benchmark_mutex_guard &) = delete;
    benchmark_mutex_guard &operator=(const benchmark_mutex_guard &) = delete;

private:
    int fd_ = -1;
};

} // namespace cellerator::bench
