#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <omp.h>
#include <limits>

// Usage: OpenMP_7 <problemSize> <mode> [seed]
// mode: reduction | atomic | critical | lock

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <problemSize> <mode> [seed]\n";
        return 1;
    }

    long long problemSize = static_cast<long long>(std::stoll(argv[1]));
    std::string mode = argv[2];
    unsigned int seed = (argc >= 4) ? static_cast<unsigned int>(std::stoul(argv[3])) : 123456u;

    if (problemSize <= 0) {
        std::cerr << "problemSize must be > 0\n";
        return 2;
    }

    std::vector<double> vectorA;
    std::vector<double> vectorB;
    try {
        vectorA.resize(static_cast<size_t>(problemSize));
        vectorB.resize(static_cast<size_t>(problemSize));
    }
    catch (const std::bad_alloc&) {
        std::cerr << "Failed to allocate vectors of size " << problemSize << "\n";
        return 3;
    }

    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (long long i = 0; i < problemSize; ++i) {
        vectorA[static_cast<size_t>(i)] = distribution(generator);
        vectorB[static_cast<size_t>(i)] = distribution(generator);
    }

    // Warm-up
    {
        volatile double warmSum = 0.0;
        long long warmCount = std::min(problemSize, 1000LL);
        for (long long i = 0; i < warmCount; ++i) {
            warmSum += vectorA[static_cast<size_t>(i)] * vectorB[static_cast<size_t>(i)];
        }
        (void)warmSum;
    }

    int numThreadsReported = omp_get_max_threads();
    double globalSum = 0.0;

    auto startTime = std::chrono::high_resolution_clock::now();

    if (mode == "reduction") {
        #pragma omp parallel for reduction(+:globalSum)
        for (long long i = 0; i < problemSize; ++i) {
            globalSum += vectorA[static_cast<size_t>(i)] * vectorB[static_cast<size_t>(i)];
        }
    }
    else if (mode == "atomic") {
        #pragma omp parallel for
        for (long long i = 0; i < problemSize; ++i) {
            double localValue = vectorA[static_cast<size_t>(i)] * vectorB[static_cast<size_t>(i)];
            #pragma omp atomic
            globalSum += localValue;
        }
    }
    else if (mode == "critical") {
        #pragma omp parallel for
        for (long long i = 0; i < problemSize; ++i) {
            double localValue = vectorA[static_cast<size_t>(i)] * vectorB[static_cast<size_t>(i)];
            #pragma omp critical
            {
                globalSum += localValue;
            }
        }
    }
    else if (mode == "lock") {
        omp_lock_t globalLock;
        omp_init_lock(&globalLock);
        #pragma omp parallel for
        for (long long i = 0; i < problemSize; ++i) {
            double localValue = vectorA[static_cast<size_t>(i)] * vectorB[static_cast<size_t>(i)];
            omp_set_lock(&globalLock);
            globalSum += localValue;
            omp_unset_lock(&globalLock);
        }
        omp_destroy_lock(&globalLock);
    }
    else {
        std::cerr << "Unknown mode: " << mode << " (use reduction|atomic|critical|lock)\n";
        return 4;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double timeSeconds = std::chrono::duration<double>(endTime - startTime).count();

    std::cout << problemSize << "," << numThreadsReported << "," << mode << "," << timeSeconds << "," << globalSum << std::endl;

    return 0;
}
