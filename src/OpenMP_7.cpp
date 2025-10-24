#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <omp.h>
#include <limits>
#include <algorithm>

// Usage: OpenMP_7 <problemSize> <mode> [seed]
// mode: reduction | atomic | critical | lock

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <problemSize> <mode> [seed]\n";
        return 1;
    }

    const std::size_t problemSize = static_cast<std::size_t>(std::stoull(argv[1]));
    const std::string mode = argv[2];
    const unsigned int seed = (argc >= 4) ? static_cast<unsigned int>(std::stoul(argv[3])) : 123456u;

    if (problemSize == 0) {
        std::cerr << "problemSize must be > 0\n";
        return 2;
    }

    std::vector<double> vectorA(problemSize);
    std::vector<double> vectorB(problemSize);

    std::mt19937_64 generator(static_cast<unsigned long long>(seed));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    for (std::size_t i = 0; i < problemSize; ++i) {
        vectorA[i] = distribution(generator);
        vectorB[i] = distribution(generator);
    }

    // Warm-up
    {
        volatile double warmSum = 0.0;
        const std::size_t warmCount = std::min<std::size_t>(problemSize, static_cast<std::size_t>(1000));
        for (std::size_t i = 0; i < warmCount; ++i) {
            warmSum += vectorA[i] * vectorB[i];
        }
        (void)warmSum;
    }

    const int numThreadsReported = omp_get_max_threads();
    double globalSum = 0.0;

    auto startTime = std::chrono::high_resolution_clock::now();

    if (mode == "reduction") {
        #pragma omp parallel for reduction(+:globalSum)
        for (std::size_t i = 0; i < problemSize; ++i) {
            globalSum += vectorA[i] * vectorB[i];
        }
    }
    else if (mode == "atomic") {
        #pragma omp parallel for
        for (std::size_t i = 0; i < problemSize; ++i) {
            double localValue = vectorA[i] * vectorB[i];
            #pragma omp atomic
            globalSum += localValue;
        }
    }
    else if (mode == "critical") {
        #pragma omp parallel for
        for (std::size_t i = 0; i < problemSize; ++i) {
            double localValue = vectorA[i] * vectorB[i];
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
        for (std::size_t i = 0; i < problemSize; ++i) {
            double localValue = vectorA[i] * vectorB[i];
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
