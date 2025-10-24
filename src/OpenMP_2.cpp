#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <algorithm>
#include <omp.h>

// Usage: OpenMP_2 <problemSize> <mode> [seed]
// mode: reduction | no_reduction

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <problemSize> <mode> [seed]\n";
        return 1;
    }

    const std::size_t problemSize = static_cast<std::size_t>(std::stoull(argv[1]));
    const std::string mode = argv[2];
    const unsigned int seed = (argc >= 4) ? static_cast<unsigned int>(std::stoul(argv[3])) : 12345u;

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
        volatile double warmUpSum = 0.0;
        const std::size_t warmUpLimit = std::min<std::size_t>(problemSize, static_cast<std::size_t>(1000));
        for (std::size_t i = 0; i < warmUpLimit; ++i) {
            warmUpSum += vectorA[i] * vectorB[i];
        }
        (void)warmUpSum;
    }

    const int numThreads = omp_get_max_threads();
    double globalSum = 0.0;

    auto startTime = std::chrono::high_resolution_clock::now();

    if (mode == "reduction") {
        #pragma omp parallel for reduction(+:globalSum)
        for (std::size_t i = 0; i < problemSize; ++i) {
            globalSum += vectorA[i] * vectorB[i];
        }
    }
    else if (mode == "no_reduction") {
        #pragma omp parallel
        {
            double localSum = 0.0;
            #pragma omp for
            for (std::size_t i = 0; i < problemSize; ++i) {
                localSum += vectorA[i] * vectorB[i];
            }

            #pragma omp critical
            {
                globalSum += localSum;
            }
        }
    }
    else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 2;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double timeSeconds = std::chrono::duration<double>(endTime - startTime).count();

    std::cout << problemSize << "," << numThreads << "," << mode << "," << timeSeconds << "," << globalSum << std::endl;
    return 0;
}
