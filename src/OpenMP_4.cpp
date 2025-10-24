#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <string>
#include <algorithm>
#include <omp.h>

// Usage: OpenMP_4 <matrixSize> <mode> [seed]
// mode: reduction | no_reduction

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrixSize> <mode> [seed]\n";
        return 1;
    }

    const std::size_t matrixSize = static_cast<std::size_t>(std::stoull(argv[1]));
    const std::string mode = argv[2];
    const unsigned int seed = (argc >= 4) ? static_cast<unsigned int>(std::stoul(argv[3])) : 12345u;

    if (matrixSize == 0) {
        std::cerr << "matrixSize must be > 0\n";
        return 2;
    }

    std::vector<double> matrixData;
    matrixData.resize(matrixSize * matrixSize);

    std::mt19937_64 generator(static_cast<unsigned long long>(seed));
    std::uniform_real_distribution<double> distribution(0.0, 1.0e6);

    for (std::size_t i = 0; i < matrixSize; ++i) {
        const std::size_t rowOffset = i * matrixSize;
        for (std::size_t j = 0; j < matrixSize; ++j) {
            matrixData[rowOffset + j] = distribution(generator);
        }
    }

    // Warm-up
    {
        volatile double warmUpSum = 0.0;
        const std::size_t warmSteps = std::min<std::size_t>(matrixSize, static_cast<std::size_t>(10));
        const std::size_t warmCols = std::min<std::size_t>(matrixSize, static_cast<std::size_t>(10));
        for (std::size_t i = 0; i < warmSteps; ++i) {
            const std::size_t rowOffset = i * matrixSize;
            for (std::size_t j = 0; j < warmCols; ++j) {
                warmUpSum += matrixData[rowOffset + j];
            }
        }
        (void)warmUpSum;
    }

    const int numThreadsReported = omp_get_max_threads();
    double globalMaxOfRowMins = std::numeric_limits<double>::lowest();

    auto startTime = std::chrono::high_resolution_clock::now();

    if (mode == "reduction") {
        #pragma omp parallel for reduction(max:globalMaxOfRowMins)
        for (std::size_t i = 0; i < matrixSize; ++i) {
            double localMin = std::numeric_limits<double>::max();
            const std::size_t rowOffset = i * matrixSize;
            for (std::size_t j = 0; j < matrixSize; ++j) {
                double val = matrixData[rowOffset + j];
                if (val < localMin)
                    localMin = val;
            }
            if (localMin > globalMaxOfRowMins)
                globalMaxOfRowMins = localMin;
        }
    }
    else if (mode == "no_reduction") {
        #pragma omp parallel
        {
            #pragma omp for
            for (std::size_t i = 0; i < matrixSize; ++i) {
                double localMin = std::numeric_limits<double>::max();
                const std::size_t rowOffset = i * matrixSize;
                for (std::size_t j = 0; j < matrixSize; ++j) {
                    double val = matrixData[rowOffset + j];
                    if (val < localMin)
                        localMin = val;
                }

                #pragma omp critical
                {
                    if (localMin > globalMaxOfRowMins)
                        globalMaxOfRowMins = localMin;
                }
            }
        }
    }
    else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 3;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double timeSeconds = std::chrono::duration<double>(endTime - startTime).count();

    std::cout << matrixSize << "," << numThreadsReported << "," << mode << "," << timeSeconds << "," << globalMaxOfRowMins << std::endl;

    return 0;
}
