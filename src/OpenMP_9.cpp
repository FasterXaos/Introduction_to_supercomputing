#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <string>
#include <omp.h>
#include <algorithm>

// Usage: OpenMP_9 <matrixSize> <mode> [innerThreads] [seed]
// mode: outer | inner | nested
// innerThreads: integer, only used for nested mode (default 1)

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrixSize> <mode> [innerThreads] [seed]\n";
        return 1;
    }

    const std::size_t matrixSize = static_cast<std::size_t>(std::stoull(argv[1]));
    const std::string mode = argv[2];
    const int innerThreads = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 1;
    const unsigned int seed = (argc >= 5) ? static_cast<unsigned int>(std::stoul(argv[4])) : 12345u;

    if (matrixSize == 0) {
        std::cerr << "matrixSize must be > 0\n";
        return 2;
    }
    if (mode != "outer" && mode != "inner" && mode != "nested") {
        std::cerr << "Unknown mode: " << mode << " (use outer|inner|nested)\n";
        return 3;
    }

    std::vector<double> matrixData(matrixSize * matrixSize);
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
        const std::size_t warmSteps = std::min<std::size_t>(matrixSize, static_cast<std::size_t>(8));
        const std::size_t warmCols = std::min<std::size_t>(matrixSize, static_cast<std::size_t>(8));
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

    if (mode == "nested") {
        omp_set_max_active_levels(2);
    }
    else {
        omp_set_max_active_levels(1);
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    if (mode == "outer") {
        #pragma omp parallel for reduction(max:globalMaxOfRowMins) schedule(dynamic)
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
    else if (mode == "inner") {
        for (std::size_t i = 0; i < matrixSize; ++i) {
            double localMin = std::numeric_limits<double>::max();
            const std::size_t rowOffset = i * matrixSize;

            #pragma omp parallel for reduction(min:localMin) schedule(static)
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
    else { // nested
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic)
            for (std::size_t i = 0; i < matrixSize; ++i) {
                double localMin = std::numeric_limits<double>::max();
                const std::size_t rowOffset = i * matrixSize;

                #pragma omp parallel for reduction(min:localMin) schedule(static) num_threads(innerThreads)
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

    auto endTime = std::chrono::high_resolution_clock::now();
    double timeSeconds = std::chrono::duration<double>(endTime - startTime).count();

    std::cout << matrixSize << "," << numThreadsReported << "," << mode << "," << innerThreads << "," << timeSeconds << "," << globalMaxOfRowMins << std::endl;

    return 0;
}
