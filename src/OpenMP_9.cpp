#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <string>
#include <omp.h>
#include <cstdlib>

// Usage: OpenMP_9 <matrixSize> <mode> [innerThreads] [seed]
// mode: outer | inner | nested
// innerThreads: integer, only used for nested mode (default 1)

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrixSize> <mode> [innerThreads] [seed]\n";
        return 1;
    }

    int matrixSize = std::stoi(argv[1]);
    std::string mode = argv[2];
    int innerThreads = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 1;
    unsigned int seed = (argc >= 5) ? static_cast<unsigned int>(std::stoul(argv[4])) : 12345u;

    if (matrixSize <= 0) {
        std::cerr << "matrixSize must be > 0\n";
        return 2;
    }
    if (mode != "outer" && mode != "inner" && mode != "nested") {
        std::cerr << "Unknown mode: " << mode << " (use outer|inner|nested)\n";
        return 3;
    }

    std::vector<double> matrixData(static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize));
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0e6);

    for (int i = 0; i < matrixSize; ++i) {
        size_t rowOffset = static_cast<size_t>(i) * static_cast<size_t>(matrixSize);
        for (int j = 0; j < matrixSize; ++j) {
            matrixData[rowOffset + static_cast<size_t>(j)] = distribution(generator);
        }
    }

    // Warm-up
    {
        volatile double warmUpSum = 0.0;
        int warmSteps = std::min(matrixSize, 8);
        for (int i = 0; i < warmSteps; ++i) {
            size_t rowOffset = static_cast<size_t>(i) * static_cast<size_t>(matrixSize);
            for (int j = 0; j < std::min(matrixSize, 8); ++j) {
                warmUpSum += matrixData[rowOffset + static_cast<size_t>(j)];
            }
        }
        (void)warmUpSum;
    }

    int numThreadsReported = omp_get_max_threads();
    double globalMaxOfRowMins = std::numeric_limits<double>::lowest();

    if (mode == "nested") {
        omp_set_max_active_levels(2);
        // omp_set_nested(1);
    }
    else {
        omp_set_max_active_levels(1);
        //omp_set_nested(0);
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    if (mode == "outer") {
        #pragma omp parallel for reduction(max:globalMaxOfRowMins) schedule(dynamic)
        for (int i = 0; i < matrixSize; ++i) {
            double localMin = std::numeric_limits<double>::max();
            size_t rowOffset = static_cast<size_t>(i) * static_cast<size_t>(matrixSize);
            for (int j = 0; j < matrixSize; ++j) {
                double val = matrixData[rowOffset + static_cast<size_t>(j)];
                if (val < localMin)
                    localMin = val;
            }
            if (localMin > globalMaxOfRowMins)
                globalMaxOfRowMins = localMin;
        }
    }
    else if (mode == "inner") {
        for (int i = 0; i < matrixSize; ++i) {
            double localMin = std::numeric_limits<double>::max();
            size_t rowOffset = static_cast<size_t>(i) * static_cast<size_t>(matrixSize);

            #pragma omp parallel for reduction(min:localMin) schedule(static)
            for (int j = 0; j < matrixSize; ++j) {
                double val = matrixData[rowOffset + static_cast<size_t>(j)];
                if (val < localMin) localMin = val;
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
            for (int i = 0; i < matrixSize; ++i) {
                double localMin = std::numeric_limits<double>::max();
                size_t rowOffset = static_cast<size_t>(i) * static_cast<size_t>(matrixSize);

                #pragma omp parallel for reduction(min:localMin) schedule(static) num_threads(innerThreads)
                for (int j = 0; j < matrixSize; ++j) {
                    double val = matrixData[rowOffset + static_cast<size_t>(j)];
                    if (val < localMin) localMin = val;
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
