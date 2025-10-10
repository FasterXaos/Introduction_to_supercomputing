#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <string>
#include <omp.h>

int main(int argc, char** argv) {
    // Usage: OpenMP_4 <matrixSize> <mode> [seed]
    // mode: reduction | no_reduction
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrixSize> <mode> [seed]\n";
        return 1;
    }

    int matrixSize = std::stoi(argv[1]);
    std::string mode = argv[2];
    unsigned int seed = (argc >= 4) ? static_cast<unsigned int>(std::stoul(argv[3])) : 12345u;

    if (matrixSize <= 0) {
        std::cerr << "matrixSize must be > 0\n";
        return 2;
    }

    std::vector<double> matrixData;
    matrixData.resize(static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize));

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
        int warmSteps = std::min(matrixSize, 10);
        for (int i = 0; i < warmSteps; ++i) {
            size_t rowOffset = static_cast<size_t>(i) * static_cast<size_t>(matrixSize);
            for (int j = 0; j < std::min(matrixSize, 10); ++j) {
                warmUpSum += matrixData[rowOffset + static_cast<size_t>(j)];
            }
        }
        (void)warmUpSum;
    }

    int numThreadsReported = omp_get_max_threads();
    double globalMaxOfRowMins = std::numeric_limits<double>::lowest();

    auto startTime = std::chrono::high_resolution_clock::now();

    if (mode == "reduction") {
        #pragma omp parallel for reduction(max:globalMaxOfRowMins)
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
    else if (mode == "no_reduction") {
        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < matrixSize; ++i) {
                double localMin = std::numeric_limits<double>::max();
                size_t rowOffset = static_cast<size_t>(i) * static_cast<size_t>(matrixSize);
                for (int j = 0; j < matrixSize; ++j) {
                    double val = matrixData[rowOffset + static_cast<size_t>(j)];
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
