#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <string>
#include <algorithm>
#include <omp.h>
#include <cstdint>

// Usage:
// OpenMP_5 <matrixSize> <mode> <matrixType> <schedule> <chunk> [bandwidth] [seed]
// matrixType: banded | triangular | full
// schedule: static | dynamic | guided
// chunk: integer chunk size for scheduling (used with omp_set_schedule)
// bandwidth: for banded matrix (half-bandwidth); optional, default = 5
// mode: reduction | no_reduction
//
// Example:
// OpenMP_5 2000 reduction banded dynamic 8 10 12345

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
            << " <matrixSize> <mode> <matrixType> <schedule> <chunk> [bandwidth] [seed]\n";
        return 1;
    }

    int matrixSize = std::stoi(argv[1]);
    std::string mode = argv[2];
    std::string matrixType = argv[3]; // banded | triangular | full
    std::string scheduleType = argv[4]; // static | dynamic | guided
    int chunkSize = std::stoi(argv[5]);
    int bandwidth = (argc >= 7) ? std::max(0, std::stoi(argv[6])) : 5;
    unsigned int seed = (argc >= 8) ? static_cast<unsigned int>(std::stoul(argv[7])) : 12345u;

    if (matrixSize <= 0) {
        std::cerr << "matrixSize must be > 0\n";
        return 2;
    }
    if (chunkSize <= 0) {
        std::cerr << "chunk must be > 0\n";
        return 3;
    }
    if (bandwidth < 0) 
        bandwidth = 0;
    if (bandwidth > matrixSize - 1)
        bandwidth = matrixSize - 1;

    omp_sched_t ompScheduleKind = omp_sched_static;
    if (scheduleType == "static") {
        ompScheduleKind = omp_sched_static;
    }
    else if (scheduleType == "dynamic") {
        ompScheduleKind = omp_sched_dynamic;
    }
    else if (scheduleType == "guided") {
        ompScheduleKind = omp_sched_guided;
    }
    else {
        std::cerr << "Unknown schedule: " << scheduleType << " (use static|dynamic|guided)\n";
        return 4;
    }

    std::vector<double> matrixData;
    matrixData.resize(static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize));

    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0e6);

    if (matrixType == "banded") {
        for (int i = 0; i < matrixSize; ++i) {
            int jlo = std::max(0, i - bandwidth);
            int jhi = std::min(matrixSize - 1, i + bandwidth);
            size_t rowOffset = static_cast<size_t>(i) * static_cast<size_t>(matrixSize);

            for (int j = 0; j < matrixSize; ++j) {
                matrixData[rowOffset + static_cast<size_t>(j)] = std::numeric_limits<double>::infinity();
            }
            for (int j = jlo; j <= jhi; ++j) {
                matrixData[rowOffset + static_cast<size_t>(j)] = distribution(generator);
            }
        }
    }
    else if (matrixType == "triangular") {
        for (int i = 0; i < matrixSize; ++i) {
            size_t rowOffset = static_cast<size_t>(i) * static_cast<size_t>(matrixSize);
            for (int j = 0; j < matrixSize; ++j) {
                if (j <= i) {
                    matrixData[rowOffset + static_cast<size_t>(j)] = distribution(generator);
                }
                else {
                    matrixData[rowOffset + static_cast<size_t>(j)] = std::numeric_limits<double>::infinity();
                }
            }
        }
    }
    else { // full
        for (int i = 0; i < matrixSize; ++i) {
            size_t rowOffset = static_cast<size_t>(i) * static_cast<size_t>(matrixSize);
            for (int j = 0; j < matrixSize; ++j) {
                matrixData[rowOffset + static_cast<size_t>(j)] = distribution(generator);
            }
        }
    }

    // Warm-up
    {
        volatile double warmUpSum = 0.0;
        int warmRows = std::min(matrixSize, 10);
        for (int i = 0; i < warmRows; ++i) {
            size_t rowOffset = static_cast<size_t>(i) * static_cast<size_t>(matrixSize);
            for (int j = 0; j < std::min(matrixSize, 10); ++j) {
                warmUpSum += matrixData[rowOffset + static_cast<size_t>(j)];
            }
        }
        (void)warmUpSum;
    }

    omp_set_schedule(ompScheduleKind, chunkSize);

    int numThreadsReported = omp_get_max_threads();
    double globalMaxOfRowMins = std::numeric_limits<double>::lowest();

    auto startTime = std::chrono::high_resolution_clock::now();

    if (mode == "reduction") {
        #pragma omp parallel for reduction(max:globalMaxOfRowMins) schedule(runtime)
        for (int i = 0; i < matrixSize; ++i) {
            double localMin = std::numeric_limits<double>::infinity();
            size_t rowOffset = static_cast<size_t>(i) * static_cast<size_t>(matrixSize);

            if (matrixType == "banded") {
                int jlo = std::max(0, i - bandwidth);
                int jhi = std::min(matrixSize - 1, i + bandwidth);
                for (int j = jlo; j <= jhi; ++j) {
                    double val = matrixData[rowOffset + static_cast<size_t>(j)];
                    if (val < localMin)
                        localMin = val;
                }
            }
            else if (matrixType == "triangular") {
                for (int j = 0; j <= i; ++j) {
                    double val = matrixData[rowOffset + static_cast<size_t>(j)];
                    if (val < localMin)
                        localMin = val;
                }
            }
            else { // full
                for (int j = 0; j < matrixSize; ++j) {
                    double val = matrixData[rowOffset + static_cast<size_t>(j)];
                    if (val < localMin)
                        localMin = val;
                }
            }

            if (localMin > globalMaxOfRowMins)
                globalMaxOfRowMins = localMin;
        }
    }
    else if (mode == "no_reduction") {
        #pragma omp parallel
        {
            #pragma omp for schedule(runtime)
            for (int i = 0; i < matrixSize; ++i) {
                double localMin = std::numeric_limits<double>::infinity();
                size_t rowOffset = static_cast<size_t>(i) * static_cast<size_t>(matrixSize);

                if (matrixType == "banded") {
                    int jlo = std::max(0, i - bandwidth);
                    int jhi = std::min(matrixSize - 1, i + bandwidth);
                    for (int j = jlo; j <= jhi; ++j) {
                        double val = matrixData[rowOffset + static_cast<size_t>(j)];
                        if (val < localMin)
                            localMin = val;
                    }
                }
                else if (matrixType == "triangular") {
                    for (int j = 0; j <= i; ++j) {
                        double val = matrixData[rowOffset + static_cast<size_t>(j)];
                        if (val < localMin)
                            localMin = val;
                    }
                }
                else {
                    for (int j = 0; j < matrixSize; ++j) {
                        double val = matrixData[rowOffset + static_cast<size_t>(j)];
                        if (val < localMin)
                            localMin = val;
                    }
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
        std::cerr << "Unknown mode: " << mode << " (use reduction|no_reduction)\n";
        return 6;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double timeSeconds = std::chrono::duration<double>(endTime - startTime).count();

    std::cout << matrixSize << ","
        << numThreadsReported << ","
        << mode << ","
        << matrixType << ","
        << bandwidth << ","
        << scheduleType << ","
        << chunkSize << ","
        << timeSeconds << ","
        << globalMaxOfRowMins
        << std::endl;

    return 0;
}
