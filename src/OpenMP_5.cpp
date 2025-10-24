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

    const std::size_t matrixSize = static_cast<std::size_t>(std::stoull(argv[1]));
    const std::string mode = argv[2];
    const std::string matrixType = argv[3]; // banded | triangular | full
    const std::string scheduleType = argv[4]; // static | dynamic | guided
    const int chunkSize = std::stoi(argv[5]);
    std::size_t bandwidth = (argc >= 7) ? static_cast<std::size_t>(std::max(0, std::stoi(argv[6]))) : static_cast<std::size_t>(5);
    const unsigned int seed = (argc >= 8) ? static_cast<unsigned int>(std::stoul(argv[7])) : 12345u;

    if (matrixSize == 0) {
        std::cerr << "matrixSize must be > 0\n";
        return 2;
    }
    if (chunkSize <= 0) {
        std::cerr << "chunk must be > 0\n";
        return 3;
    }
    if (bandwidth > (matrixSize == 0 ? 0 : matrixSize - 1))
        bandwidth = (matrixSize == 0) ? 0 : matrixSize - 1;

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
    matrixData.resize(matrixSize * matrixSize);

    std::mt19937_64 generator(static_cast<unsigned long long>(seed));
    std::uniform_real_distribution<double> distribution(0.0, 1.0e6);

    if (matrixType == "banded") {
        for (std::size_t i = 0; i < matrixSize; ++i) {
            std::size_t rowOffset = i * matrixSize;
            // initialize row with infinities
            for (std::size_t j = 0; j < matrixSize; ++j) {
                matrixData[rowOffset + j] = std::numeric_limits<double>::infinity();
            }
            const std::size_t jlo = (i > bandwidth) ? (i - bandwidth) : 0;
            const std::size_t jhi = std::min(matrixSize - 1, i + bandwidth);
            for (std::size_t j = jlo; j <= jhi; ++j) {
                matrixData[rowOffset + j] = distribution(generator);
            }
        }
    }
    else if (matrixType == "triangular") {
        for (std::size_t i = 0; i < matrixSize; ++i) {
            std::size_t rowOffset = i * matrixSize;
            for (std::size_t j = 0; j < matrixSize; ++j) {
                if (j <= i) {
                    matrixData[rowOffset + j] = distribution(generator);
                }
                else {
                    matrixData[rowOffset + j] = std::numeric_limits<double>::infinity();
                }
            }
        }
    }
    else { // full
        for (std::size_t i = 0; i < matrixSize; ++i) {
            std::size_t rowOffset = i * matrixSize;
            for (std::size_t j = 0; j < matrixSize; ++j) {
                matrixData[rowOffset + j] = distribution(generator);
            }
        }
    }

    // Warm-up
    {
        volatile double warmUpSum = 0.0;
        const std::size_t warmRows = std::min<std::size_t>(matrixSize, static_cast<std::size_t>(10));
        const std::size_t warmCols = std::min<std::size_t>(matrixSize, static_cast<std::size_t>(10));
        for (std::size_t i = 0; i < warmRows; ++i) {
            std::size_t rowOffset = i * matrixSize;
            for (std::size_t j = 0; j < warmCols; ++j) {
                warmUpSum += matrixData[rowOffset + j];
            }
        }
        (void)warmUpSum;
    }

    omp_set_schedule(ompScheduleKind, chunkSize);

    const int numThreadsReported = omp_get_max_threads();
    double globalMaxOfRowMins = std::numeric_limits<double>::lowest();

    auto startTime = std::chrono::high_resolution_clock::now();

    if (mode == "reduction") {
        #pragma omp parallel for reduction(max:globalMaxOfRowMins) schedule(runtime)
        for (std::size_t i = 0; i < matrixSize; ++i) {
            double localMin = std::numeric_limits<double>::infinity();
            const std::size_t rowOffset = i * matrixSize;

            if (matrixType == "banded") {
                const std::size_t jlo = (i > bandwidth) ? (i - bandwidth) : 0;
                const std::size_t jhi = std::min(matrixSize - 1, i + bandwidth);
                for (std::size_t j = jlo; j <= jhi; ++j) {
                    double val = matrixData[rowOffset + j];
                    if (val < localMin)
                        localMin = val;
                }
            }
            else if (matrixType == "triangular") {
                for (std::size_t j = 0; j <= i; ++j) {
                    double val = matrixData[rowOffset + j];
                    if (val < localMin)
                        localMin = val;
                }
            }
            else { // full
                for (std::size_t j = 0; j < matrixSize; ++j) {
                    double val = matrixData[rowOffset + j];
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
            for (std::size_t i = 0; i < matrixSize; ++i) {
                double localMin = std::numeric_limits<double>::infinity();
                const std::size_t rowOffset = i * matrixSize;

                if (matrixType == "banded") {
                    const std::size_t jlo = (i > bandwidth) ? (i - bandwidth) : 0;
                    const std::size_t jhi = std::min(matrixSize - 1, i + bandwidth);
                    for (std::size_t j = jlo; j <= jhi; ++j) {
                        double val = matrixData[rowOffset + j];
                        if (val < localMin)
                            localMin = val;
                    }
                }
                else if (matrixType == "triangular") {
                    for (std::size_t j = 0; j <= i; ++j) {
                        double val = matrixData[rowOffset + j];
                        if (val < localMin)
                            localMin = val;
                    }
                }
                else {
                    for (std::size_t j = 0; j < matrixSize; ++j) {
                        double val = matrixData[rowOffset + j];
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
