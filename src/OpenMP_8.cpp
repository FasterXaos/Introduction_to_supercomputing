#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <thread>
#include <atomic>
#include <omp.h>

// Usage:
// OpenMP_8 <numVectors> <vectorSize> <mode> [seed]
// mode: sections | sequential

struct InputHeader {
    std::int64_t numVectors;
    std::int64_t vectorSize;
};

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <numVectors> <vectorSize> <mode> [seed]\n";
        return 1;
    }

    const std::int64_t numVectors = static_cast<std::int64_t>(std::stoll(argv[1]));
    const std::int64_t vectorSize = static_cast<std::int64_t>(std::stoll(argv[2]));
    const std::string mode = argv[3];
    const unsigned int seed = (argc >= 5) ? static_cast<unsigned int>(std::stoul(argv[4])) : 123456u;

    if (numVectors <= 0 || vectorSize <= 0) {
        std::cerr << "numVectors and vectorSize must be > 0\n";
        return 2;
    }

    const std::string resultsDir = "../results";
    const std::string inputFilePath = resultsDir + "/OpenMP_8_input.bin";

    {
        std::ofstream outFile(inputFilePath, std::ios::binary | std::ios::trunc);
        if (!outFile) {
            std::cerr << "Failed to open file for writing: " << inputFilePath << "\n";
            return 3;
        }

        InputHeader header{ numVectors, vectorSize };
        outFile.write(reinterpret_cast<const char*>(&header), sizeof(header));

        std::mt19937_64 generator(static_cast<unsigned long long>(seed));
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        std::vector<double> bufferA(static_cast<size_t>(vectorSize));
        std::vector<double> bufferB(static_cast<size_t>(vectorSize));

        for (std::int64_t v = 0; v < numVectors; ++v) {
            for (std::int64_t i = 0; i < vectorSize; ++i) {
                bufferA[static_cast<size_t>(i)] = distribution(generator);
                bufferB[static_cast<size_t>(i)] = distribution(generator);
            }
            outFile.write(reinterpret_cast<const char*>(bufferA.data()), static_cast<std::streamsize>(vectorSize * sizeof(double)));
            outFile.write(reinterpret_cast<const char*>(bufferB.data()), static_cast<std::streamsize>(vectorSize * sizeof(double)));
        }
        outFile.close();
    }

    double totalSum = 0.0;
    const int numThreadsReported = omp_get_max_threads();
    auto timeStart = std::chrono::high_resolution_clock::now();

    if (mode == "sequential" || numThreadsReported < 2) {
        std::ifstream inFile(inputFilePath, std::ios::binary);
        if (!inFile) {
            std::cerr << "Failed to open input file for reading\n"; return 4;
        }
        InputHeader header;
        inFile.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (header.numVectors != numVectors || header.vectorSize != vectorSize) {
            std::cerr << "Header mismatch\n"; return 5;
        }
        std::vector<double> vectorA(static_cast<size_t>(vectorSize));
        std::vector<double> vectorB(static_cast<size_t>(vectorSize));
        for (std::int64_t v = 0; v < numVectors; ++v) {
            inFile.read(reinterpret_cast<char*>(vectorA.data()), static_cast<std::streamsize>(vectorSize * sizeof(double)));
            inFile.read(reinterpret_cast<char*>(vectorB.data()), static_cast<std::streamsize>(vectorSize * sizeof(double)));
            double localSum = 0.0;
            for (std::int64_t i = 0; i < vectorSize; ++i) {
                localSum += vectorA[static_cast<size_t>(i)] * vectorB[static_cast<size_t>(i)];
            }
            totalSum += localSum;
        }
        inFile.close();
    }
    else if (mode == "sections") {
        const int bufferCapacity = 4;
        std::vector<std::vector<double>> bufferA(static_cast<size_t>(bufferCapacity), std::vector<double>(static_cast<size_t>(vectorSize)));
        std::vector<std::vector<double>> bufferB(static_cast<size_t>(bufferCapacity), std::vector<double>(static_cast<size_t>(vectorSize)));

        std::atomic<int> countSlots(0);
        std::atomic<bool> finishedReading(false);
        int headIdx = 0;
        int tailIdx = 0;
        omp_lock_t bufferLock;
        omp_init_lock(&bufferLock);

        omp_lock_t sumLock;
        omp_init_lock(&sumLock);

        #pragma omp parallel default(none) shared(inputFilePath, numVectors, vectorSize, bufferCapacity, bufferA, bufferB, countSlots, finishedReading, headIdx, tailIdx, bufferLock, sumLock, std::cout, std::cerr) reduction(+:totalSum)
        {
            #pragma omp sections
            {
                #pragma omp section
                {
                    std::ifstream inFile(inputFilePath, std::ios::binary);
                    if (!inFile) {
                        #pragma omp critical
                        {
                            std::cerr << "Reader: failed to open input file: " << inputFilePath << "\n";
                        }
                        finishedReading.store(true);
                    }
                    else {
                        InputHeader header;
                        inFile.read(reinterpret_cast<char*>(&header), sizeof(header));
                        for (std::int64_t v = 0; v < numVectors; ++v) {
                            while (countSlots.load(std::memory_order_acquire) == bufferCapacity) {
                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            }

                            omp_set_lock(&bufferLock);
                            std::vector<double>& outA = bufferA[static_cast<size_t>(tailIdx)];
                            std::vector<double>& outB = bufferB[static_cast<size_t>(tailIdx)];
                            inFile.read(reinterpret_cast<char*>(outA.data()), static_cast<std::streamsize>(vectorSize * sizeof(double)));
                            inFile.read(reinterpret_cast<char*>(outB.data()), static_cast<std::streamsize>(vectorSize * sizeof(double)));
                            tailIdx = (tailIdx + 1) % bufferCapacity;
                            countSlots.fetch_add(1, std::memory_order_release);
                            omp_unset_lock(&bufferLock);
                        }
                        inFile.close();
                        finishedReading.store(true);
                    }
                }

                #pragma omp section
                {
                    std::vector<double> localA(static_cast<size_t>(vectorSize));
                    std::vector<double> localB(static_cast<size_t>(vectorSize));

                    while (!finishedReading.load(std::memory_order_acquire) || countSlots.load(std::memory_order_acquire) > 0) {
                        while (countSlots.load(std::memory_order_acquire) == 0) {
                            if (finishedReading.load(std::memory_order_acquire))
                                break;
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                        if (countSlots.load(std::memory_order_acquire) == 0 && finishedReading.load(std::memory_order_acquire)) {
                            break;
                        }

                        omp_set_lock(&bufferLock);
                        std::swap(localA, bufferA[static_cast<size_t>(headIdx)]);
                        std::swap(localB, bufferB[static_cast<size_t>(headIdx)]);
                        headIdx = (headIdx + 1) % bufferCapacity;
                        countSlots.fetch_sub(1, std::memory_order_release);
                        omp_unset_lock(&bufferLock);

                        double localSum = 0.0;
                        for (std::int64_t i = 0; i < vectorSize; ++i) {
                            localSum += localA[static_cast<size_t>(i)] * localB[static_cast<size_t>(i)];
                        }

                        totalSum += localSum;
                    }
                }
            }
        }
        omp_destroy_lock(&bufferLock);
        omp_destroy_lock(&sumLock);
    }
    else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 6;
    }

    auto timeEnd = std::chrono::high_resolution_clock::now();
    double timeSeconds = std::chrono::duration<double>(timeEnd - timeStart).count();

    std::cout << numVectors << "," << vectorSize << "," << numThreadsReported << "," << mode << "," << timeSeconds << "," << totalSum << std::endl;

    return 0;
}
