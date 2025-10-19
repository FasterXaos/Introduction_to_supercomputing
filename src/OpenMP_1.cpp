#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <string>
#include <algorithm>
#include <omp.h>

// Usage: OpenMP_1 <problemSize> <mode> [seed]
// mode: reduction | no_reduction

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <problemSize> <mode> [seed]\n";
        return 1;
    }

    int problemSize = std::stoi(argv[1]);
    std::string mode = argv[2];
    unsigned int seed = (argc >= 4) ? static_cast<unsigned int>(std::stoul(argv[3])) : 12345u;

    std::vector<int> dataVector;
    dataVector.resize(problemSize);
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(0, 1000000000);

    for (int i = 0; i < problemSize; ++i) {
        dataVector[i] = distribution(generator);
    }

    // Warm-up
    {
        volatile long long warmUpSum = 0;
        for (int i = 0; i < std::min(problemSize, 1000); ++i) {
            warmUpSum += dataVector[i];
        }
        (void)warmUpSum;
    }

    int numThreads = omp_get_max_threads();
    int globalMin = std::numeric_limits<int>::max();

    auto startTime = std::chrono::high_resolution_clock::now();

    if (mode == "reduction") {
        #pragma omp parallel for reduction(min: globalMin)
        for (int i = 0; i < problemSize; ++i) {
            if (dataVector[i] < globalMin)
                globalMin = dataVector[i];
        }
    } else if (mode == "no_reduction") {
        #pragma omp parallel
        {
            int localMin = std::numeric_limits<int>::max();

            #pragma omp for
            for (int i = 0; i < problemSize; ++i) {
                if (dataVector[i] < localMin) 
                    localMin = dataVector[i];
            }

            #pragma omp critical
            {
                if (localMin < globalMin)
                    globalMin = localMin;
            }
        }
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 2;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double timeSeconds = std::chrono::duration<double>(endTime - startTime).count();

    std::cout << problemSize << "," << numThreads << "," << mode << "," << timeSeconds << "," << globalMin << std::endl;

    return 0;
}
