#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <omp.h>

int main(int argc, char** argv) {
    // Usage: OpenMP_2 <problemSize> <mode> [seed]
    // mode: reduction | no_reduction
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <problemSize> <mode> [seed]\n";
        return 1;
    }

    int problemSize = std::stoi(argv[1]);
    std::string mode = argv[2];
    unsigned int seed = (argc >= 4) ? static_cast<unsigned int>(std::stoul(argv[3])) : 12345u;

    std::vector<double> vectorA(problemSize);
    std::vector<double> vectorB(problemSize);

    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < problemSize; ++i) {
        vectorA[i] = distribution(generator);
        vectorB[i] = distribution(generator);
    }

    // Warm-up
    {
        volatile double warmUpSum = 0.0;
        for (int i = 0; i < std::min(problemSize, 1000); ++i) {
            warmUpSum += vectorA[i] * vectorB[i];
        }
        (void)warmUpSum;
    }

    int numThreads = omp_get_max_threads();
    double globalSum = 0.0;

    auto startTime = std::chrono::high_resolution_clock::now();

    if (mode == "reduction") {
        #pragma omp parallel for reduction(+:globalSum)
        for (int i = 0; i < problemSize; ++i) {
            globalSum += vectorA[i] * vectorB[i];
        }
    }
    else if (mode == "no_reduction") {
        #pragma omp parallel
        {
            double localSum = 0.0;
            #pragma omp for
            for (int i = 0; i < problemSize; ++i) {
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
