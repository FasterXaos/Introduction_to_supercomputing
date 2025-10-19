#include <iostream>
#include <chrono>
#include <random>
#include <cmath>
#include <string>
#include <vector>
#include <omp.h>

// Usage:
// OpenMP_6 <problemSize> <schedule> <chunk> <heavyProbability> <lightWork> <heavyWork> [seed]
// schedule: static | dynamic | guided
// chunk: integer chunk size for scheduling
// heavyProbability: double in [0,1], probability that an iteration is "heavy"
// lightWork: number of inner micro-iterations for "light" iteration (e.g. 10)
// heavyWork: number of inner micro-iterations for "heavy" iteration (e.g. 1000)
// seed: optional RNG seed (unsigned int)
//
// Example:
// OpenMP_6 1000000 dynamic 10 0.1 10 1000 12345

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
            << " <problemSize> <schedule> <chunk> <heavyProbability> <lightWork> <heavyWork> [seed]\n";
        return 1;
    }

    int problemSize = std::stoi(argv[1]);
    std::string scheduleType = argv[2];
    int chunkSize = std::stoi(argv[3]);
    double heavyProbability = std::stod(argv[4]);
    int lightWork = std::stoi(argv[5]);
    int heavyWork = std::stoi(argv[6]);
    unsigned int seed = (argc >= 8) ? static_cast<unsigned int>(std::stoul(argv[7])) : 123456u;

    if (problemSize <= 0 || chunkSize <= 0 || lightWork < 0 || heavyWork < 0) {
        std::cerr << "Invalid numeric argument(s)\n";
        return 2;
    }
    if (heavyProbability < 0.0 || heavyProbability > 1.0) {
        std::cerr << "heavyProbability must be in [0,1]\n";
        return 3;
    }

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

    omp_set_schedule(ompScheduleKind, chunkSize);

    int numThreadsReported = omp_get_max_threads();
    volatile double warmUp = 0.0;

    // Warm-up
    {
        for (int i = 0; i < std::min(problemSize, 100); ++i) {
            for (int k = 0; k < std::min(10, lightWork); ++k) {
                warmUp += std::sin(static_cast<double>(i + k));
            }
        }
        (void)warmUp;
    }

    double globalSum = 0.0;

    auto startTime = std::chrono::high_resolution_clock::now();


    #pragma omp parallel default(none) shared(problemSize, heavyProbability, lightWork, heavyWork, seed) reduction(+:globalSum)
    {
        int threadId = omp_get_thread_num();
        std::mt19937 localRng(seed + static_cast<unsigned int>(threadId) * 6969u);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        #pragma omp for schedule(runtime)
        for (int i = 0; i < problemSize; ++i) {
            double r = dist(localRng);
            int innerLoops = (r < heavyProbability) ? heavyWork : lightWork;

            double localValue = 0.0;
            // use simple trig-work which is not easily optimized away
            for (int k = 0; k < innerLoops; ++k) {
                double x = static_cast<double>(i) * 1e-6 + static_cast<double>(k) * 1e-3;
                localValue += std::sin(x) * std::cos(x + 0.123) + std::sqrt(std::fmod(x + 1.234, 100.0));
            }
            globalSum += localValue;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double timeSeconds = std::chrono::duration<double>(endTime - startTime).count();

    std::cout << problemSize << "," << numThreadsReported << "," << scheduleType << "," << chunkSize
        << "," << timeSeconds << "," << globalSum << std::endl;

    return 0;
}
