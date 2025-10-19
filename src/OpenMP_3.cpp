#include <iostream>
#include <chrono>
#include <cmath>
#include <string>
#include <omp.h>
#include <cstdint>

// Usage: OpenMP_3 <numIntervals> <mode> <a> <b>
// mode: reduction | no_reduction
    
// The simplest function. It can be replaced/expanded if necessary.
static inline double integrandFunction(double x) {
    return std::sin(x);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <numIntervals> <mode> <a> <b>\n";
        return 1;
    }

    std::int64_t numIntervals = static_cast<std::int64_t>(std::stoll(argv[1]));
    std::string mode = argv[2];
    double lowerBound = std::stod(argv[3]);
    double upperBound = std::stod(argv[4]);

    if (numIntervals <= 0) {
        std::cerr << "numIntervals must be > 0\n";
        return 2;
    }
    if (upperBound <= lowerBound) {
        std::cerr << "upperBound must be > lowerBound\n";
        return 3;
    }

    // Warm-up
    {
        double warmUpSum = 0.0;
        double warmUpH = (upperBound - lowerBound) / static_cast<double>(numIntervals);
        int warmUpSteps = static_cast<int>(std::min<std::int64_t>(numIntervals, 1000));
        for (int i = 0; i < warmUpSteps; ++i) {
            double x = lowerBound + static_cast<double>(i) * warmUpH;
            warmUpSum += integrandFunction(x);
        }
        (void)warmUpSum;
    }

    int numThreadsReported = omp_get_max_threads();
    double integralResult = 0.0;
    double stepSize = (upperBound - lowerBound) / static_cast<double>(numIntervals);

    auto startTime = std::chrono::high_resolution_clock::now();

    if (mode == "reduction") {
        #pragma omp parallel for reduction(+:integralResult)
        for (std::int64_t i = 0; i < numIntervals; ++i) {
            double x = lowerBound + static_cast<double>(i) * stepSize;
            integralResult += integrandFunction(x);
        }
        integralResult *= stepSize;
    }
    else if (mode == "no_reduction") {
        #pragma omp parallel
        {
            double localSum = 0.0;
            #pragma omp for
            for (std::int64_t i = 0; i < numIntervals; ++i) {
                double x = lowerBound + static_cast<double>(i) * stepSize;
                localSum += integrandFunction(x);
            }

            #pragma omp critical
            {
                integralResult += localSum;
            }
        }
        integralResult *= stepSize;
    }
    else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 4;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double timeSeconds = std::chrono::duration<double>(endTime - startTime).count();

    std::cout << numIntervals << "," << numThreadsReported << "," << mode << "," << timeSeconds << "," << integralResult << std::endl;

    return 0;
}
