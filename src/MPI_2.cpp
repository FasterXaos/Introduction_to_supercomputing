#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <iomanip>
#include <cstdint>

// Usage:
//   MPI_2 <problemSize> [seed]
//
// Example:
//   mpiexec -n 4 ./MPI_2 10000000 12345

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int numProcesses = 1;
    int processRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    if (argc < 2) {
        if (processRank == 0) {
            std::cerr << "Usage: " << argv[0] << " <problemSize> [seed]\n";
        }
        MPI_Finalize();
        return 1;
    }

    const std::size_t problemSize = static_cast<std::size_t>(std::stoull(argv[1]));
    const unsigned int seed = (argc >= 3) ? static_cast<unsigned int>(std::stoul(argv[2])) : 123456u;

    if (problemSize == 0) {
        if (processRank == 0)
            std::cerr << "problemSize must be > 0\n";
        MPI_Finalize();
        return 2;
    }

    std::vector<double> fullA;
    std::vector<double> fullB;
    if (processRank == 0) {
        fullA.resize(problemSize);
        fullB.resize(problemSize);
        std::mt19937_64 generator(static_cast<unsigned long long>(seed));
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (std::size_t i = 0; i < problemSize; ++i) {
            fullA[i] = distribution(generator);
            fullB[i] = distribution(generator);
        }
    }

    const std::size_t base = problemSize / static_cast<std::size_t>(numProcesses);
    const int remainder = static_cast<int>(problemSize % static_cast<std::size_t>(numProcesses));

    std::vector<int> sendCounts(numProcesses), displacements(numProcesses);
    if (processRank == 0) {
        std::size_t offset = 0;
        for (int p = 0; p < numProcesses; ++p) {
            std::size_t countForP = base + (p < remainder ? 1u : 0u);
            sendCounts[p] = static_cast<int>(countForP);
            displacements[p] = static_cast<int>(offset);
            offset += countForP;
        }
    }

    MPI_Bcast(sendCounts.data(), numProcesses, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displacements.data(), numProcesses, MPI_INT, 0, MPI_COMM_WORLD);

    const int localCount = sendCounts[processRank];
    std::vector<double> localA(static_cast<std::size_t>(localCount));
    std::vector<double> localB(static_cast<std::size_t>(localCount));

    MPI_Barrier(MPI_COMM_WORLD);
    const double timeStart = MPI_Wtime();

    MPI_Scatterv(
        (processRank == 0 ? fullA.data() : nullptr),
        (processRank == 0 ? sendCounts.data() : nullptr),
        (processRank == 0 ? displacements.data() : nullptr),
        MPI_DOUBLE,
        (localCount > 0 ? localA.data() : nullptr),
        localCount,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Scatterv(
        (processRank == 0 ? fullB.data() : nullptr),
        (processRank == 0 ? sendCounts.data() : nullptr),
        (processRank == 0 ? displacements.data() : nullptr),
        MPI_DOUBLE,
        (localCount > 0 ? localB.data() : nullptr),
        localCount,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double localDot = 0.0;
    for (int i = 0; i < localCount; ++i) {
        localDot += localA[static_cast<std::size_t>(i)] * localB[static_cast<std::size_t>(i)];
    }

    double globalDot = 0.0;
    MPI_Reduce(&localDot, &globalDot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double timeEnd = MPI_Wtime();
    const double timeSeconds = timeEnd - timeStart;

    if (processRank == 0) {
        std::cout << problemSize << "," << numProcesses << "," << std::fixed << std::setprecision(6)
            << timeSeconds << "," << globalDot << std::endl;
    }

    MPI_Finalize();
    return 0;
}
