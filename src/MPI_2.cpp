#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <iomanip>

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

    const std::int64_t problemSize = static_cast<std::int64_t>(std::stoll(argv[1]));
    const unsigned int seed = (argc >= 3) ? static_cast<unsigned int>(std::stoul(argv[2])) : 123456u;

    if (problemSize <= 0) {
        if (processRank == 0)
            std::cerr << "problemSize must be > 0\n";
        MPI_Finalize();
        return 2;
    }

    std::vector<double> fullA;
    std::vector<double> fullB;
    std::vector<int> sendCounts(numProcesses, 0);
    std::vector<int> displacements(numProcesses, 0);

    if (processRank == 0) {
        fullA.resize(static_cast<std::size_t>(problemSize));
        fullB.resize(static_cast<std::size_t>(problemSize));

        std::mt19937_64 generator(static_cast<unsigned long long>(seed));
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        for (std::int64_t i = 0; i < problemSize; ++i) {
            fullA[static_cast<std::size_t>(i)] = distribution(generator);
            fullB[static_cast<std::size_t>(i)] = distribution(generator);
        }

        const std::int64_t base = problemSize / numProcesses;
        const int remainder = static_cast<int>(problemSize % numProcesses);
        int offset = 0;
        for (int p = 0; p < numProcesses; ++p) {
            const int count = static_cast<int>(base + (p < remainder ? 1 : 0));
            sendCounts[p] = count;
            displacements[p] = offset;
            offset += count;
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
        sendCounts.data(),
        displacements.data(),
        MPI_DOUBLE,
        (localCount > 0 ? localA.data() : nullptr),
        localCount,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Scatterv(
        (processRank == 0 ? fullB.data() : nullptr),
        sendCounts.data(),
        displacements.data(),
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
