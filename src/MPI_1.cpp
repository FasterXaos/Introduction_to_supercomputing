#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <iomanip>
#include <cstdint>
#include <limits>

// Usage:
//   MPI_1 <vectorSize> <mode> [seed]
//   mode: min | max
//
// Example:
//   mpiexec -n 4 ./MPI_1 1000000 min 12345

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldSize = 1;
    int worldRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (argc < 3) {
        if (worldRank == 0) {
            std::cerr << "Usage: " << argv[0] << " <vectorSize> <mode> [seed]\n";
        }
        MPI_Finalize();
        return 1;
    }

    const std::size_t vectorSize = static_cast<std::size_t>(std::stoull(argv[1]));
    const std::string mode = argv[2];
    const unsigned int seed = (argc >= 4) ? static_cast<unsigned int>(std::stoul(argv[3])) : 123456u;

    if (vectorSize == 0) {
        if (worldRank == 0)
            std::cerr << "vectorSize must be > 0\n";
        MPI_Finalize();
        return 2;
    }

    const bool wantMin = (mode == "min");
    const bool wantMax = (mode == "max");
    if (!wantMin && !wantMax) {
        if (worldRank == 0)
            std::cerr << "Unknown mode: " << mode << " (use min|max)\n";
        MPI_Finalize();
        return 3;
    }

    std::vector<double> fullVector;
    if (worldRank == 0) {
        fullVector.resize(vectorSize);
        std::mt19937_64 generator(static_cast<unsigned long long>(seed));
        std::uniform_real_distribution<double> distribution(0.0, 1.0e6);
        for (std::size_t i = 0; i < vectorSize; ++i) fullVector[i] = distribution(generator);
    }

    const std::size_t base = vectorSize / static_cast<std::size_t>(worldSize);
    const int remainder = static_cast<int>(vectorSize % static_cast<std::size_t>(worldSize));

    std::vector<int> sendCounts(worldSize);
    std::vector<int> displacements(worldSize);
    if (worldRank == 0) {
        std::size_t offset = 0;
        for (int p = 0; p < worldSize; ++p) {
            std::size_t rowsForP = base + (p < remainder ? 1u : 0u);
            sendCounts[p] = static_cast<int>(rowsForP);
            displacements[p] = static_cast<int>(offset);
            offset += rowsForP;
        }
    }

    MPI_Bcast(sendCounts.data(), worldSize, MPI_INT, 0, MPI_COMM_WORLD);

    const int localCount = sendCounts[worldRank];
    std::vector<double> localBuffer(static_cast<std::size_t>(localCount));

    MPI_Barrier(MPI_COMM_WORLD);
    const double timeStart = MPI_Wtime();

    MPI_Scatterv(
        (worldRank == 0 ? fullVector.data() : nullptr),
        (worldRank == 0 ? sendCounts.data() : nullptr),
        (worldRank == 0 ? displacements.data() : nullptr),
        MPI_DOUBLE,
        (localCount > 0 ? localBuffer.data() : nullptr),
        localCount,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double localResult;
    if (localCount == 0) {
        localResult = wantMin ? std::numeric_limits<double>::max() : std::numeric_limits<double>::lowest();
    }
    else {
        localResult = localBuffer[0];
        for (int i = 1; i < localCount; ++i) {
            const double val = localBuffer[static_cast<std::size_t>(i)];
            if (wantMin) {
                if (val < localResult) localResult = val;
            }
            else {
                if (val > localResult) localResult = val;
            }
        }
    }

    double globalResult = 0.0;
    if (wantMin) {
        MPI_Reduce(&localResult, &globalResult, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Reduce(&localResult, &globalResult, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double timeEnd = MPI_Wtime();
    const double timeSeconds = timeEnd - timeStart;

    if (worldRank == 0) {
        std::cout << vectorSize << "," << worldSize << "," << mode << "," << std::fixed << std::setprecision(6) << timeSeconds << "," << globalResult << std::endl;
    }

    MPI_Finalize();
    return 0;
}
