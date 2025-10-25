#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <cstdint>

// Usage:
//   MPI_7 <messageSizeBytes> <numIterations> <computeUnits> <mode> [seed]
// Modes:
//   blocking     - blocking send/recv (MPI_Sendrecv) with computation either before or after
//   nonblocking  - MPI_Irecv/MPI_Isend, do compute, then MPI_Waitall
//   comm_only    - only communication (blocking)
//   compute_only - only computation
//
// Example:
//   mpiexec -n 6 ./MPI_7 65536 50 200 nonblocking 12345

using std::size_t;

static void doComputeWork(int computeUnits) {
    double accumulator = 0.0;
    const int innerLoopCount = 1000;
    for (int u = 0; u < computeUnits; ++u) {
        for (int k = 0; k < innerLoopCount; ++k) {
            double x = static_cast<double>(u * innerLoopCount + k) * 1e-6;
            accumulator += std::sin(x) * std::cos(x + 0.123) + std::sqrt(std::fmod(x + 1.234, 100.0));
        }
    }
    // prevent optimizer from removing work
    volatile double blackhole = accumulator;
    (void)blackhole;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldSize = 1, worldRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (argc < 5) {
        if (worldRank == 0) {
            std::cerr << "Usage: " << argv[0] << " <messageSizeBytes> <numIterations> <computeUnits> <mode> [seed]\n";
            std::cerr << "mode: blocking | nonblocking | comm_only | compute_only\n";
        }
        MPI_Finalize();
        return 1;
    }

    const long long messageSizeSigned = std::stoll(argv[1]);
    const std::size_t messageSize = static_cast<std::size_t>(std::max<long long>(0LL, messageSizeSigned));
    const int numIterations = std::stoi(argv[2]);
    const int computeUnits = std::stoi(argv[3]);
    const std::string mode = argv[4];
    const unsigned int seed = (argc >= 6) ? static_cast<unsigned int>(std::stoul(argv[5])) : 123456u;

    if (messageSize == 0 && mode != "compute_only") {
        // allow zero-size when user wants compute_only, otherwise warn but proceed
    }
    if (numIterations <= 0 || computeUnits < 0) {
        if (worldRank == 0)
            std::cerr << "Invalid numeric arguments\n";
        MPI_Finalize();
        return 2;
    }

    std::vector<char> sendBuffer(messageSize, 0);
    std::vector<char> recvBuffer(messageSize, 0);

    const int bufferCountInt = (messageSize > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        ? std::numeric_limits<int>::max()
        : static_cast<int>(messageSize);

    std::mt19937_64 rng(static_cast<unsigned long long>(seed + static_cast<unsigned int>(worldRank) * 13u));
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < sendBuffer.size(); ++i) {
        sendBuffer[i] = static_cast<char>(dist(rng));
    }

    const int destRank = (worldRank + 1) % worldSize;
    const int srcRank = (worldRank - 1 + worldSize) % worldSize;
    const int tagA = 100;

    // Warm-up
    MPI_Barrier(MPI_COMM_WORLD);
    for (int w = 0; w < 2; ++w) {
        if (mode != "compute_only" && worldSize > 1) {
            MPI_Sendrecv(sendBuffer.data(), bufferCountInt, MPI_BYTE, destRank, tagA,
                recvBuffer.data(), bufferCountInt, MPI_BYTE, srcRank, tagA,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double totalWallTime = 0.0;
    double totalCommTime = 0.0;    // measured communication time (blocking sendrecv or Wait)
    double totalComputeTime = 0.0; // measured compute time

    const double globalStart = MPI_Wtime();

    if (mode == "blocking") {
        for (int iter = 0; iter < numIterations; ++iter) {
            const double compStart = MPI_Wtime();
            if (computeUnits > 0) doComputeWork(computeUnits);
            const double compEnd = MPI_Wtime();
            totalComputeTime += (compEnd - compStart);

            const double commStart = MPI_Wtime();
            if (worldSize > 1)
                MPI_Sendrecv(sendBuffer.data(), bufferCountInt, MPI_BYTE, destRank, tagA,
                    recvBuffer.data(), bufferCountInt, MPI_BYTE, srcRank, tagA,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            const double commEnd = MPI_Wtime();
            totalCommTime += (commEnd - commStart);
        }
    }
    else if (mode == "nonblocking") {
        for (int iter = 0; iter < numIterations; ++iter) {
            MPI_Request reqs[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
            MPI_Status stats[2];

            if (worldSize > 1) {
                MPI_Irecv(recvBuffer.data(), bufferCountInt, MPI_BYTE, srcRank, tagA, MPI_COMM_WORLD, &reqs[0]);
                MPI_Isend(sendBuffer.data(), bufferCountInt, MPI_BYTE, destRank, tagA, MPI_COMM_WORLD, &reqs[1]);
            }

            const double compStart = MPI_Wtime();
            if (computeUnits > 0) doComputeWork(computeUnits);
            const double compEnd = MPI_Wtime();
            totalComputeTime += (compEnd - compStart);

            if (worldSize > 1) {
                const double waitStart = MPI_Wtime();
                MPI_Waitall(2, reqs, stats);
                const double waitEnd = MPI_Wtime();
                totalCommTime += (waitEnd - waitStart);
            }
        }
    }
    else if (mode == "comm_only") {
        for (int iter = 0; iter < numIterations; ++iter) {
            const double commStart = MPI_Wtime();
            if (worldSize > 1)
                MPI_Sendrecv(sendBuffer.data(), bufferCountInt, MPI_BYTE, destRank, tagA,
                    recvBuffer.data(), bufferCountInt, MPI_BYTE, srcRank, tagA,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            const double commEnd = MPI_Wtime();
            totalCommTime += (commEnd - commStart);
        }
    }
    else if (mode == "compute_only") {
        for (int iter = 0; iter < numIterations; ++iter) {
            const double compStart = MPI_Wtime();
            if (computeUnits > 0) doComputeWork(computeUnits);
            const double compEnd = MPI_Wtime();
            totalComputeTime += (compEnd - compStart);
        }
    }
    else {
        if (worldRank == 0) std::cerr << "Unknown mode: " << mode << "\n";
        MPI_Finalize();
        return 3;
    }

    const double globalEnd = MPI_Wtime();
    totalWallTime = globalEnd - globalStart;

    double sumWallTime = 0.0, sumCommTime = 0.0, sumComputeTime = 0.0;
    MPI_Reduce(&totalWallTime, &sumWallTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&totalCommTime, &sumCommTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&totalComputeTime, &sumComputeTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        const double avgWall = sumWallTime / static_cast<double>(worldSize);
        const double avgComm = sumCommTime / static_cast<double>(worldSize);
        const double avgCompute = sumComputeTime / static_cast<double>(worldSize);

        std::cout << "MPI_7," << static_cast<unsigned long long>(messageSize) << "," << worldSize << "," << mode << "," << numIterations << "," << computeUnits << ","
            << std::fixed << std::setprecision(6) << avgWall << "," << avgComm << "," << avgCompute << std::endl;
    }

    MPI_Finalize();
    return 0;
}
