#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <random>
#include <iomanip>

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

    long long messageSizeBytes = std::stoll(argv[1]);
    int numIterations = std::stoi(argv[2]);
    int computeUnits = std::stoi(argv[3]);
    std::string mode = argv[4];
    unsigned int seed = (argc >= 6) ? static_cast<unsigned int>(std::stoul(argv[5])) : 123456u;

    if (messageSizeBytes < 0 || numIterations <= 0 || computeUnits < 0) {
        if (worldRank == 0) std::cerr << "Invalid numeric arguments\n";
        MPI_Finalize();
        return 2;
    }

    std::vector<char> sendBuffer(static_cast<size_t>(messageSizeBytes), 0);
    std::vector<char> recvBuffer(static_cast<size_t>(messageSizeBytes), 0);

    std::mt19937_64 rng(static_cast<unsigned long long>(seed + static_cast<unsigned int>(worldRank) * 13u));
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < sendBuffer.size(); ++i) sendBuffer[i] = static_cast<char>(dist(rng));

    int destRank = (worldRank + 1) % worldSize;
    int srcRank = (worldRank - 1 + worldSize) % worldSize;
    const int tagA = 100;

    // warm-up
    MPI_Barrier(MPI_COMM_WORLD);
    for (int w = 0; w < 2; ++w) {
        MPI_Sendrecv(sendBuffer.data(), static_cast<int>(sendBuffer.size()), MPI_BYTE, destRank, tagA,
            recvBuffer.data(), static_cast<int>(recvBuffer.size()), MPI_BYTE, srcRank, tagA,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double totalWallTime = 0.0;
    double totalCommTime = 0.0;    // measured communication time (blocking sendrecv or Wait)
    double totalComputeTime = 0.0; // measured compute time

    double globalStart = MPI_Wtime();

    if (mode == "blocking") {
        for (int iter = 0; iter < numIterations; ++iter) {
            double compStart = MPI_Wtime();
            if (computeUnits > 0)
                doComputeWork(computeUnits);
            double compEnd = MPI_Wtime();
            totalComputeTime += (compEnd - compStart);

            double commStart = MPI_Wtime();
            MPI_Sendrecv(sendBuffer.data(), static_cast<int>(sendBuffer.size()), MPI_BYTE, destRank, tagA,
                recvBuffer.data(), static_cast<int>(recvBuffer.size()), MPI_BYTE, srcRank, tagA,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double commEnd = MPI_Wtime();
            totalCommTime += (commEnd - commStart);
        }
    }
    else if (mode == "nonblocking") {
        for (int iter = 0; iter < numIterations; ++iter) {
            MPI_Request reqs[2];
            MPI_Status stats[2];

            int recvReqErr = MPI_Irecv(recvBuffer.data(), static_cast<int>(recvBuffer.size()), MPI_BYTE, srcRank, tagA, MPI_COMM_WORLD, &reqs[0]);
            int sendReqErr = MPI_Isend(sendBuffer.data(), static_cast<int>(sendBuffer.size()), MPI_BYTE, destRank, tagA, MPI_COMM_WORLD, &reqs[1]);
            (void)recvReqErr; (void)sendReqErr;

            double compStart = MPI_Wtime();
            if (computeUnits > 0)
                doComputeWork(computeUnits);
            double compEnd = MPI_Wtime();
            totalComputeTime += (compEnd - compStart);

            double waitStart = MPI_Wtime();
            MPI_Waitall(2, reqs, stats);
            double waitEnd = MPI_Wtime();
            totalCommTime += (waitEnd - waitStart);
        }
    }
    else if (mode == "comm_only") {
        for (int iter = 0; iter < numIterations; ++iter) {
            double commStart = MPI_Wtime();
            MPI_Sendrecv(sendBuffer.data(), static_cast<int>(sendBuffer.size()), MPI_BYTE, destRank, tagA,
                recvBuffer.data(), static_cast<int>(recvBuffer.size()), MPI_BYTE, srcRank, tagA,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double commEnd = MPI_Wtime();
            totalCommTime += (commEnd - commStart);
        }
    }
    else if (mode == "compute_only") {
        for (int iter = 0; iter < numIterations; ++iter) {
            double compStart = MPI_Wtime();
            if (computeUnits > 0)
                doComputeWork(computeUnits);
            double compEnd = MPI_Wtime();
            totalComputeTime += (compEnd - compStart);
        }
    }
    else {
        if (worldRank == 0) std::cerr << "Unknown mode: " << mode << "\n";
        MPI_Finalize();
        return 3;
    }

    double globalEnd = MPI_Wtime();
    totalWallTime = globalEnd - globalStart;

    double sumWallTime = 0.0, sumCommTime = 0.0, sumComputeTime = 0.0;
    MPI_Reduce(&totalWallTime, &sumWallTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&totalCommTime, &sumCommTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&totalComputeTime, &sumComputeTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        double avgWall = sumWallTime / static_cast<double>(worldSize);
        double avgComm = sumCommTime / static_cast<double>(worldSize);
        double avgCompute = sumComputeTime / static_cast<double>(worldSize);

        std::cout << "MPI_7," << messageSizeBytes << "," << worldSize << "," << mode << "," << numIterations << "," << computeUnits << ","
            << std::fixed << std::setprecision(6) << avgWall << "," << avgComm << "," << avgCompute << std::endl;
    }

    MPI_Finalize();
    return 0;
}
