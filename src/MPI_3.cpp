#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <cstdint>

// Usage:
//   MPI_3 <messageSizeBytes> [numIterations]
//
// Example:
//   mpiexec -n 2 ./MPI_3 1024 10000

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldSize = 1;
    int worldRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (argc < 2) {
        if (worldRank == 0) {
            std::cerr << "Usage: " << argv[0] << " <messageSizeBytes> [numIterations]\n";
        }
        MPI_Finalize();
        return 1;
    }

    const std::int64_t messageSizeBytes = static_cast<std::int64_t>(std::stoll(argv[1]));
    const std::int64_t messageSizeSafe = (messageSizeBytes < 0) ? 0 : messageSizeBytes;

    int numIterations = 1000;
    if (argc >= 3) {
        numIterations = std::max(1, std::atoi(argv[2]));
    }
    else {
        if (messageSizeSafe <= 64) numIterations = 20000;
        else if (messageSizeSafe <= 1024) numIterations = 5000;
        else if (messageSizeSafe <= 65536) numIterations = 2000;
        else if (messageSizeSafe <= 524288) numIterations = 500;
        else if (messageSizeSafe <= 2097152) numIterations = 200;
        else numIterations = 50;
    }

    if (worldSize != 2) {
        if (worldRank == 0) {
            std::cerr << "MPI_3 requires exactly 2 MPI processes. Current worldSize=" << worldSize << std::endl;
        }
        MPI_Finalize();
        return 2;
    }

    const std::size_t bufferSize = static_cast<std::size_t>(messageSizeSafe);
    std::vector<char> sendBuffer(bufferSize, 'x');
    std::vector<char> recvBuffer(bufferSize, 0);

    // Warm-up
    const int warmUpIterations = std::min(10, numIterations);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int iter = 0; iter < warmUpIterations; ++iter) {
        if (worldRank == 0) {
            MPI_Send(sendBuffer.data(), static_cast<int>(bufferSize), MPI_CHAR, 1, 100, MPI_COMM_WORLD);
            MPI_Recv(recvBuffer.data(), static_cast<int>(bufferSize), MPI_CHAR, 1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else {
            MPI_Recv(recvBuffer.data(), static_cast<int>(bufferSize), MPI_CHAR, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(sendBuffer.data(), static_cast<int>(bufferSize), MPI_CHAR, 0, 101, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double timeStart = MPI_Wtime();

    for (int iter = 0; iter < numIterations; ++iter) {
        if (worldRank == 0) {
            MPI_Send(sendBuffer.data(), static_cast<int>(bufferSize), MPI_CHAR, 1, 100, MPI_COMM_WORLD);
            MPI_Recv(recvBuffer.data(), static_cast<int>(bufferSize), MPI_CHAR, 1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else {
            MPI_Recv(recvBuffer.data(), static_cast<int>(bufferSize), MPI_CHAR, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(sendBuffer.data(), static_cast<int>(bufferSize), MPI_CHAR, 0, 101, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double timeEnd = MPI_Wtime();

    const double totalTimeSeconds = timeEnd - timeStart;
    const double avgRoundTripSeconds = totalTimeSeconds / static_cast<double>(numIterations);

    double bandwidthBytesPerSec = 0.0;
    if (avgRoundTripSeconds > 0.0) {
        // one round-trip sends messageSize bytes twice -> estimate one-way bandwidth = messageSize / (RTT/2)
        bandwidthBytesPerSec = (bufferSize > 0) ? static_cast<double>(bufferSize) / (avgRoundTripSeconds * 0.5) : 0.0;
    }

    if (worldRank == 0) {
        std::cout << bufferSize << "," << worldSize << "," << numIterations << ","
            << std::fixed << std::setprecision(6) << totalTimeSeconds << ","
            << std::fixed << std::setprecision(9) << avgRoundTripSeconds << ","
            << std::fixed << std::setprecision(3) << bandwidthBytesPerSec << std::endl;
    }

    MPI_Finalize();
    return 0;
}
