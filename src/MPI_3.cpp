#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdlib>

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

    long long messageSize = std::stoll(argv[1]);
    if (messageSize < 0) messageSize = 0;

    int numIterations = 1000;
    if (argc >= 3) {
        numIterations = std::max(1, std::atoi(argv[2]));
    }
    else {
        if (messageSize <= 64) numIterations = 20000;
        else if (messageSize <= 1024) numIterations = 5000;
        else if (messageSize <= 65536) numIterations = 2000;
        else if (messageSize <= 524288) numIterations = 500;
        else if (messageSize <= 2097152) numIterations = 200;
        else numIterations = 50;
    }

    if (worldSize != 2) {
        if (worldRank == 0) {
            std::cerr << "MPI_3 requires exactly 2 MPI processes. Current worldSize=" << worldSize << std::endl;
        }
        MPI_Finalize();
        return 2;
    }

    std::vector<char> sendBuffer(static_cast<size_t>(messageSize), 'x');
    std::vector<char> recvBuffer(static_cast<size_t>(messageSize), 0);

    // warm-up
    int warmUpIterations = std::min(10, numIterations);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < warmUpIterations; ++i) {
        if (worldRank == 0) {
            MPI_Send(sendBuffer.data(), static_cast<int>(messageSize), MPI_CHAR, 1, 100, MPI_COMM_WORLD);
            MPI_Recv(recvBuffer.data(), static_cast<int>(messageSize), MPI_CHAR, 1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else {
            MPI_Recv(recvBuffer.data(), static_cast<int>(messageSize), MPI_CHAR, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(sendBuffer.data(), static_cast<int>(messageSize), MPI_CHAR, 0, 101, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double timeStart = MPI_Wtime();

    for (int iter = 0; iter < numIterations; ++iter) {
        if (worldRank == 0) {
            MPI_Send(sendBuffer.data(), static_cast<int>(messageSize), MPI_CHAR, 1, 100, MPI_COMM_WORLD);
            MPI_Recv(recvBuffer.data(), static_cast<int>(messageSize), MPI_CHAR, 1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else {
            MPI_Recv(recvBuffer.data(), static_cast<int>(messageSize), MPI_CHAR, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(sendBuffer.data(), static_cast<int>(messageSize), MPI_CHAR, 0, 101, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double timeEnd = MPI_Wtime();

    double totalTimeSeconds = timeEnd - timeStart;
    double avgRoundTripSeconds = totalTimeSeconds / static_cast<double>(numIterations);

    double bandwidthBytesPerSec = 0.0;
    if (avgRoundTripSeconds > 0.0)
        bandwidthBytesPerSec = static_cast<double>(messageSize) / (avgRoundTripSeconds * 0.5);

    if (worldRank == 0) {
        std::cout << messageSize << "," << worldSize << "," << numIterations << ","
            << std::fixed << std::setprecision(6) << totalTimeSeconds << ","
            << std::fixed << std::setprecision(9) << avgRoundTripSeconds << ","
            << std::fixed << std::setprecision(3) << bandwidthBytesPerSec << std::endl;
    }

    MPI_Finalize();
    return 0;
}
