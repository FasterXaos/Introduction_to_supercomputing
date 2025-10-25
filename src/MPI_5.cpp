#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cstdint>

// Usage:
//   MPI_5 <messageSizeBytes> <numMessages> <computeMicroseconds> <numIterations> [computeMode]
//   computeMode: sleep | busy (default sleep)

static void busyWaitMicroseconds(long long microseconds) {
    if (microseconds <= 0)
        return;
    const auto t0 = std::chrono::high_resolution_clock::now();
    while (true) {
        const auto t1 = std::chrono::high_resolution_clock::now();
        const long long elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (elapsed >= microseconds)
            break;
        volatile double sink = std::sin(static_cast<double>(elapsed));
        (void)sink;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldSize = 1;
    int worldRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (argc < 5) {
        if (worldRank == 0) {
            std::cerr << "Usage: " << argv[0] << " <messageSizeBytes> <numMessages> <computeMicroseconds> <numIterations> [computeMode]\n";
            std::cerr << "computeMode: sleep | busy (default sleep)\n";
        }
        MPI_Finalize();
        return 1;
    }

    const std::int64_t messageSizeSigned = std::stoll(argv[1]);
    const std::size_t messageSize = static_cast<std::size_t>(messageSizeSigned < 0 ? 0 : messageSizeSigned);

    const int numMessages = std::max(0, std::atoi(argv[2]));
    const long long computeMicroseconds = std::stoll(argv[3]);
    const int numIterations = std::max(1, std::atoi(argv[4]));
    const std::string computeMode = (argc >= 6) ? argv[5] : "sleep";

    std::vector<char> sendBuffer(messageSize, 'x');
    std::vector<char> recvBuffer(messageSize, 0);

    const int messageSizeInt = (messageSize > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        ? std::numeric_limits<int>::max()
        : static_cast<int>(messageSize);

    const int destRank = (worldSize > 0) ? ((worldRank + 1) % worldSize) : 0;
    const int srcRank = (worldSize > 0) ? ((worldRank - 1 + worldSize) % worldSize) : 0;
    const int tagBase = 1000;

    // Warm-up
    MPI_Barrier(MPI_COMM_WORLD);
    const int warmUpIters = std::min(10, numIterations);
    for (int wi = 0; wi < warmUpIters; ++wi) {
        if (worldSize > 1 && messageSize > 0 && numMessages > 0) {
            for (int m = 0; m < numMessages; ++m) {
                const int tag = tagBase + ((wi + m) & 0x7fff);
                MPI_Sendrecv(
                    sendBuffer.data(), messageSizeInt, MPI_CHAR, destRank, tag,
                    recvBuffer.data(), messageSizeInt, MPI_CHAR, srcRank, tag,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE
                );
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double timeStart = MPI_Wtime();

    for (int iter = 0; iter < numIterations; ++iter) {
        if (computeMicroseconds > 0) {
            if (computeMode == "busy") {
                busyWaitMicroseconds(computeMicroseconds);
            }
            else {
                std::this_thread::sleep_for(std::chrono::microseconds(computeMicroseconds));
            }
        }

        if (worldSize > 1 && messageSize > 0 && numMessages > 0) {
            for (int m = 0; m < numMessages; ++m) {
                const int tag = tagBase + ((iter + m) & 0x7fff);
                MPI_Sendrecv(
                    sendBuffer.data(), messageSizeInt, MPI_CHAR, destRank, tag,
                    recvBuffer.data(), messageSizeInt, MPI_CHAR, srcRank, tag,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE
                );
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double timeEnd = MPI_Wtime();

    const double totalTimeSeconds = timeEnd - timeStart;
    const double avgTimePerIteration = totalTimeSeconds / static_cast<double>(numIterations);
    const double totalBytesSentPerProcess = static_cast<double>(messageSize) * static_cast<double>(numMessages) * static_cast<double>(numIterations);
    const double bandwidthBytesPerSec = (totalTimeSeconds > 0.0) ? (totalBytesSentPerProcess / totalTimeSeconds) : 0.0;

    if (worldRank == 0) {
        std::cout << messageSize << "," << numMessages << "," << computeMicroseconds << "," << numIterations << ","
            << worldSize << "," << std::fixed << std::setprecision(6) << totalTimeSeconds << ","
            << std::fixed << std::setprecision(9) << avgTimePerIteration << ","
            << std::fixed << std::setprecision(3) << bandwidthBytesPerSec << std::endl;
    }

    MPI_Finalize();
    return 0;
}
