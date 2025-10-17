#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <iomanip>
#include <cstdlib>
#include <cmath>

// Usage:
//   MPI_5 <messageSizeBytes> <numMessages> <computeMicroseconds> <numIterations> [computeMode]
//   computeMode: sleep | busy (default sleep)

static void busyWaitMicroseconds(long long microseconds) {
    if (microseconds <= 0) return;
    auto t0 = std::chrono::high_resolution_clock::now();
    while (true) {
        auto t1 = std::chrono::high_resolution_clock::now();
        long long elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (elapsed >= microseconds)
            break;
        // do a tiny operation to avoid being optimized out
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

    long long messageSizeBytes = std::stoll(argv[1]);
    int numMessages = std::max(0, std::atoi(argv[2]));
    long long computeMicroseconds = std::stoll(argv[3]);
    int numIterations = std::max(1, std::atoi(argv[4]));
    std::string computeMode = (argc >= 6) ? argv[5] : "sleep";

    if (messageSizeBytes < 0)
        messageSizeBytes = 0;

    std::vector<char> sendBuffer(static_cast<size_t>(messageSizeBytes), 'x');
    std::vector<char> recvBuffer(static_cast<size_t>(messageSizeBytes));

    int destRank = (worldRank + 1) % worldSize;
    int srcRank = (worldRank - 1 + worldSize) % worldSize;
    int tagBase = 1000;

    // warm-up
    MPI_Barrier(MPI_COMM_WORLD);
    int warmUpIters = std::min(10, numIterations);
    for (int wi = 0; wi < warmUpIters; ++wi) {
        if (worldSize > 1 && messageSizeBytes > 0 && numMessages > 0) {
            for (int m = 0; m < numMessages; ++m) {
                MPI_Sendrecv(sendBuffer.data(), static_cast<int>(messageSizeBytes), MPI_CHAR, destRank, tagBase + (wi + m) % 32767,
                    recvBuffer.data(), static_cast<int>(messageSizeBytes), MPI_CHAR, srcRank, tagBase + (wi + m) % 32767,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double timeStart = MPI_Wtime();

    for (int iter = 0; iter < numIterations; ++iter) {
        if (computeMicroseconds > 0) {
            if (computeMode == "busy") {
                busyWaitMicroseconds(computeMicroseconds);
            }
            else {
                std::this_thread::sleep_for(std::chrono::microseconds(computeMicroseconds));
            }
        }

        if (worldSize > 1 && messageSizeBytes > 0 && numMessages > 0) {
            for (int m = 0; m < numMessages; ++m) {
                int tag = tagBase + ((iter + m) & 0x7fff);
                MPI_Sendrecv(sendBuffer.data(), static_cast<int>(messageSizeBytes), MPI_CHAR, destRank, tag,
                    recvBuffer.data(), static_cast<int>(messageSizeBytes), MPI_CHAR, srcRank, tag,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double timeEnd = MPI_Wtime();
    double totalTimeSeconds = timeEnd - timeStart;
    double avgTimePerIteration = totalTimeSeconds / static_cast<double>(numIterations);

    double totalBytesSentPerProcess = static_cast<double>(messageSizeBytes) * static_cast<double>(numMessages) * static_cast<double>(numIterations);

    double bandwidthBytesPerSec = 0.0;
    if (totalTimeSeconds > 0.0) bandwidthBytesPerSec = totalBytesSentPerProcess / totalTimeSeconds;

    if (worldRank == 0) {
        std::cout << messageSizeBytes << "," << numMessages << "," << computeMicroseconds << "," << numIterations << ","
            << worldSize << "," << std::fixed << std::setprecision(6) << totalTimeSeconds << ","
            << std::fixed << std::setprecision(9) << avgTimePerIteration << ","
            << std::fixed << std::setprecision(3) << bandwidthBytesPerSec << std::endl;
    }

    MPI_Finalize();
    return 0;
}
