// Usage:
//   MPI_8 <messageSizeBytes> <mode> [numIterations]
//   modes: separate | sendrecv | isend_irecv
// Example:
//   mpiexec -n 2 ./MPI_7 65536 sendrecv 10000

#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <algorithm>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldSize = 1;
    int worldRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (argc < 3) {
        if (worldRank == 0) {
            std::cerr << "Usage: " << argv[0] << " <messageSizeBytes> <mode> [numIterations]\n";
            std::cerr << "mode: separate | sendrecv | isend_irecv\n";
        }
        MPI_Finalize();
        return 1;
    }

    long long messageSize = std::stoll(argv[1]);
    std::string mode = argv[2];

    int numIterations = 1000;
    if (argc >= 4) {
        numIterations = std::max(1, std::atoi(argv[3]));
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
            std::cerr << "MPI_8 requires exactly 2 MPI processes. Current worldSize=" << worldSize << std::endl;
        }
        MPI_Finalize();
        return 2;
    }

    if (messageSize < 0) messageSize = 0;
    int intMessageSize = static_cast<int>(messageSize);

    std::vector<char> sendBuffer(static_cast<size_t>(messageSize), 'x');
    std::vector<char> recvBuffer(static_cast<size_t>(messageSize), 0);

    const int tagSend = 100;
    const int tagRecv = tagSend;
    int partnerRank = (worldRank == 0) ? 1 : 0;

    // warm-up
    int warmUpIterations = std::min(10, numIterations);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < warmUpIterations; ++i) {
        if (mode == "sendrecv") {
            MPI_Sendrecv(sendBuffer.data(), intMessageSize, MPI_CHAR, partnerRank, tagSend,
                recvBuffer.data(), intMessageSize, MPI_CHAR, partnerRank, tagRecv,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if (mode == "isend_irecv") {
            MPI_Request reqs[2];
            MPI_Irecv(recvBuffer.data(), intMessageSize, MPI_CHAR, partnerRank, tagRecv, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(sendBuffer.data(), intMessageSize, MPI_CHAR, partnerRank, tagSend, MPI_COMM_WORLD, &reqs[1]);
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        }
        else { // separate
            if (worldRank == 0) {
                MPI_Send(sendBuffer.data(), intMessageSize, MPI_CHAR, partnerRank, tagSend, MPI_COMM_WORLD);
                MPI_Recv(recvBuffer.data(), intMessageSize, MPI_CHAR, partnerRank, tagRecv, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else {
                MPI_Recv(recvBuffer.data(), intMessageSize, MPI_CHAR, partnerRank, tagSend, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(sendBuffer.data(), intMessageSize, MPI_CHAR, partnerRank, tagRecv, MPI_COMM_WORLD);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double timeStart = MPI_Wtime();

    if (mode == "sendrecv") {
        for (int iter = 0; iter < numIterations; ++iter) {
            MPI_Sendrecv(sendBuffer.data(), intMessageSize, MPI_CHAR, partnerRank, tagSend,
                recvBuffer.data(), intMessageSize, MPI_CHAR, partnerRank, tagRecv,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else if (mode == "isend_irecv") {
        for (int iter = 0; iter < numIterations; ++iter) {
            MPI_Request reqs[2];
            MPI_Irecv(recvBuffer.data(), intMessageSize, MPI_CHAR, partnerRank, tagRecv, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(sendBuffer.data(), intMessageSize, MPI_CHAR, partnerRank, tagSend, MPI_COMM_WORLD, &reqs[1]);
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        }
    }
    else { // separate
        for (int iter = 0; iter < numIterations; ++iter) {
            if (worldRank == 0) {
                MPI_Send(sendBuffer.data(), intMessageSize, MPI_CHAR, partnerRank, tagSend, MPI_COMM_WORLD);
                MPI_Recv(recvBuffer.data(), intMessageSize, MPI_CHAR, partnerRank, tagRecv, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else {
                MPI_Recv(recvBuffer.data(), intMessageSize, MPI_CHAR, partnerRank, tagSend, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(sendBuffer.data(), intMessageSize, MPI_CHAR, partnerRank, tagRecv, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double timeEnd = MPI_Wtime();

    double totalTimeSeconds = timeEnd - timeStart;
    double avgRoundTripSeconds = totalTimeSeconds / static_cast<double>(numIterations);

    double bandwidthBytesPerSec = 0.0;
    if (avgRoundTripSeconds > 0.0) {
        bandwidthBytesPerSec = static_cast<double>(messageSize) / (avgRoundTripSeconds * 0.5);
    }

    if (worldRank == 0) {
        std::cout << "MPI_7," << messageSize << "," << worldSize << "," << mode << "," << numIterations << ","
            << std::fixed << std::setprecision(6) << totalTimeSeconds << ","
            << std::fixed << std::setprecision(9) << avgRoundTripSeconds << ","
            << std::fixed << std::setprecision(3) << bandwidthBytesPerSec << std::endl;
    }

    MPI_Finalize();
    return 0;
}
