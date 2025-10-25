#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <algorithm>

// Implemented collectives (custom):
//   customBroadcast (binomial tree)
//   customReduce (binomial tree, sum)
//   customScatter (root sends chunks to targets via pairwise sends)
//   customGather (reverse of scatter)
//   customAllGather (recursive doubling-like)
//   customAllToAll (pairwise cyclic exchanges)
//
// Usage:
//   MPI_9 <opName> <messageSizeBytes> [numIterations]
// opName: bcast | reduce | scatter | gather | allgather | alltoall
// Example:
//   mpiexec -n 4 ./MPI_9 bcast 65536 200

using std::size_t;

constexpr int TAG_BCAST = 1001;
constexpr int TAG_REDUCE = 1002;
constexpr int TAG_SCATTER = 1003;
constexpr int TAG_GATHER = 1004;
constexpr int TAG_ALLGATH = 1005;
constexpr int TAG_ALLTOALL = 1006;

static unsigned long long computeChecksum(const char* data, size_t length) {
    unsigned long long sum = 0ULL;
    for (size_t i = 0; i < length; ++i) {
        sum += static_cast<unsigned char>(data[i]);
    }
    return sum;
}

static void customBroadcast(char* buffer, int countBytes, int root, MPI_Comm comm) {
    int worldSize = 0, worldRank = 0;
    MPI_Comm_size(comm, &worldSize);
    MPI_Comm_rank(comm, &worldRank);

    int rankRel = (worldRank - root + worldSize) % worldSize;

    for (int k = 0; (1 << k) < worldSize; ++k) {
        int mask = (1 << k);
        if (rankRel & mask) {
            int srcRel = rankRel - mask;
            int src = (srcRel + root) % worldSize;
            MPI_Recv(buffer, countBytes, MPI_BYTE, src, TAG_BCAST, comm, MPI_STATUS_IGNORE);
        }
        else {
            int dstRel = rankRel + mask;
            if (dstRel < worldSize) {
                int dst = (dstRel + root) % worldSize;
                MPI_Send(buffer, countBytes, MPI_BYTE, dst, TAG_BCAST, comm);
            }
        }
    }
}

static void customReduce(const double* sendBuf, double* recvBuf, int countDoubles, int root, MPI_Comm comm) {
    int worldSize = 0, worldRank = 0;
    MPI_Comm_size(comm, &worldSize);
    MPI_Comm_rank(comm, &worldRank);

    std::vector<double> localBuf(static_cast<size_t>(countDoubles));
    if (sendBuf != nullptr) {
        std::memcpy(localBuf.data(), sendBuf, static_cast<size_t>(countDoubles) * sizeof(double));
    }
    else {
        std::fill(localBuf.begin(), localBuf.end(), 0.0);
    }

    int rankRel = (worldRank - root + worldSize) % worldSize;

    for (int k = 0; (1 << k) < worldSize; ++k) {
        int mask = (1 << k);
        if (rankRel & mask) {
            int srcRel = rankRel - mask;
            int src = (srcRel + root) % worldSize;
            std::vector<double> recvTemp(static_cast<size_t>(countDoubles));
            MPI_Recv(recvTemp.data(), countDoubles, MPI_DOUBLE, src, TAG_REDUCE, comm, MPI_STATUS_IGNORE);
            for (int i = 0; i < countDoubles; ++i) {
                localBuf[i] += recvTemp[i];
            }
        }
        else {
            int dstRel = rankRel + mask;
            if (dstRel < worldSize) {
                int dst = (dstRel + root) % worldSize;
                MPI_Send(localBuf.data(), countDoubles, MPI_DOUBLE, dst, TAG_REDUCE, comm);
            }
        }
    }

    if (worldRank == root && recvBuf != nullptr) {
        std::memcpy(recvBuf, localBuf.data(), static_cast<size_t>(countDoubles) * sizeof(double));
    }
}

static void customScatter(const char* sendBuffer, char* recvBuffer, int messageSize, int root, MPI_Comm comm) {
    int worldSize = 0, worldRank = 0;
    MPI_Comm_size(comm, &worldSize);
    MPI_Comm_rank(comm, &worldRank);

    if (worldRank == root) {
        for (int p = 0; p < worldSize; ++p) {
            if (p == root) {
                std::memcpy(recvBuffer, sendBuffer + static_cast<size_t>(p) * static_cast<size_t>(messageSize), static_cast<size_t>(messageSize));
            }
            else {
                MPI_Send(sendBuffer + static_cast<size_t>(p) * static_cast<size_t>(messageSize), messageSize, MPI_BYTE, p, TAG_SCATTER, comm);
            }
        }
    }
    else {
        MPI_Recv(recvBuffer, messageSize, MPI_BYTE, root, TAG_SCATTER, comm, MPI_STATUS_IGNORE);
    }
}

static void customGather(const char* sendBuffer, char* recvBuffer, int messageSize, int root, MPI_Comm comm) {
    int worldSize = 0, worldRank = 0;
    MPI_Comm_size(comm, &worldSize);
    MPI_Comm_rank(comm, &worldRank);

    if (worldRank == root) {
        std::memcpy(recvBuffer + static_cast<size_t>(root) * static_cast<size_t>(messageSize), sendBuffer, static_cast<size_t>(messageSize));
        for (int p = 0; p < worldSize; ++p) {
            if (p == root)
                continue;
            MPI_Recv(recvBuffer + static_cast<size_t>(p) * static_cast<size_t>(messageSize), messageSize, MPI_BYTE, p, TAG_GATHER, comm, MPI_STATUS_IGNORE);
        }
    }
    else {
        MPI_Send(sendBuffer, messageSize, MPI_BYTE, root, TAG_GATHER, comm);
    }
}

static void customAllGather(const char* sendBuffer, char* recvBuffer, int messageSize, MPI_Comm comm) {
    int worldSize = 0, worldRank = 0;
    MPI_Comm_size(comm, &worldSize);
    MPI_Comm_rank(comm, &worldRank);

    std::memcpy(recvBuffer + static_cast<size_t>(worldRank) * static_cast<size_t>(messageSize),
        sendBuffer, static_cast<size_t>(messageSize));

    int maxSteps = 0;
    while ((1 << maxSteps) < worldSize) {
        ++maxSteps;
    }

    for (int k = 0; k < maxSteps; ++k) {
        int partner = worldRank ^ (1 << k);
        if (partner >= worldSize)
            continue;

        size_t knownStart = static_cast<size_t>(worldRank & ~((1 << k) - 1));
        size_t knownCount = static_cast<size_t>(1ULL << k);
        if (knownStart + knownCount > static_cast<size_t>(worldSize)) {
            knownCount = static_cast<size_t>(worldSize) - knownStart;
        }
        size_t sendBytes = knownCount * static_cast<size_t>(messageSize);

        std::vector<char> tempSend;
        tempSend.resize(sendBytes);
        std::memcpy(tempSend.data(), recvBuffer + knownStart * static_cast<size_t>(messageSize), sendBytes);

        int sendCountInt = (sendBytes > static_cast<size_t>(std::numeric_limits<int>::max()))
            ? std::numeric_limits<int>::max()
            : static_cast<int>(sendBytes);

        MPI_Sendrecv_replace(tempSend.data(), sendCountInt, MPI_BYTE, partner, TAG_ALLGATH, partner, TAG_ALLGATH, comm, MPI_STATUS_IGNORE);

        size_t partnerKnownStart = static_cast<size_t>(partner & ~((1 << k) - 1));
        size_t recvBytes = sendBytes;
        if (partnerKnownStart + recvBytes / static_cast<size_t>(messageSize) > static_cast<size_t>(worldSize)) {
            recvBytes = (static_cast<size_t>(worldSize) - partnerKnownStart) * static_cast<size_t>(messageSize);
        }
        std::memcpy(recvBuffer + partnerKnownStart * static_cast<size_t>(messageSize), tempSend.data(), recvBytes);
    }
}

static void customAllToAll(const char* sendBuffer, char* recvBuffer, int chunkSize, MPI_Comm comm) {
    int worldSize = 0, worldRank = 0;
    MPI_Comm_size(comm, &worldSize);
    MPI_Comm_rank(comm, &worldRank);

    std::memcpy(recvBuffer + static_cast<size_t>(worldRank) * static_cast<size_t>(chunkSize),
        sendBuffer + static_cast<size_t>(worldRank) * static_cast<size_t>(chunkSize),
        static_cast<size_t>(chunkSize));

    for (int step = 1; step < worldSize; ++step) {
        int sendTo = (worldRank + step) % worldSize;
        int recvFrom = (worldRank - step + worldSize) % worldSize;
        const char* sendPtr = sendBuffer + static_cast<size_t>(sendTo) * static_cast<size_t>(chunkSize);
        char* recvPtr = recvBuffer + static_cast<size_t>(recvFrom) * static_cast<size_t>(chunkSize);
        MPI_Sendrecv(sendPtr, chunkSize, MPI_BYTE, sendTo, TAG_ALLTOALL,
            recvPtr, chunkSize, MPI_BYTE, recvFrom, TAG_ALLTOALL,
            comm, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldSize = 1, worldRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (argc < 3) {
        if (worldRank == 0) {
            std::cerr << "Usage: " << argv[0] << " <opName> <messageSizeBytes> [numIterations]\n";
            std::cerr << "opName: bcast | reduce | scatter | gather | allgather | alltoall\n";
        }
        MPI_Finalize();
        return 1;
    }

    const std::string opName = argv[1];
    const long long messageSizeBytesLL = std::stoll(argv[2]);
    const long long messageSizeBytesNonNeg = (messageSizeBytesLL < 0) ? 0LL : messageSizeBytesLL;

    if (messageSizeBytesNonNeg > static_cast<long long>(std::numeric_limits<int>::max())) {
        if (worldRank == 0) {
            std::cerr << "Warning: messageSizeBytes exceeds INT_MAX; capping to INT_MAX for MPI calls.\n";
        }
    }
    const int messageSizeBytes = (messageSizeBytesNonNeg > static_cast<long long>(std::numeric_limits<int>::max()))
        ? std::numeric_limits<int>::max()
        : static_cast<int>(messageSizeBytesNonNeg);

    int numIterations = 100;
    if (argc >= 4) numIterations = std::max(1, std::atoi(argv[3]));
    if (argc < 4) {
        if (messageSizeBytes <= 64) numIterations = 20000;
        else if (messageSizeBytes <= 1024) numIterations = 5000;
        else if (messageSizeBytes <= 65536) numIterations = 2000;
        else if (messageSizeBytes <= 524288) numIterations = 500;
        else numIterations = 100;
    }

    const int chunk = messageSizeBytes;

    std::vector<char> sendBuffer;
    std::vector<char> recvBuffer;
    std::vector<double> sendReduceD;
    std::vector<double> recvReduceD;

    if (opName == "bcast") {
        sendBuffer.resize(static_cast<size_t>(messageSizeBytes), 0);
        recvBuffer.resize(static_cast<size_t>(messageSizeBytes), 0);
    }
    else if (opName == "reduce") {
        int countDoubles = std::max(1, messageSizeBytes / static_cast<int>(sizeof(double)));
        sendReduceD.resize(static_cast<size_t>(countDoubles));
        recvReduceD.resize(static_cast<size_t>(countDoubles));
    }
    else if (opName == "scatter" || opName == "gather" || opName == "allgather" || opName == "alltoall") {
        sendBuffer.resize(static_cast<size_t>(chunk) * static_cast<size_t>(worldSize));
        recvBuffer.resize(static_cast<size_t>(chunk) * static_cast<size_t>(worldSize));
    }
    else {
        if (worldRank == 0) std::cerr << "Unknown opName: " << opName << "\n";
        MPI_Finalize();
        return 2;
    }

    unsigned int seed = 123456u + static_cast<unsigned int>(worldRank) * 33u;
    std::mt19937_64 rng(static_cast<unsigned long long>(seed));
    std::uniform_int_distribution<int> dist(0, 255);

    if (!sendBuffer.empty()) {
        for (size_t i = 0; i < sendBuffer.size(); ++i) {
            sendBuffer[i] = static_cast<char>(dist(rng));
        }
    }
    if (!recvBuffer.empty())
        std::fill(recvBuffer.begin(), recvBuffer.end(), 0);
    if (!sendReduceD.empty()) {
        for (size_t i = 0; i < sendReduceD.size(); ++i) {
            sendReduceD[i] = static_cast<double>((rng() % 1000) / 7.0);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Warm-up
    if (opName == "bcast") {
        int root = 0;
        if (worldRank == root)
            std::memcpy(recvBuffer.data(), sendBuffer.data(), static_cast<size_t>(messageSizeBytes));
        customBroadcast(recvBuffer.data(), messageSizeBytes, root, MPI_COMM_WORLD);
    }
    else if (opName == "reduce") {
        int root = 0;
        std::vector<double> tmpRecv(sendReduceD.size(), 0.0);
        customReduce(sendReduceD.data(), tmpRecv.data(), static_cast<int>(sendReduceD.size()), root, MPI_COMM_WORLD);
    }
    else if (opName == "scatter") {
        customScatter(sendBuffer.data(), recvBuffer.data(), chunk, 0, MPI_COMM_WORLD);
    }
    else if (opName == "gather") {
        customGather(sendBuffer.data(), recvBuffer.data(), chunk, 0, MPI_COMM_WORLD);
    }
    else if (opName == "allgather") {
        customAllGather(sendBuffer.empty() ? nullptr : sendBuffer.data() + static_cast<size_t>(worldRank) * static_cast<size_t>(chunk),
            recvBuffer.data(), chunk, MPI_COMM_WORLD);
    }
    else if (opName == "alltoall") {
        customAllToAll(sendBuffer.data(), recvBuffer.data(), chunk, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // measure custom
    double startCustom = MPI_Wtime();
    for (int it = 0; it < numIterations; ++it) {
        if (opName == "bcast") {
            int root = 0;
            if (worldRank == root)
                std::memcpy(recvBuffer.data(), sendBuffer.data(), static_cast<size_t>(messageSizeBytes));
            customBroadcast(recvBuffer.data(), messageSizeBytes, root, MPI_COMM_WORLD);
        }
        else if (opName == "reduce") {
            int root = 0;
            std::vector<double> tmpRecv(sendReduceD.size(), 0.0);
            customReduce(sendReduceD.data(), tmpRecv.data(), static_cast<int>(sendReduceD.size()), root, MPI_COMM_WORLD);
            if (worldRank == root)
                std::memcpy(recvReduceD.data(), tmpRecv.data(), tmpRecv.size() * sizeof(double));
        }
        else if (opName == "scatter") {
            customScatter(sendBuffer.data(), recvBuffer.data(), chunk, 0, MPI_COMM_WORLD);
        }
        else if (opName == "gather") {
            customGather(sendBuffer.data(), recvBuffer.data(), chunk, 0, MPI_COMM_WORLD);
        }
        else if (opName == "allgather") {
            customAllGather(sendBuffer.data() + static_cast<size_t>(worldRank) * static_cast<size_t>(chunk), recvBuffer.data(), chunk, MPI_COMM_WORLD);
        }
        else if (opName == "alltoall") {
            customAllToAll(sendBuffer.data(), recvBuffer.data(), chunk, MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double endCustom = MPI_Wtime();
    double customTime = (endCustom - startCustom) / static_cast<double>(numIterations);

    // measure MPI builtin
    MPI_Barrier(MPI_COMM_WORLD);
    double startMpi = MPI_Wtime();
    for (int it = 0; it < numIterations; ++it) {
        if (opName == "bcast") {
            int root = 0;
            if (worldRank == root)
                std::memcpy(recvBuffer.data(), sendBuffer.data(), static_cast<size_t>(messageSizeBytes));
            MPI_Bcast(recvBuffer.data(), messageSizeBytes, MPI_BYTE, root, MPI_COMM_WORLD);
        }
        else if (opName == "reduce") {
            int root = 0;
            std::vector<double> tmpRecv(sendReduceD.size(), 0.0);
            MPI_Reduce(sendReduceD.data(), tmpRecv.data(), static_cast<int>(sendReduceD.size()), MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
            if (worldRank == root)
                std::memcpy(recvReduceD.data(), tmpRecv.data(), tmpRecv.size() * sizeof(double));
        }
        else if (opName == "scatter") {
            int root = 0;
            MPI_Scatter((worldRank == root ? sendBuffer.data() : nullptr), chunk, MPI_BYTE,
                recvBuffer.data(), chunk, MPI_BYTE, root, MPI_COMM_WORLD);
        }
        else if (opName == "gather") {
            int root = 0;
            MPI_Gather(sendBuffer.data(), chunk, MPI_BYTE,
                (worldRank == root ? recvBuffer.data() : nullptr), chunk, MPI_BYTE, root, MPI_COMM_WORLD);
        }
        else if (opName == "allgather") {
            MPI_Allgather(sendBuffer.data() + static_cast<size_t>(worldRank) * static_cast<size_t>(chunk), chunk, MPI_BYTE,
                recvBuffer.data(), chunk, MPI_BYTE, MPI_COMM_WORLD);
        }
        else if (opName == "alltoall") {
            MPI_Alltoall(sendBuffer.data(), chunk, MPI_BYTE,
                recvBuffer.data(), chunk, MPI_BYTE, MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double endMpi = MPI_Wtime();
    double mpiTime = (endMpi - startMpi) / static_cast<double>(numIterations);

    unsigned long long checksum = 0ULL;
    if (opName == "bcast") {
        checksum = computeChecksum(recvBuffer.data(), static_cast<size_t>(messageSizeBytes));
    }
    else if (opName == "reduce") {
        if (worldRank == 0) {
            const char* bytes = reinterpret_cast<const char*>(recvReduceD.data());
            checksum = computeChecksum(bytes, recvReduceD.size() * sizeof(double));
        }
    }
    else if (opName == "scatter") {
        unsigned long long localSum = computeChecksum(recvBuffer.data(), static_cast<size_t>(chunk));
        MPI_Reduce(&localSum, &checksum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else if (opName == "gather") {
        if (worldRank == 0)
            checksum = computeChecksum(recvBuffer.data(), static_cast<size_t>(chunk) * static_cast<size_t>(worldSize));
    }
    else if (opName == "allgather") {
        unsigned long long localSum = computeChecksum(recvBuffer.data(), static_cast<size_t>(chunk) * static_cast<size_t>(worldSize));
        MPI_Reduce(&localSum, &checksum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else if (opName == "alltoall") {
        unsigned long long localSum = computeChecksum(recvBuffer.data(), static_cast<size_t>(chunk) * static_cast<size_t>(worldSize));
        MPI_Reduce(&localSum, &checksum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (worldRank == 0) {
        std::cout << "MPI_9," << opName << "," << messageSizeBytes << "," << worldSize << ","
            << std::fixed << std::setprecision(9) << customTime << ","
            << std::fixed << std::setprecision(9) << mpiTime << "," << checksum << std::endl;
    }

    MPI_Finalize();
    return 0;
}
