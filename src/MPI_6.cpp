#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <climits>
#include <new>
#include <cstddef>
#include <cstdint>

// Modes:
//   collective  : MPI_Scatterv(A) + MPI_Bcast(B)
//   manual_std  : MPI_Send / MPI_Irecv
//   manual_ssend: MPI_Ssend / MPI_Irecv
//   manual_bsend: MPI_Bsend / MPI_Irecv (root attaches buffer)
//   manual_rsend: MPI_Rsend / MPI_Irecv (receives must be posted before sends)
//
// Usage:
//   MPI_6 <matrixSize> <sendMode> [seed]
// Example:
//   mpiexec -n 4 ./MPI_6 512 manual_ssend 12345

using std::size_t;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldSize = 1;
    int worldRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (argc < 3) {
        if (worldRank == 0) {
            std::cerr << "Usage: " << argv[0] << " <matrixSize> <sendMode> [seed]\n";
            std::cerr << "sendMode: collective | manual_std | manual_ssend | manual_bsend | manual_rsend\n";
        }
        MPI_Finalize();
        return 1;
    }

    const std::int64_t matrixSizeSigned = std::stoll(argv[1]);
    const size_t matrixSize = static_cast<size_t>(matrixSizeSigned);
    const std::string sendMode = argv[2];
    const unsigned int seed = (argc >= 4) ? static_cast<unsigned int>(std::stoul(argv[3])) : 123456u;

    if (matrixSize == 0) {
        if (worldRank == 0)
            std::cerr << "matrixSize must be > 0\n";
        MPI_Finalize();
        return 2;
    }

    std::vector<double> fullA;
    std::vector<double> fullB;
    if (worldRank == 0) {
        fullA.assign(matrixSize * matrixSize, 0.0);
        fullB.assign(matrixSize * matrixSize, 0.0);
        std::mt19937_64 generator(static_cast<unsigned long long>(seed));
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (size_t i = 0; i < matrixSize; ++i) {
            const size_t rowOff = i * matrixSize;
            for (size_t j = 0; j < matrixSize; ++j) {
                fullA[rowOff + j] = distribution(generator);
                fullB[rowOff + j] = distribution(generator);
            }
        }
    }

    const size_t baseRows = matrixSize / static_cast<size_t>(worldSize);
    const int remainder = static_cast<int>(matrixSize % static_cast<size_t>(worldSize));

    std::vector<int> sendCounts(worldSize, 0);
    std::vector<int> displacements(worldSize, 0);
    if (worldRank == 0) {
        size_t offset = 0;
        for (int p = 0; p < worldSize; ++p) {
            const size_t rowsForP = baseRows + (p < remainder ? 1u : 0u);
            sendCounts[p] = static_cast<int>(rowsForP * matrixSize);
            displacements[p] = static_cast<int>(offset * matrixSize);
            offset += rowsForP;
        }
    }

    MPI_Bcast(sendCounts.data(), worldSize, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displacements.data(), worldSize, MPI_INT, 0, MPI_COMM_WORLD);

    const int localCount = sendCounts[worldRank];
    const int matrixElemsInt = static_cast<int>(matrixSize * matrixSize);
    const int localRows = (matrixSize == 0) ? 0 : (localCount / static_cast<int>(matrixSize));

    std::vector<double> localA(static_cast<size_t>(std::max(0, localCount)));
    std::vector<double> localB(static_cast<size_t>(matrixSize * matrixSize));

    MPI_Barrier(MPI_COMM_WORLD);
    const double timeStart = MPI_Wtime();

    if (sendMode == "collective") {
        MPI_Scatterv(
            (worldRank == 0 ? fullA.data() : nullptr),
            (worldRank == 0 ? sendCounts.data() : nullptr),
            (worldRank == 0 ? displacements.data() : nullptr),
            MPI_DOUBLE,
            (localCount > 0 ? localA.data() : nullptr),
            localCount,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        MPI_Bcast((worldRank == 0 ? fullB.data() : localB.data()), matrixElemsInt, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (worldRank == 0) {
            std::copy(fullB.begin(), fullB.end(), localB.begin());
        }
    }
    else {
        MPI_Request recvRequestA = MPI_REQUEST_NULL;
        MPI_Request recvRequestB = MPI_REQUEST_NULL;

        if (localCount > 0 && worldRank != 0) {
            MPI_Irecv(localA.data(), localCount, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD, &recvRequestA);
        }
        if (worldRank != 0) {
            MPI_Irecv(localB.data(), matrixElemsInt, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD, &recvRequestB);
        }

        char* bsendBuffer = nullptr;
        int bsendBufferSize = 0;
        if (sendMode == "manual_bsend" && worldRank == 0) {
            const long long bytesPerDouble = static_cast<long long>(sizeof(double));
            long long requiredBytes = 0;
            for (int p = 1; p < worldSize; ++p) {
                const long long aBytes = static_cast<long long>(sendCounts[p]) * bytesPerDouble;
                const long long bBytes = static_cast<long long>(matrixElemsInt) * bytesPerDouble;
                requiredBytes += (aBytes + static_cast<long long>(MPI_BSEND_OVERHEAD));
                requiredBytes += (bBytes + static_cast<long long>(MPI_BSEND_OVERHEAD));
            }
            const double safetyFactor = 2.0;
            long double scaled = static_cast<long double>(requiredBytes) * safetyFactor;
            const long long safetyMargin = 4LL * 1024 * 1024;
            long long estimatedBytes = static_cast<long long>(scaled) + safetyMargin;

            if (estimatedBytes > static_cast<long long>(INT_MAX) - 1024) {
                if (worldRank == 0) {
                    std::cerr << "Warning: required MPI_Bsend buffer (" << estimatedBytes << " bytes) exceeds INT_MAX; capping to INT_MAX-1024.\n";
                }
                estimatedBytes = static_cast<long long>(INT_MAX) - 1024;
            }
            if (estimatedBytes < 0)
                estimatedBytes = static_cast<long long>(INT_MAX) - 1024;

            bsendBufferSize = static_cast<int>(estimatedBytes);

            try {
                bsendBuffer = new char[bsendBufferSize];
            }
            catch (const std::bad_alloc& ex) {
                std::cerr << "Error: failed to allocate bsend buffer of size " << bsendBufferSize << " bytes: " << ex.what() << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            const int attachErr = MPI_Buffer_attach(bsendBuffer, bsendBufferSize);
            if (attachErr != MPI_SUCCESS) {
                std::cerr << "Error: MPI_Buffer_attach failed when attaching buffer of size " << bsendBufferSize << " bytes.\n";
                void* detachedPtr = nullptr;
                int detachedSize = 0;
                MPI_Buffer_detach(&detachedPtr, &detachedSize);
                delete[] bsendBuffer;
                bsendBuffer = nullptr;
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (worldRank == 0) {
            for (int p = 0; p < worldSize; ++p) {
                const int count = sendCounts[p];
                const int disp = displacements[p];
                if (p == 0) {
                    if (count > 0) {
                        std::copy(fullA.begin() + disp, fullA.begin() + disp + count, localA.begin());
                    }
                    std::copy(fullB.begin(), fullB.end(), localB.begin());
                    continue;
                }

                const double* sendPtrA = (count > 0) ? (fullA.data() + disp) : nullptr;
                const double* sendPtrB = fullB.data();

                if (sendMode == "manual_std") {
                    if (count > 0)
                        MPI_Send(const_cast<double*>(sendPtrA), count, MPI_DOUBLE, p, 101, MPI_COMM_WORLD);
                    MPI_Send(const_cast<double*>(sendPtrB), matrixElemsInt, MPI_DOUBLE, p, 102, MPI_COMM_WORLD);
                }
                else if (sendMode == "manual_ssend") {
                    if (count > 0)
                        MPI_Ssend(const_cast<double*>(sendPtrA), count, MPI_DOUBLE, p, 101, MPI_COMM_WORLD);
                    MPI_Ssend(const_cast<double*>(sendPtrB), matrixElemsInt, MPI_DOUBLE, p, 102, MPI_COMM_WORLD);
                }
                else if (sendMode == "manual_bsend") {
                    if (count > 0)
                        MPI_Bsend(const_cast<double*>(sendPtrA), count, MPI_DOUBLE, p, 101, MPI_COMM_WORLD);
                    MPI_Bsend(const_cast<double*>(sendPtrB), matrixElemsInt, MPI_DOUBLE, p, 102, MPI_COMM_WORLD);
                }
                else if (sendMode == "manual_rsend") {
                    if (count > 0)
                        MPI_Rsend(const_cast<double*>(sendPtrA), count, MPI_DOUBLE, p, 101, MPI_COMM_WORLD);
                    MPI_Rsend(const_cast<double*>(sendPtrB), matrixElemsInt, MPI_DOUBLE, p, 102, MPI_COMM_WORLD);
                }
                else {
                    if (count > 0)
                        MPI_Send(const_cast<double*>(sendPtrA), count, MPI_DOUBLE, p, 101, MPI_COMM_WORLD);
                    MPI_Send(const_cast<double*>(sendPtrB), matrixElemsInt, MPI_DOUBLE, p, 102, MPI_COMM_WORLD);
                }
            }
        }

        if (worldRank != 0) {
            if (localCount > 0)
                MPI_Wait(&recvRequestA, MPI_STATUS_IGNORE);
            MPI_Wait(&recvRequestB, MPI_STATUS_IGNORE);
        }

        if (sendMode == "manual_bsend" && worldRank == 0) {
            void* detachedPtr = nullptr;
            int detachedSize = 0;
            MPI_Buffer_detach(&detachedPtr, &detachedSize);
            if (bsendBuffer) {
                delete[] bsendBuffer;
                bsendBuffer = nullptr;
            }
        }
    }

    std::vector<double> localC(static_cast<size_t>(localRows) * matrixSize, 0.0);
    for (int i = 0; i < localRows; ++ii) {
        const size_t aRowOffset = static_cast<size_t>(i) * matrixSize;
        const size_t cRowOffset = static_cast<size_t>(i) * matrixSize;
        for (size_t k = 0; k < matrixSize; ++k) {
            const double aVal = localA[aRowOffset + k];
            const size_t bRowOffset = k * matrixSize;
            for (size_t j = 0; j < matrixSize; ++j) {
                localC[cRowOffset + j] += aVal * localB[bRowOffset + j];
            }
        }
    }

    std::vector<int> recvCounts(worldSize, 0);
    std::vector<int> recvDispls(worldSize, 0);
    if (worldRank == 0) {
        int offsetRows = 0;
        for (int p = 0; p < worldSize; ++p) {
            const int rowsForP = static_cast<int>(baseRows + (p < remainder ? 1u : 0u));
            recvCounts[p] = rowsForP * static_cast<int>(matrixSize);
            recvDispls[p] = offsetRows * static_cast<int>(matrixSize);
            offsetRows += rowsForP;
        }
    }

    std::vector<double> fullC;
    if (worldRank == 0)
        fullC.assign(matrixSize * matrixSize, 0.0);

    MPI_Gatherv(
        (localC.empty() ? nullptr : localC.data()),
        static_cast<int>(localC.size()),
        MPI_DOUBLE,
        (worldRank == 0 ? fullC.data() : nullptr),
        (worldRank == 0 ? recvCounts.data() : nullptr),
        (worldRank == 0 ? recvDispls.data() : nullptr),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    const double timeEnd = MPI_Wtime();
    const double elapsedSeconds = timeEnd - timeStart;

    if (worldRank == 0) {
        double checksum = 0.0;
        for (double v : fullC) {
            checksum += v;
        }
        std::cout << matrixSize << "," << worldSize << "," << sendMode << "," << std::fixed << std::setprecision(6) << elapsedSeconds << "," << std::setprecision(12) << checksum << std::endl;
    }

    MPI_Finalize();
    return 0;
}
