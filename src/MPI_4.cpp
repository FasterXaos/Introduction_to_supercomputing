#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <numeric>

// Two matrix-multiplication algorithms with MPI:
// blockRow : simple row-block distribution (scatter rows of A, broadcast B)
// cannon   : Cannon's algorithm on q x q process grid (q^2 == numProcesses)
//
// Usage:
//   MPI_4 <matrixSize> <mode> [seed]
//   modes: blockRow | cannon

using std::size_t;

static void multiplyAddBlock(const double* blockA, const double* blockB, double* blockC, int blockSize) {
    for (int i = 0; i < blockSize; ++i) {
        for (int k = 0; k < blockSize; ++k) {
            double aVal = blockA[static_cast<size_t>(i) * blockSize + k];
            const double* bRow = blockB + static_cast<size_t>(k) * blockSize;
            double* cRow = blockC + static_cast<size_t>(i) * blockSize;
            for (int j = 0; j < blockSize; ++j) {
                cRow[j] += aVal * bRow[j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldSize = 1, worldRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (argc < 3) {
        if (worldRank == 0) {
            std::cerr << "Usage: " << argv[0] << " <matrixSize> <mode> [seed]\n";
            std::cerr << "mode: blockRow | cannon\n";
        }
        MPI_Finalize();
        return 1;
    }

    int matrixSize = std::stoi(argv[1]);
    std::string modeRequested = argv[2];
    unsigned int seed = (argc >= 4) ? static_cast<unsigned int>(std::stoul(argv[3])) : 123456u;

    if (matrixSize <= 0) {
        if (worldRank == 0) std::cerr << "matrixSize must be > 0\n";
        MPI_Finalize();
        return 2;
    }

    std::string mode = modeRequested;
    int q = 0;
    if (modeRequested == "cannon") {
        double sqrtP = std::sqrt(static_cast<double>(worldSize));
        q = static_cast<int>(std::floor(sqrtP + 0.5));
        if (q * q != worldSize || (matrixSize % q) != 0) {
            if (worldRank == 0) {
                std::cerr << "Cannon conditions not met (need numProcesses to be perfect square and matrixSize % sqrtP == 0). Falling back to blockRow.\n";
            }
            mode = "blockRow";
        }
    }

    std::vector<double> fullA;
    std::vector<double> fullB;
    if (worldRank == 0) {
        fullA.assign(static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize), 0.0);
        fullB.assign(static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize), 0.0);
        std::mt19937_64 generator(static_cast<unsigned long long>(seed));
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        for (int i = 0; i < matrixSize; ++i) {
            for (int j = 0; j < matrixSize; ++j) {
                fullA[static_cast<size_t>(i) * matrixSize + j] = distribution(generator);
                fullB[static_cast<size_t>(i) * matrixSize + j] = distribution(generator);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double timeStart = MPI_Wtime();

    std::vector<double> localC;
    double elapsedSeconds = 0.0;

    if (mode == "blockRow") {
        int baseRows = matrixSize / worldSize;
        int remainder = matrixSize % worldSize;
        int localRows = baseRows + (worldRank < remainder ? 1 : 0);

        std::vector<int> sendCounts(worldSize), displacements(worldSize);
        if (worldRank == 0) {
            int offset = 0;
            for (int p = 0; p < worldSize; ++p) {
                int rowsForP = baseRows + (p < remainder ? 1 : 0);
                sendCounts[p] = rowsForP * matrixSize;
                displacements[p] = offset * matrixSize;
                offset += rowsForP;
            }
        }

        std::vector<double> localA(static_cast<size_t>(localRows) * static_cast<size_t>(matrixSize), 0.0);
        std::vector<double> localB(static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize), 0.0);

        MPI_Scatterv(
            (worldRank == 0 ? fullA.data() : nullptr),
            (worldRank == 0 ? sendCounts.data() : nullptr),
            (worldRank == 0 ? displacements.data() : nullptr),
            MPI_DOUBLE,
            (localRows > 0 ? localA.data() : nullptr),
            localRows * matrixSize,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );

        MPI_Bcast((worldRank == 0 ? fullB.data() : localB.data()), matrixSize * matrixSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double* bData = (worldRank == 0 ? fullB.data() : localB.data());
        localC.assign(static_cast<size_t>(localRows) * static_cast<size_t>(matrixSize), 0.0);

        for (int i = 0; i < localRows; ++i) {
            size_t aRowOffset = static_cast<size_t>(i) * matrixSize;
            size_t cRowOffset = static_cast<size_t>(i) * matrixSize;
            for (int k = 0; k < matrixSize; ++k) {
                double aVal = localA[aRowOffset + static_cast<size_t>(k)];
                size_t bRowOffset = static_cast<size_t>(k) * matrixSize;
                for (int j = 0; j < matrixSize; ++j) {
                    localC[cRowOffset + static_cast<size_t>(j)] += aVal * bData[bRowOffset + static_cast<size_t>(j)];
                }
            }
        }

        std::vector<int> recvCounts(worldSize), recvDispls(worldSize);
        if (worldRank == 0) {
            for (int p = 0, offsetRows = 0; p < worldSize; ++p) {
                int rowsForP = baseRows + (p < remainder ? 1 : 0);
                recvCounts[p] = rowsForP * matrixSize;
                recvDispls[p] = offsetRows * matrixSize;
                offsetRows += rowsForP;
            }
        }

        std::vector<double> fullC;
        if (worldRank == 0) {
            fullC.assign(static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize), 0.0);
        }

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
        elapsedSeconds = MPI_Wtime() - timeStart;

        if (worldRank == 0) {
            double checksum = 0.0;
            for (double v : fullC) {
                checksum += v;
            }
            std::cout << matrixSize << "," << worldSize << "," << "blockRow" << "," << std::fixed << std::setprecision(6) << elapsedSeconds << "," << std::setprecision(12) << checksum << std::endl;
        }
    }
    else if (mode == "cannon") {
        q = static_cast<int>(std::floor(std::sqrt(static_cast<double>(worldSize)) + 0.5));
        int blockSize = matrixSize / q;

        int dims[2] = {q, q};
        int periods[2] = {1, 1};
        MPI_Comm cartComm;
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cartComm);

        int myCoords[2];
        MPI_Cart_coords(cartComm, worldRank, 2, myCoords);
        int myRow = myCoords[0], myCol = myCoords[1];

        std::vector<double> localAblock(static_cast<size_t>(blockSize) * static_cast<size_t>(blockSize), 0.0);
        std::vector<double> localBblock(static_cast<size_t>(blockSize) * static_cast<size_t>(blockSize), 0.0);
        localC.assign(static_cast<size_t>(blockSize) * static_cast<size_t>(blockSize), 0.0);

        if (worldRank == 0) {
            for (int p = 0; p < worldSize; ++p) {
                int coords[2];
                MPI_Cart_coords(cartComm, p, 2, coords);
                int prow = coords[0], pcol = coords[1];
                
                int rowStart = prow * blockSize;
                int colStart = pcol * blockSize;

                std::vector<double> packA(static_cast<size_t>(blockSize) * static_cast<size_t>(blockSize));
                std::vector<double> packB(static_cast<size_t>(blockSize) * static_cast<size_t>(blockSize));
                for (int bi = 0; bi < blockSize; ++bi) {
                    int aRow = rowStart + bi;
                    size_t srcOffsetA = static_cast<size_t>(aRow) * matrixSize + colStart;
                    for (int bj = 0; bj < blockSize; ++bj) {
                        packA[static_cast<size_t>(bi) * blockSize + bj] = fullA[srcOffsetA + static_cast<size_t>(bj)];
                    }
                }
                for (int bi = 0; bi < blockSize; ++bi) {
                    int bRow = rowStart + bi;
                    size_t srcOffsetB = static_cast<size_t>(bRow) * matrixSize + colStart;
                    for (int bj = 0; bj < blockSize; ++bj) {
                        packB[static_cast<size_t>(bi) * blockSize + bj] = fullB[srcOffsetB + static_cast<size_t>(bj)];
                    }
                }

                if (p == 0) {
                    localAblock = std::move(packA);
                    localBblock = std::move(packB);
                }
                else {
                    MPI_Send(packA.data(), blockSize * blockSize, MPI_DOUBLE, p, 17, cartComm);
                    MPI_Send(packB.data(), blockSize * blockSize, MPI_DOUBLE, p, 19, cartComm);
                }
            }
        }
        else {
            MPI_Recv(localAblock.data(), blockSize * blockSize, MPI_DOUBLE, 0, 17, cartComm, MPI_STATUS_IGNORE);
            MPI_Recv(localBblock.data(), blockSize * blockSize, MPI_DOUBLE, 0, 19, cartComm, MPI_STATUS_IGNORE);
        }

        int leftRank, rightRank;
        MPI_Cart_shift(cartComm, 1, -myRow, &rightRank, &leftRank);
        
        for (int step = 0; step < myRow; ++step) {
            int srcRank, dstRank;
            MPI_Cart_shift(cartComm, 1, 1, &srcRank, &dstRank);
            MPI_Sendrecv_replace(localAblock.data(), blockSize * blockSize, MPI_DOUBLE,
                dstRank, 31, srcRank, 31, cartComm, MPI_STATUS_IGNORE);
        }

        for (int step = 0; step < myCol; ++step) {
            int srcRank, dstRank;
            MPI_Cart_shift(cartComm, 0, 1, &srcRank, &dstRank);
            MPI_Sendrecv_replace(localBblock.data(), blockSize * blockSize, MPI_DOUBLE,
                dstRank, 33, srcRank, 33, cartComm, MPI_STATUS_IGNORE);
        }

        for (int iter = 0; iter < q; ++iter) {
            multiplyAddBlock(localAblock.data(), localBblock.data(), localC.data(), blockSize);

            int srcA, dstA;
            MPI_Cart_shift(cartComm, 1, -1, &srcA, &dstA);
            MPI_Sendrecv_replace(localAblock.data(), blockSize * blockSize, MPI_DOUBLE, dstA, 41, srcA, 41, cartComm, MPI_STATUS_IGNORE);

            int srcB, dstB;
            MPI_Cart_shift(cartComm, 0, -1, &srcB, &dstB);
            MPI_Sendrecv_replace(localBblock.data(), blockSize * blockSize, MPI_DOUBLE, dstB, 43, srcB, 43, cartComm, MPI_STATUS_IGNORE);
        }

        if (worldRank == 0) {
            std::vector<double> fullC(static_cast<size_t>(matrixSize) * static_cast<size_t>(matrixSize), 0.0);
            
            int rowStart = 0;
            int colStart = 0;
            for (int bi = 0; bi < blockSize; ++bi) {
                size_t dstOff = static_cast<size_t>(rowStart + bi) * matrixSize + colStart;
                for (int bj = 0; bj < blockSize; ++bj) {
                    fullC[dstOff + static_cast<size_t>(bj)] = localC[static_cast<size_t>(bi) * blockSize + bj];
                }
            }

            for (int p = 1; p < worldSize; ++p) {
                int coords[2];
                MPI_Cart_coords(cartComm, p, 2, coords);
                int prow = coords[0], pcol = coords[1];
                int destRowStart = prow * blockSize;
                int destColStart = pcol * blockSize;
                std::vector<double> recvBlock(static_cast<size_t>(blockSize) * static_cast<size_t>(blockSize));
                MPI_Recv(recvBlock.data(), blockSize * blockSize, MPI_DOUBLE, p, 51, cartComm, MPI_STATUS_IGNORE);
                for (int bi = 0; bi < blockSize; ++bi) {
                    size_t dstOff = static_cast<size_t>(destRowStart + bi) * matrixSize + destColStart;
                    for (int bj = 0; bj < blockSize; ++bj) {
                        fullC[dstOff + static_cast<size_t>(bj)] = recvBlock[static_cast<size_t>(bi) * blockSize + bj];
                    }
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
            elapsedSeconds = MPI_Wtime() - timeStart;

            double checksum = 0.0;
            for (double v : fullC) {
                checksum += v;
            }
            std::cout << matrixSize << "," << worldSize << "," << "cannon" << "," << std::fixed << std::setprecision(6) << elapsedSeconds << "," << std::setprecision(12) << checksum << std::endl;
        }
        else {
            MPI_Send(localC.data(), blockSize * blockSize, MPI_DOUBLE, 0, 51, cartComm);
            MPI_Barrier(MPI_COMM_WORLD);
        }

        MPI_Comm_free(&cartComm);
    }

    MPI_Finalize();
    return 0;
}
