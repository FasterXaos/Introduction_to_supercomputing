#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <cstddef>
#include <cstdint>
#include <algorithm>

// Usage:
//   MPI_10 <matrixRows> <matrixCols> <blockRows> <blockCols> <method> [seed]
// method: derived | pack | manual

using std::size_t;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldSize = 1, worldRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    if (argc < 6) {
        if (worldRank == 0) {
            std::cerr << "Usage: " << argv[0] << " <matrixRows> <matrixCols> <blockRows> <blockCols> <method> [seed]\n";
            std::cerr << "method: derived | pack | manual\n";
        }
        MPI_Finalize();
        return 1;
    }

    const long long matrixRowsLL = std::stoll(argv[1]);
    const long long matrixColsLL = std::stoll(argv[2]);
    const long long blockRowsLL = std::stoll(argv[3]);
    const long long blockColsLL = std::stoll(argv[4]);
    const std::string method = argv[5];
    const unsigned int seed = (argc >= 7) ? static_cast<unsigned int>(std::stoul(argv[6])) : 123456u;

    if (matrixRowsLL <= 0 || matrixColsLL <= 0 || blockRowsLL <= 0 || blockColsLL <= 0) {
        if (worldRank == 0) std::cerr << "All sizes must be > 0\n";
        MPI_Finalize();
        return 2;
    }

    const size_t matrixRows = static_cast<size_t>(matrixRowsLL);
    const size_t matrixCols = static_cast<size_t>(matrixColsLL);
    const size_t blockRows = static_cast<size_t>(blockRowsLL);
    const size_t blockCols = static_cast<size_t>(blockColsLL);

    if (blockRows > matrixRows || blockCols > matrixCols) {
        if (worldRank == 0)
            std::cerr << "blockRows/blockCols must be <= matrixRows/matrixCols\n";
        MPI_Finalize();
        return 3;
    }

    std::vector<double> fullMatrix;
    if (worldRank == 0) {
        fullMatrix.assign(matrixRows * matrixCols, 0.0);
        std::mt19937_64 generator(static_cast<unsigned long long>(seed));
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        for (size_t i = 0; i < matrixRows; ++i) {
            const size_t rowOff = i * matrixCols;
            for (size_t j = 0; j < matrixCols; ++j) {
                fullMatrix[rowOff + j] = distribution(generator);
            }
        }
    }

    const size_t blockSizeElements = blockRows * blockCols;
    std::vector<double> recvBuffer(blockSizeElements, 0.0);

    auto computeStart = [&](int targetRank) -> std::pair<int, int> {
        size_t startRow = (static_cast<size_t>(targetRank) * blockRows) % matrixRows;
        size_t startCol = (static_cast<size_t>(targetRank) * blockCols) % matrixCols;
        if (startRow + blockRows > matrixRows)
            startRow = matrixRows - blockRows;
        if (startCol + blockCols > matrixCols)
            startCol = matrixCols - blockCols;
        return {static_cast<int>(startRow), static_cast<int>(startCol)};
        };

    MPI_Barrier(MPI_COMM_WORLD);
    double timeStart = MPI_Wtime();

    if (method == "derived") {
        MPI_Datatype vectorType;
        MPI_Type_vector(static_cast<int>(blockRows), static_cast<int>(blockCols), static_cast<int>(matrixCols), MPI_DOUBLE, &vectorType);

        MPI_Datatype resizedType;
        MPI_Aint lb = 0;
        MPI_Aint extent = static_cast<MPI_Aint>(sizeof(double) * blockCols);
        MPI_Type_create_resized(vectorType, lb, extent, &resizedType);
        MPI_Type_commit(&resizedType);

        if (worldRank == 0) {
            for (int p = 1; p < worldSize; ++p) {
                auto [startRow, startCol] = computeStart(p);
                double* sendPtr = fullMatrix.data() + static_cast<size_t>(startRow) * matrixCols + static_cast<size_t>(startCol);
                MPI_Send(sendPtr, 1, resizedType, p, 100 + p, MPI_COMM_WORLD);
            }
        }
        else {
            MPI_Recv(recvBuffer.data(), static_cast<int>(blockSizeElements), MPI_DOUBLE, 0, 100 + worldRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Type_free(&resizedType);
        MPI_Type_free(&vectorType);
    }
    else if (method == "pack") {
        int packRowSize = 0;
        MPI_Pack_size(static_cast<int>(blockCols), MPI_DOUBLE, MPI_COMM_WORLD, &packRowSize);
        int packBufferSize = packRowSize * static_cast<int>(blockRows) + 1024;
        if (packBufferSize < 0)
            packBufferSize = 1024;
        std::vector<char> packBuffer(static_cast<size_t>(packBufferSize));

        if (worldRank == 0) {
            for (int p = 1; p < worldSize; ++p) {
                auto [startRow, startCol] = computeStart(p);
                int position = 0;
                for (int r = 0; r < static_cast<int>(blockRows); ++r) {
                    double* srcPtr = fullMatrix.data() + static_cast<size_t>(startRow + r) * matrixCols + static_cast<size_t>(startCol);
                    MPI_Pack(srcPtr, static_cast<int>(blockCols), MPI_DOUBLE, packBuffer.data(), packBufferSize, &position, MPI_COMM_WORLD);
                }
                MPI_Send(packBuffer.data(), position, MPI_PACKED, p, 200 + p, MPI_COMM_WORLD);
            }
        }
        else {
            MPI_Status status;
            MPI_Probe(0, 200 + worldRank, MPI_COMM_WORLD, &status);
            int incomingSize = 0;
            MPI_Get_count(&status, MPI_PACKED, &incomingSize);
            std::vector<char> incomingBuffer(static_cast<size_t>(incomingSize));
            MPI_Recv(incomingBuffer.data(), incomingSize, MPI_PACKED, 0, 200 + worldRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int position = 0;
            for (int r = 0; r < static_cast<int>(blockRows); ++r) {
                MPI_Unpack(incomingBuffer.data(), incomingSize, &position,
                    recvBuffer.data() + static_cast<size_t>(r) * blockCols, static_cast<int>(blockCols), MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }
    }
    else if (method == "manual") {
        std::vector<double> packBuffer(blockSizeElements);
        if (worldRank == 0) {
            for (int p = 1; p < worldSize; ++p) {
                auto [startRow, startCol] = computeStart(p);
                double* dstPtr = packBuffer.data();
                for (int r = 0; r < static_cast<int>(blockRows); ++r) {
                    double* srcPtr = fullMatrix.data() + static_cast<size_t>(startRow + r) * matrixCols + static_cast<size_t>(startCol);
                    std::memcpy(dstPtr + static_cast<size_t>(r) * blockCols, srcPtr, static_cast<size_t>(blockCols) * sizeof(double));
                }
                MPI_Send(packBuffer.data(), static_cast<int>(blockSizeElements), MPI_DOUBLE, p, 300 + p, MPI_COMM_WORLD);
            }
        }
        else {
            MPI_Recv(recvBuffer.data(), static_cast<int>(blockSizeElements), MPI_DOUBLE, 0, 300 + worldRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else {
        if (worldRank == 0)
            std::cerr << "Unknown method: " << method << "\n";
        MPI_Finalize();
        return 4;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double timeEnd = MPI_Wtime();
    double elapsedSeconds = timeEnd - timeStart;

    double localSum = 0.0;
    if (worldRank != 0) {
        for (size_t i = 0; i < blockSizeElements; ++i) {
            localSum += recvBuffer[i];
        }
    }
    double globalSum = 0.0;
    MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        std::cout << "MPI_9," << method << "," << matrixRows << "," << matrixCols << "," << blockRows << "," << blockCols << "," << worldSize << ","
            << std::fixed << std::setprecision(6) << elapsedSeconds << "," << std::setprecision(12) << globalSum << std::endl;
    }

    MPI_Finalize();
    return 0;
}
