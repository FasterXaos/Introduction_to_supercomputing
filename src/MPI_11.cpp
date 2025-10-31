#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <numeric>
#include <iomanip>

// Usage:
//   MPI_11 <numIterations> [gridRows gridCols] [seed]

static double computeMedian(std::vector<double>& valuesVector) {
    if (valuesVector.empty())
        return 0.0;
    std::sort(valuesVector.begin(), valuesVector.end());
    const std::size_t n = valuesVector.size();

    if (n % 2 == 1)
        return valuesVector[n / 2];
    return 0.5 * (valuesVector[n / 2 - 1] + valuesVector[n / 2]);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldSize = 1;
    int worldRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    std::size_t numIterations = 1000u;
    if (argc >= 2) {
        numIterations = static_cast<std::size_t>(std::max(1, std::atoi(argv[1])));
    }

    int gridRows = 0;
    int gridCols = 0;
    if (argc >= 4) {
        gridRows = std::max(1, std::atoi(argv[2]));
        gridCols = std::max(1, std::atoi(argv[3]));
    }
    else {
        int approx = static_cast<int>(std::floor(std::sqrt(static_cast<double>(worldSize))));
        int selectedRows = 1;
        for (int r = approx; r >= 1; --r) {
            if (worldSize % r == 0) {
                selectedRows = r;
                break;
            }
        }
        gridRows = selectedRows;
        gridCols = worldSize / gridRows;
    }

    if (gridRows * gridCols != worldSize) {
        if (worldRank == 0) {
            std::cerr << "Grid size mismatch: gridRows * gridCols != numProcesses\n";
            std::cerr << "Requested: " << gridRows << " x " << gridCols << " , but numProcesses = " << worldSize << "\n";
        }
        MPI_Finalize();
        return 2;
    }

    int dims[2] = { gridRows, gridCols };
    int periods[2] = { 0, 0 };
    MPI_Comm cartComm = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cartComm);

    int coords[2] = { 0, 0 };
    MPI_Cart_coords(cartComm, worldRank, 2, coords);
    int myRow = coords[0];
    int myCol = coords[1];

    int remainDimsRow[2] = { 0, 1 };
    MPI_Comm rowComm = MPI_COMM_NULL;
    MPI_Cart_sub(cartComm, remainDimsRow, &rowComm);

    int remainDimsCol[2] = { 1, 0 };
    MPI_Comm colComm = MPI_COMM_NULL;
    MPI_Cart_sub(cartComm, remainDimsCol, &colComm);

    std::vector<std::pair<std::string, MPI_Comm>> commList;
    commList.emplace_back(std::string("world"), MPI_COMM_WORLD);
    commList.emplace_back(std::string("row"), rowComm);
    commList.emplace_back(std::string("col"), colComm);

    const double localValueBase = static_cast<double>(worldRank + 1);

    for (const auto& commEntry : commList) {
        const std::string commLabel = commEntry.first;
        MPI_Comm measuredComm = commEntry.second;

        std::vector<double> iterationTimes;
        iterationTimes.reserve(numIterations);

        for (std::size_t iter = 0; iter < numIterations; ++iter) {
            const double localValue = localValueBase + 1e-6 * static_cast<double>(iter);

            MPI_Barrier(measuredComm);

            const double t0 = MPI_Wtime();
            double globalSum = 0.0;
            MPI_Allreduce(&localValue, &globalSum, 1, MPI_DOUBLE, MPI_SUM, measuredComm);
            const double t1 = MPI_Wtime();
            iterationTimes.push_back(t1 - t0);
        }

        const double medianTime = computeMedian(iterationTimes);

        const double localValueCheck = localValueBase;
        double globalSumCheck = 0.0;
        MPI_Allreduce(&localValueCheck, &globalSumCheck, 1, MPI_DOUBLE, MPI_SUM, measuredComm);

        if (worldRank == 0) {
            std::cout << "MPI_11,"
                << gridRows << "," << gridCols << ","
                << worldSize << ","
                << commLabel << ","
                << std::fixed << std::setprecision(6) << medianTime << ","
                << std::setprecision(12) << globalSumCheck << std::endl;
        }
    }

    if (rowComm != MPI_COMM_NULL)
        MPI_Comm_free(&rowComm);
    if (colComm != MPI_COMM_NULL)
        MPI_Comm_free(&colComm);
    if (cartComm != MPI_COMM_NULL)
        MPI_Comm_free(&cartComm);

    MPI_Finalize();
    return 0;
}
