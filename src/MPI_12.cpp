#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>

// Usage:
//   MPI_12 <numIterations> [gridRows gridCols]

static double computeAverage(double value) {
    return value;
}

static void printLineOnRoot(int worldRank, const std::string& line) {
    if (worldRank == 0) std::cout << line << std::endl;
}

static std::pair<int, int> chooseGridDims(int worldSize, int requestedRows, int requestedCols) {
    if (requestedRows > 0 && requestedCols > 0) {
        if (requestedRows * requestedCols == worldSize)
            return { requestedRows, requestedCols };
    }
    int approx = static_cast<int>(std::floor(std::sqrt(static_cast<double>(worldSize))));
    for (int r = approx; r >= 1; --r) {
        if (worldSize % r == 0) {
            return { r, worldSize / r };
        }
    }
    return { 1, worldSize };
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldSize = 1;
    int worldRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    int numIterations = 200;
    int requestedRows = 0, requestedCols = 0;
    if (argc >= 2) numIterations = std::max(1, std::atoi(argv[1]));
    if (argc >= 4) {
        requestedRows = std::max(1, std::atoi(argv[2]));
        requestedCols = std::max(1, std::atoi(argv[3]));
    }

    auto gridDims = chooseGridDims(worldSize, requestedRows, requestedCols);
    int gridRows = gridDims.first;
    int gridCols = gridDims.second;

    // 1) Cartesian (non-periodic)
    // 2) Torus (periodic both dims)
    // 3) Graph (custom adjacency)
    // 4) Star (center connected to all)

    auto measureAllreduceAvg = [&](MPI_Comm comm, int iterations, double& outFinalGlobal) -> double {
        if (comm == MPI_COMM_NULL) {
            outFinalGlobal = 0.0;
            return -1.0;
        }
        double localValue = static_cast<double>(worldRank + 1);
        
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();
        double globalValue = 0.0;
        for (int it = 0; it < iterations; ++it) {
            double iterValue = localValue + 1e-7 * it;
            MPI_Allreduce(&iterValue, &globalValue, 1, MPI_DOUBLE, MPI_SUM, comm);
        }
        double t1 = MPI_Wtime();
        double localElapsed = t1 - t0;
        double maxElapsed = 0.0;
        MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        double reducedFinal = globalValue;
        double finalGlobalOnWorldRoot = 0.0;
        MPI_Reduce(&reducedFinal, &finalGlobalOnWorldRoot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        outFinalGlobal = finalGlobalOnWorldRoot;
        if (maxElapsed <= 0.0)
            return 0.0;
        return maxElapsed / static_cast<double>(iterations);
        };

    // --- 1) Cartesian ---
    MPI_Comm cartComm = MPI_COMM_NULL;
    {
        int dims[2] = { gridRows, gridCols };
        int periods[2] = { 0, 0 };
        int reorder = 1;
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartComm);
    }

    if (cartComm != MPI_COMM_NULL) {
        int cartRank = -1;
        MPI_Comm_rank(cartComm, &cartRank);
        int coords[2] = { 0, 0 };
        MPI_Cart_coords(cartComm, cartRank, 2, coords);
        int left = MPI_PROC_NULL, right = MPI_PROC_NULL, up = MPI_PROC_NULL, down = MPI_PROC_NULL;
        MPI_Cart_shift(cartComm, 1, 1, &left, &right);
        MPI_Cart_shift(cartComm, 0, 1, &up, &down);
        std::cout << "CART,worldRank=" << worldRank << ",cartRank=" << cartRank
            << ",coords=" << coords[0] << "x" << coords[1]
            << ",neighbors(left,right,up,down)=" << left << "," << right << "," << up << "," << down << std::endl;
    }
    else {
        if (worldRank == 0)
            std::cout << "CART,creation_failed" << std::endl;
    }

    double finalGlobal = 0.0;
    double avgTimeCart = -1.0;
    if (cartComm != MPI_COMM_NULL) {
        avgTimeCart = measureAllreduceAvg(cartComm, numIterations, finalGlobal);
    }

    if (worldRank == 0) {
        std::ostringstream oss;
        oss << "MPI_12,cart," << gridRows << "," << gridCols << "," << worldSize << ","
            << (cartComm != MPI_COMM_NULL ? "1" : "0") << ","
            << std::fixed << std::setprecision(6) << avgTimeCart << "," << std::setprecision(12) << finalGlobal;
        std::cout << oss.str() << std::endl;
    }

    // --- 2) Torus (periodic Cartesian) ---
    MPI_Comm torusComm = MPI_COMM_NULL;
    {
        int dims[2] = { gridRows, gridCols };
        int periods[2] = { 1, 1 };
        int reorder = 1;
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &torusComm);
    }

    if (torusComm != MPI_COMM_NULL) {
        int torusRank; MPI_Comm_rank(torusComm, &torusRank);
        int coords[2]; MPI_Cart_coords(torusComm, torusRank, 2, coords);
        int left = MPI_PROC_NULL, right = MPI_PROC_NULL, up = MPI_PROC_NULL, down = MPI_PROC_NULL;
        MPI_Cart_shift(torusComm, 1, 1, &left, &right);
        MPI_Cart_shift(torusComm, 0, 1, &up, &down);
        std::cout << "TORUS,worldRank=" << worldRank << ",torusRank=" << torusRank
            << ",coords=" << coords[0] << "x" << coords[1]
            << ",neighbors(left,right,up,down)=" << left << "," << right << "," << up << "," << down << std::endl;
    }
    else {
        if (worldRank == 0)
            std::cout << "TORUS,creation_failed" << std::endl;
    }

    double avgTimeTorus = -1.0;
    finalGlobal = 0.0;
    if (torusComm != MPI_COMM_NULL) {
        avgTimeTorus = measureAllreduceAvg(torusComm, numIterations, finalGlobal);
    }
    if (worldRank == 0) {
        std::ostringstream oss;
        oss << "MPI_12,torus," << gridRows << "," << gridCols << "," << worldSize << ","
            << (torusComm != MPI_COMM_NULL ? "1" : "0") << ","
            << std::fixed << std::setprecision(6) << avgTimeTorus << "," << std::setprecision(12) << finalGlobal;
        std::cout << oss.str() << std::endl;
    }

    // --- 3) Graph topology (custom adjacency) ---
    MPI_Comm graphComm = MPI_COMM_NULL;
    std::vector<int> index;
    std::vector<int> edges;
    {
        index.resize(worldSize);
        edges.clear();
        for (int r = 0; r < worldSize; ++r) {
            std::vector<int> neighbors;
            int n1 = (r - 1 + worldSize) % worldSize;
            int n2 = (r + 1) % worldSize;
            int n3 = (r - 2 + worldSize) % worldSize;
            int n4 = (r + 2) % worldSize;
            
            neighbors.push_back(n1);
            if (n2 != n1)
                neighbors.push_back(n2);
            if (n3 != n1 && n3 != n2)
                neighbors.push_back(n3);
            if (n4 != n1 && n4 != n2 && n4 != n3)
                neighbors.push_back(n4);
            for (int nb : neighbors) {
                edges.push_back(nb);
            }
            index[r] = static_cast<int>(edges.size());
        }
        MPI_Graph_create(MPI_COMM_WORLD, worldSize, index.data(), edges.data(), /*reorder=*/0, &graphComm);
    }

    if (graphComm != MPI_COMM_NULL) {
        int graphRank; MPI_Comm_rank(graphComm, &graphRank);
        int neighborCount = 0;
        MPI_Graph_neighbors_count(graphComm, graphRank, &neighborCount);
        std::vector<int> neighborList(static_cast<size_t>(neighborCount));
        MPI_Graph_neighbors(graphComm, graphRank, neighborCount, neighborList.data());
        std::cout << "GRAPH,worldRank=" << worldRank << ",graphRank=" << graphRank
            << ",neighborsCount=" << neighborCount << ",neighbors=";
        for (size_t i = 0; i < neighborList.size(); ++i) {
            if (i)
                std::cout << ";";
            std::cout << neighborList[i];
        }
        std::cout << std::endl;
    }
    else {
        if (worldRank == 0)
            std::cout << "GRAPH,creation_failed" << std::endl;
    }

    double avgTimeGraph = -1.0;
    finalGlobal = 0.0;
    if (graphComm != MPI_COMM_NULL) {
        avgTimeGraph = measureAllreduceAvg(graphComm, numIterations, finalGlobal);
    }
    if (worldRank == 0) {
        std::ostringstream oss;
        oss << "MPI_12,graph,0,0," << worldSize << ","
            << (graphComm != MPI_COMM_NULL ? "1" : "0") << ","
            << std::fixed << std::setprecision(6) << avgTimeGraph << "," << std::setprecision(12) << finalGlobal;
        std::cout << oss.str() << std::endl;
    }

    // --- 4) Star topology (center = rank 0 connected to all others) ---
    MPI_Comm starComm = MPI_COMM_NULL;
    {
        std::vector<int> starIndex(worldSize);
        std::vector<int> starEdges;
        for (int r = 0; r < worldSize; ++r) {
            if (r == 0) {
                for (int nb = 1; nb < worldSize; ++nb) {
                    starEdges.push_back(nb);
                }
            }
            else {
                starEdges.push_back(0);
            }
            starIndex[r] = static_cast<int>(starEdges.size());
        }
        MPI_Graph_create(MPI_COMM_WORLD, worldSize, starIndex.data(), starEdges.data(), /*reorder=*/0, &starComm);
    }

    if (starComm != MPI_COMM_NULL) {
        int starRank; MPI_Comm_rank(starComm, &starRank);
        int neighborCount = 0;
        MPI_Graph_neighbors_count(starComm, starRank, &neighborCount);
        std::vector<int> neighborList(static_cast<size_t>(neighborCount));
        MPI_Graph_neighbors(starComm, starRank, neighborCount, neighborList.data());
        std::cout << "STAR,worldRank=" << worldRank << ",starRank=" << starRank
            << ",neighborsCount=" << neighborCount << ",neighbors=";
        for (size_t i = 0; i < neighborList.size(); ++i) {
            if (i) std::cout << ";";
            std::cout << neighborList[i];
        }
        std::cout << std::endl;
    }
    else {
        if (worldRank == 0)
            std::cout << "STAR,creation_failed" << std::endl;
    }

    double avgTimeStar = -1.0;
    finalGlobal = 0.0;
    if (starComm != MPI_COMM_NULL) {
        avgTimeStar = measureAllreduceAvg(starComm, numIterations, finalGlobal);
    }
    if (worldRank == 0) {
        std::ostringstream oss;
        oss << "MPI_12,star,0,0," << worldSize << ","
            << (starComm != MPI_COMM_NULL ? "1" : "0") << ","
            << std::fixed << std::setprecision(6) << avgTimeStar << "," << std::setprecision(12) << finalGlobal;
        std::cout << oss.str() << std::endl;
    }

    if (cartComm != MPI_COMM_NULL)
        MPI_Comm_free(&cartComm);
    if (torusComm != MPI_COMM_NULL)
        MPI_Comm_free(&torusComm);
    if (graphComm != MPI_COMM_NULL)
        MPI_Comm_free(&graphComm);
    if (starComm != MPI_COMM_NULL)
        MPI_Comm_free(&starComm);

    MPI_Finalize();
    return 0;
}
