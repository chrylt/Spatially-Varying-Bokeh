/*
 */

// This must precede inclusion of spectrum.h
#define USE_DOUBLE

#include "spectrum.h"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <getopt.h>
#include <string>
#include <fstream>

Float scale = 0.5;                                                              // This is the default of PSA
bool loglog = false;
bool allBins = false;

const char *USAGE_MESSAGE = "Usage: %s [options] file1, [file2, ..]\n"
"Options:\n"
"   -c <cycles>         Default is 2\n"
"   -o <fileName>       Plot to fileName\n"
"   -w <width>          Default is 10 * sqrt(N)\n"
"   -W                  The points are weighted. "
                        "Format: x y weight\n"
"   -r <fileName>       Outputs a radial power plot\n"
"   -s <freq. scale>    For radial power; default is 0.5, as in PSA\n"
"   -l                  Use log-log scale for radial power spectrum plot\n"
"   -x <minFrequency>   Default is 0\n"
"   -X <maxFrequency>   Default is largest harmonic\n"
"   -y <minAmplitude>   Default is 0\n"
"   -Y <maxAmplitude>   Default is 4.2\n"
"   -t <tikz>           Append tikz instructions to plot\n"
"   -e                  Report evacuated energy\n"
"   -A <average>        Set the DC peak\n"
;

int main(int argc, char** argv) {
    int opt;                                                                    // For use by getopt, the command line options utility
    std::string plotFileName;
    std::string rpFileName;
    Float cycles = 5;
    int width = 0;
    bool reportEnergy = false;
    bool weighted = false;
    while ((opt = getopt(argc, argv, "c:o:r:w:s:lax:X:y:Y:t:eWA:")) != -1) {
        switch (opt) {
            case 'c': cycles = atof(optarg); break;
            case 'o': plotFileName = optarg; break;
            case 'r': rpFileName = optarg; break;
            case 'w': width = atoi(optarg); break;
            case 's': scale = atof(optarg); break;
            case 'l': loglog = true; break;
            case 'a': allBins = true; break;
            case 'x': xmin = atof(optarg); break;
            case 'X': xmax = atof(optarg); break;
            case 'y': ymin = atof(optarg); break;
            case 'Y': ymax = atof(optarg); break;
            case 't': userInstructions = optarg; break;
            case 'e': reportEnergy = true; break;
            case 'W': weighted = true; break;
            case 'A': average = atof(optarg); break;
            default: fprintf(stderr, USAGE_MESSAGE, argv[0]); exit(1);
        }
    }
    if (optind >= argc) {
        fprintf(stderr, USAGE_MESSAGE, argv[0]); exit(1);
    }
    int n = argc - optind;
    int N;
    std::fstream file(argv[optind]);
    file >> N;
    file.close();
    std::vector<PointFloat> p(n * N);
    std::vector<Float> wt;
    if (weighted) wt.resize(n * N);
    for (int fileNo = 0, pointNo = 0; fileNo < n; fileNo++) {
        file.open(argv[optind + fileNo]);
        int N_fileNo;
        file >> N_fileNo;
        if (N_fileNo != N) {
            fprintf(
                stderr, "Number of points in %s is different from %s\n",
                argv[optind], argv[optind + fileNo]
            );
            exit(1);
        }
        for (int i = 0; i < N; i++) {
            file >> p[pointNo].x >> p[pointNo].y;
            if (weighted) file >> wt[pointNo];
            if (file.eof()) {
                fprintf(
                    stderr, "Failed to load all points from %s\n",
                    argv[optind + fileNo]
                );
                exit(2);
            }
            pointNo++;
        }
        file.close();
    }
    if (!width) width = 2 * cycles * sqrt(N);
    if (width & 1) width -= 1;
    clock_t t0 = clock();
    PointFloat *p_gpu;
    Float *wt_gpu = NULL;
    int dataSize = n * N * sizeof(PointFloat);
    cudaMalloc(&p_gpu, dataSize);
    cudaMemcpy(p_gpu, p.data(), dataSize, cudaMemcpyHostToDevice);
    if (weighted) {
        cudaMalloc(&wt_gpu, n * N * sizeof(Float));
        cudaMemcpy(wt_gpu, p.data(),
        n * N * sizeof(Float), cudaMemcpyHostToDevice);
    }
    std::vector<Float> spectrum = powerSpectrum(p_gpu, wt_gpu, n, N, width);
    cudaFree(p_gpu);
    if (weighted) cudaFree(wt_gpu);
    if (!plotFileName.empty()) {
        plotSpectrum(spectrum, width, plotFileName.c_str());
    }
    if (!rpFileName.empty()) {
        plotRadialPower(
            spectrum, width, rpFileName.c_str(), N, scale, loglog, allBins
        );
    }
    if (reportEnergy) {
        printf("%f\n", bnLoss(spectrum, width, N));
    }
    clock_t t1 = clock();
    Float totalTime = (Float)(t1 - t0) / CLOCKS_PER_SEC;
    fprintf(
        stderr, "Total time = %.6fs, averaged %d sets of %d points each\n",
        totalTime, n, N
    );
}
