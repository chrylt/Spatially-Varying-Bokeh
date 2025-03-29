/*
 * Adaptive GBN sampling of a density map.
 * This is a minimal code for testing.
 * The density map is expected as a .pgm file.
 * 2022-09-21: Final revision by Abdalla Ahmed for inclusion with the paper
 * 2022-10-02: Used weighted average instead of density
 * 2022-10-26: Used impulses for pixels
 * 2023-08-23: Used a new kernel adaptation model using nearest neighbor
 * 2023-09-28: Restored modeling pixels as impulses (Dirac deltas)
 * 2023-09-29: Changed default energy sigma to sqrt(M_PI/4)
 * 2023-10-06: Replaced amplitudes by volumes
 */

#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <cmath>
#include <string>
#include <signal.h>
#include <stdint.h>
#include <numeric>                                                              // For std::accumulate
//#include <algorithm>                                                            // For std::max_element

//#define USE_DOUBLE

#include "2048.h"
#include "gpu-point-set.h"

// =============================================================================
// Utils
// =============================================================================

// =============================================================================
// Global Parameters
// =============================================================================

static bool interruptFlag = false;                                              // To stop the optimization manually
bool areaPixels = false;

// =============================================================================
// CUDA Kernel Optimization: Alternative to Algorithm 2 in the paper
// =============================================================================

__global__
void optimizeKernels(
    int N,                                                                      // Number of points
    Point *p,                                                                   // Point coordinates
    Float *v,                                                                   // Shaping factors
    Float unitAreaInv                                                           // Exponent coefficient: Sigma squared, multiplied by 2, inversed, negated
) {                                                                             // Scale amplitudes only
    int i = blockIdx.x * blockDim.x + threadIdx.x;                              // Thread index used as point index
    if (i >= N) return;                                                         // Needed where N is not a multiple of thread block size
    Float ddMin(1e12);                                                          // squared distance to nearest neighbor; initialized to a sufficiently large value
    for (int j = 0; j < N; j++) {
        if (j == i) continue;                                                   // Skip self
        Float dx = p[j].x - p[i].x;
        Float dy = p[j].y - p[i].y;
        Float dd = dx * dx + dy * dy;
        ddMin = min(ddMin, dd);
    }
    v[i] = ddMin * unitAreaInv;                                                 // Volume is a factor of this
}


// =============================================================================
// CUDA Position Optimization
// =============================================================================

__global__
void relaxOnce(
    int N,                                                                      // Number of sample points
    Point *pIn,                                                                 // Input point coordinates
    Point *pOut,                                                                // Output coordinates
    Float *v,                                                                   // Relative volumes
    int w, int h, Float *bmp,                                                   // Image parameters and pixel data
    Float sigma,                                                                // Sigma for unit area
    Float t,                                                                    // Gradient time step
    Float vPixel,
    bool areaPixels = false
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;                              // Thread id
    if (i >= N) return;                                                         // Needed when number of points in not a multiple of block size
    Vector grad = {ZERO, ZERO};                                                 // Gradient
    //t *= Float(0.25);                                                           // Our default, mostly empirical

    t *= 0.5 * v[i];


    //t *= sqrtF(v[i]);                                                           // Scale timeStep to local scale
    Float ss2InvNeg = -1 / (2 * sigma * sigma);
    // -------------------------------------------------------------------------
    // Attraction of image
    // -------------------------------------------------------------------------
    Float support = SUPPORT * sigma * sqrtF(v[i] + vPixel);
    int Xmin = max(floor(pIn[i].x - support), Float(    0));
    int Xmax = min( ceil(pIn[i].x + support), Float(w - 1));
    int Ymin = max(floor(pIn[i].y - support), Float(    0));
    int Ymax = min( ceil(pIn[i].y + support), Float(h - 1));
    Float a = 1 / (v[i] + vPixel);
    if (areaPixels) {
        Float c = -a * ss2InvNeg;
        Float sqrtC = sqrt(c);
        for (int Y = Ymin; Y <= Ymax; Y++) {
            Float yb(pIn[i].y - Y - 1), yt(yb + 1);
            Float gy = expF(-c * yt * yt) - exp(-c * yb * yb);
            Float ey = erfF(sqrtC * yb) - erf(sqrtC * yt);
            for (int X = Xmin; X <= Xmax; X++) {
                int j = Y * w + X;
                if (!bmp[j]) continue;
                Float xl(pIn[i].x - X - 1), xr(xl + 1);
                Float gx = exp(-c * xr * xr) - exp(-c * xl * xl);
                Float ex = erf(sqrtC * xl) - erf(sqrtC * xr);
                grad.x -= bmp[j] * gx * ey;
                grad.y -= bmp[j] * gy * ex;
            }
        }
        grad.x *= sqrt(M_PI) / (4 * pow(c, 1.5));
        grad.y *= sqrt(M_PI) / (4 * pow(c, 1.5));
    }
    else {
        Float y(pIn[i].y - HALF), x(pIn[i].x - HALF);
        Float a_x_ss2InvNeg = a * ss2InvNeg;
        for (int Y = Ymin; Y <= Ymax; Y++) {
            //Float dy = pIn[i].y - (Y + HALF);
            Float dy = y - Y;
            for (int X = Xmin; X <= Xmax; X++) {
                int j = Y * w + X;
                if (!bmp[j]) continue;
                //Float dx = pIn[i].x - (X + HALF);
                Float dx = x - X;
                Float rr = dx * dx + dy * dy;
                Float g = expF(a_x_ss2InvNeg * rr);
                g *= bmp[j];
                grad.x -= dx * g;
                grad.y -= dy * g;
            }
        }
    }
    grad.x *= a * a;
    grad.y *= a * a;
    // -------------------------------------------------------------------------
    // Repulsion of other points:
    // -------------------------------------------------------------------------
    for (int j = 0; j < N; j++) {
        if (j == i) continue;
        Float dx = pIn[i].x - pIn[j].x;
        Float dy = pIn[i].y - pIn[j].y;
        Float rr = dx * dx + dy * dy;
        Float a = 1 / (v[i] + v[j]);
        Float g = expF(a * ss2InvNeg * rr);
        g *= a * a;
        grad.x += dx * g;
        grad.y += dy * g;
    }
    // -------------------------------------------------------------------------
    // Clamp displacement to local neighborhood
    // -------------------------------------------------------------------------
    Float stepSq = t * t * (grad.x * grad.x + grad.y * grad.y);
    Float displacementMaxSq = Float(0.25) * sigma * sigma * v[i];
    if (stepSq > displacementMaxSq) {
        t *= sqrtF(displacementMaxSq / stepSq);
    }
    // -------------------------------------------------------------------------
    // Update point:
    // -------------------------------------------------------------------------
    Float x = pIn[i].x + t * grad.x;
    Float y = pIn[i].y + t * grad.y;
    if (
        x >= 0 && x < w && y >= 0 && y < h                                      // Restrict to image boundary. In theory it should not be needed
    ) {                                                                         // If new location is acceptible
        pOut[i] = {x, y};                                                       // Commit
    }
    else {                                                                      // Otherwise
        pOut[i] = pIn[i];                                                       // Fall back to old location
    }
}

// =============================================================================
// Adaptive Class. Common GPU-related items kept in a base class.
// =============================================================================

class GBNAdaptive : public GPUPointSet {
private:
    Float sigma;                                                                // Sigma of energy kernel
    Float sigmaPixel;                                                           // Filtering sigma of pixels; mainly for testing; e.g. 0 means Dirac deltas
    Float *bmp;                                                                 // List of pixels in GPU memory
    Float *v;                                                                   // Buffer for holding relative volumes
    int gradientScalingMode;                                                    // 0 (default): a[i].a[i], 1: a[i], 2: 1
    void init(const std::vector<Float> &pixelsFrom0To1);                        // Initialize parameters and buffers
public:
    GBNAdaptive(                                                                // Construct from scratch
        int N,                                                                  // Number of points
        Image img,                                                              // Density map
        int initialization = 0                                                  // 0: full-random, 1: weighted random, 2: 2048
    );
    GBNAdaptive(char *txtFileName, const Image img);                            // Load a given point set
    void setSigma(Float value);                                                 // Set sigma to a given value
    void setSigmaPixel(Float value) { sigmaPixel = value; };
    void setGradientScalingMode(int value) { gradientScalingMode = value; };
    void optimize(int iterations, Float terminateThreshold = 0);                // Apply optimization
    ~GBNAdaptive();
};

// -----------------------------------------------------------------------------
// Construct from scratch
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Alternative to std::max_element, cos <algorithm> causes compilation issues
// with nvcc
// -----------------------------------------------------------------------------
template <typename Iterator>
Iterator my_max_element(Iterator first, Iterator last) {
    if (first == last) return last;

    Iterator largest = first;
    ++first;
    for (; first != last; ++first) {
        if (*largest < *first) {
            largest = first;
        }
    }
    return largest;
}
// -----------------------------------------------------------------------------


GBNAdaptive::GBNAdaptive(
    int N, Image img, int initialization
) : GPUPointSet(N, img.w, img.h) {
    switch (initialization) {
        case 0: {
            for (int i = 0; i < N; i++) { p[i] = {rand(w), rand(h)}; }
            break;
        }
        case 1: default: {
            Float maxPixel = *my_max_element( // *std::max_element(
                img.pixels.begin(), img.pixels.end()
            );                                                                  // Highest pixel density; high is darker
            for (int i = 0; i < N; /*conditionally increment inside loop*/) {   // Insert points proportional to density using rejection sampling
                Float x = rand(w);                                              // Random x in [0, w)
                Float y = rand(h);                                              // Random y in [0, h)
                Float threshold = rand(maxPixel);                               // Random threshold
                if (img[int(y) * w + int(x)] > threshold) {                     // Is underlying pixel above threshold?
                    p[i++] = {x, y};                                            // If yes, insert a point at this location
                }
            }
            break;
        }
        case 2: {
            C2048 slicer(img);
            p = slicer.slice(N);
            break;
        }
    }
    init(img.pixels);                                                           // Common initialization tasks
}

// -----------------------------------------------------------------------------
// Load a given point set
// -----------------------------------------------------------------------------

GBNAdaptive::GBNAdaptive(
    char *txtFileName, Image img
) : GPUPointSet(txtFileName, img.w, img.h) {
    init(img.pixels);
}

// -----------------------------------------------------------------------------
// Initialize parameters and buffers
// -----------------------------------------------------------------------------

void GBNAdaptive::init(const std::vector<Float> &pixelsFrom0To1) {
    std::vector<Float> bmpCPU = pixelsFrom0To1;                                 // Create a local copy for processing
    Float mass = std::accumulate(bmpCPU.begin(), bmpCPU.end(), ZERO);           // Sum of all pixels
    outBlackRatio = mass / (w * h);                                             // Default output rendering black level to match input image
    for (int i = 0; i < w * h; i++) { bmpCPU[i] *= (N / mass); }                // Make mass of pixels equl to mass of points
    setSigma(1);
    setSigmaPixel(-1);                                                          // Same as points
    cudaMalloc(&bmp , h * w * sizeof(Float));                                   // Allocate memory for pixels
    cudaMalloc(&v  , N * sizeof(Float));                                        // Allocate memory for shaping factors
    setTimeStep(1);
    cudaMemcpy(
        bmp  , bmpCPU.data(), h * w * sizeof(Float), cudaMemcpyHostToDevice
    );                                                                          // Copy pixels to device memory
    setGradientScalingMode(0);
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------

GBNAdaptive::~GBNAdaptive() {
    cudaFree(bmp);                                                              // Free image buffer
    cudaFree(v);                                                                // Free shaping-factors buffer
}

// -----------------------------------------------------------------------------
// Set energy sigma
// -----------------------------------------------------------------------------

void GBNAdaptive::setSigma(Float value) {
    sigma = value                                                               // Input energy kernel sigma
            * sqrt(Float(h * w) / N)                                            // Scale to point area
            / sqrt(2)                                                           // This is the filtering kernel
            / sqrt(2)                                                           // Account for the pixels, 50% share
    ;
    fprintf(stderr, "Energy sigma set to = %f, %f pixels\n", value, sigma);
}

// -----------------------------------------------------------------------------
// Optimization Routine
// -----------------------------------------------------------------------------

void GBNAdaptive::optimize(int iterations, Float terminateThreshold) {
    cudaMemcpy(p1  , p.data(), N * sizeof(Point), cudaMemcpyHostToDevice);      // Copy points to device memory buffer
    cudaMemcpy(pRef,       p1, N * sizeof(Point), cudaMemcpyDeviceToDevice);    // Copy initial point locations as a reference
    if (!frameBaseName.empty()) snapshot(0);
    const Float unitAreaInv = N / Float(h * w);
    const Float vPixel = (
        sigmaPixel >= 0 ?                                                       // If there is a supplied sigma, mainly for experimentation
        (sigmaPixel/sigma) * (sigmaPixel/sigma) :                               // Then use it
        unitAreaInv                                                             // Default: use the same sigma as the points, noting that ddMin of pixels is 1
    );
    Float timeStep = s;
    for (int i = 0; i < iterations && !interruptFlag; i++) {
        fprintf(stderr, "\rIteration: %6d", i);
        // ---------------------------------------------------------------------
        // Kernel Optimization
        // ---------------------------------------------------------------------
        optimizeKernels<<<NB,BS>>>(N, p1, v, unitAreaInv);
        // ---------------------------------------------------------------------
        // Adjust time step according to current coverage
        // ---------------------------------------------------------------------
//         Float vAvg;
//         if (i < 100 || i % 100 == 0) {                                          // This does not have to be updated so frequently
//             cudaMemcpy(tmp1, v, N * sizeof(Float), cudaMemcpyDeviceToDevice);
//             invAll<<<NB,BS>>>(tmp1, N);
//             Float *vAvgGPU = aggregate(tmp1, N, tmp2);
//             cudaMemcpy(
//                 &vAvg, vAvgGPU, sizeof(Float), cudaMemcpyDeviceToHost
//             );
//             vAvg *= 1. / N;
//         }
//         fprintf(stderr, ", vAvg = %f", vAvg);
//         Float timeStep = s * vAvg;
        // ---------------------------------------------------------------------
        // Position Optimization
        // ---------------------------------------------------------------------
        relaxOnce<<<NB, BS>>>(
            N, p1, p2, v, w, h, bmp, sigma, timeStep, vPixel, areaPixels
        );
        std::swap(p1, p2);
        cudaDeviceSynchronize();
        Float dmax = maxDistance(p1, p2);
        fprintf(stderr, ", max displacement: %12.6e", dmax);
        if (dmax <= terminateThreshold) break;
        if (!frameBaseName.empty()) snapshot(i + 1);
    }
    fprintf(stderr, "\n");
    cudaMemcpy(p.data(), p1, N * sizeof(Point), cudaMemcpyDeviceToHost);        // Copy back to CPU memory
}



// =============================================================================
// User Interruption Handling
// =============================================================================

static void signalHandler(int signalCode) {
    fprintf(stderr, "Aborting ...\n");
    interruptFlag = true;
}

const char *USAGE_MESSAGE =
"Usage:"
" %s [options] <image.pgm> <point count> <iterations> "
"[output.abc output.efg]\n"
"Suppoted save types are .txt, .eps, .png, .pdf\n"
"Options:\n"
" -g <sigma>            User-supplied sigma\n"
" -G <sigma>            Filtering sigma of pixels; default is 0 (deltas)\n"
" -A                    Use area pixels; default is points\n"
" -t <time step>        Scale factor for gradient relative to default\n"
" -T <threshold>        Terminate optimization upon reaching a threshold step. "
                       "Requires computing distances, hence adds some delay.\n"
" -c <outBlackRatio>    Coverage ratio of black in renderings, "
                       "default is computed from image mean\n"
" -l <fileName>         Load points from txt file\n"
" -f <frameName>        Take snapshots named frameName%%07d.ext\n"
" -F <format>           Bitmask of frame name extensions: "
                       "1: txt, 2: eps, 4: png, 8: pdf; default is 4.\n"
" -W/-w <width>         Width of recorded frames; default is image width\n"
" -H/-h <height>        Height of recorded frames; default is image height\n"
" -C <condition>        Condition for frame recording: "
                       "1: visible change, 2: powers of 2 iteration index, "
                       "therwise: all iterations.\n"
" -Q <#frames>          Recycle through this number of frame indices; "
                       "default is to index by iteration numbers.\n"
" -d <threshold>        Record a frame if threshold is exceede, measured in "
                       "recording resolution. default is 1.\n"
" -M <margin>           Add a fractional margin in plots; default is not\n"
" -q <power>            Raise pixels to power; "
                       "use 2 pixels for line-based rendering\n"
" -I <initialization>   0: random, 1: weighted random (default), 2: 2048\n"
" -S <gradScalingMode>  0 (default): a[i].a[i], 1: a[i], 2: 1\n"
;


int main(int argc,char **argv) {
    int opt;
    char *inputFileName = NULL;
    char *frameBaseName = NULL;
    Float t = 0;
    Float sigma = 0;
    Float sigmaPixel(0);
    Float outBlackRatio = 0;
    int W = 0, H = 0;
    int snapshotCondition = 0;
    int frameFormat = -1;
    Float terminateThreshold = 0;
    int initialization = 1;
    Float margin = 0;
    Float noticeableDisplacement = 0;
    Float power = 1;
    int frameCount = 0;
    int gradientScalingMode = 0;
    while (
        (opt = getopt(argc, argv, "g:G:At:c:l:f:W:w:H:h:C:F:T:M:d:q:Q:I:S:")) != -1
    ) {
        switch (opt) {
            case 'g': sigma = atof(optarg); break;
            case 'G': sigmaPixel = atof(optarg); break;
            case 'A': areaPixels = true; break;
            case 't': t = atof(optarg); break;
            case 'c': outBlackRatio = atof(optarg); break;
            case 'l': inputFileName = optarg; break;
            case 'f': frameBaseName = optarg; break;
            case 'F': frameFormat = atoi(optarg); break;
            case 'W': case 'w': W = atoi(optarg); if (!H) H = W; break;
            case 'H': case 'h': H = atoi(optarg); if (!W) W = H; break;
            case 'C': snapshotCondition = atoi(optarg); break;
            case 'T': terminateThreshold = atof(optarg); break;
            case 'I': initialization = atoi(optarg); break;
            case 'M': margin = atof(optarg); break;
            case 'd': noticeableDisplacement = atof(optarg); break;
            case 'q': power = atof(optarg); break;
            case 'Q': frameCount = atoi(optarg); break;
            case 'S': gradientScalingMode = atoi(optarg); break;
            default: fprintf(stderr, USAGE_MESSAGE, argv[0]); exit(1);
        }
    }
    if (optind > argc - 3) {
        fprintf(stderr, USAGE_MESSAGE, argv[0]);
        exit(1);
    }
    signal(SIGINT, signalHandler);
    char *imageFileName = argv[optind++];
    int N = atoi(argv[optind++]);
    int iterations = atoi(argv[optind++]);
    Image img = loadPGM(imageFileName, power);
    GBNAdaptive gbn = (
        inputFileName ?
        GBNAdaptive(inputFileName, img) :
        GBNAdaptive(N, img, initialization)
    );
    if (sigma) gbn.setSigma(sigma);
    if (sigmaPixel) gbn.setSigmaPixel(sigmaPixel);
    if (t) gbn.setTimeStep(t);
    if (outBlackRatio) gbn.setOutBlackRatio(outBlackRatio);
    if (frameBaseName) gbn.setFrameBaseName(frameBaseName);
    if (snapshotCondition) gbn.setSnaptshotCondition(snapshotCondition);
    if (frameFormat >= 0) gbn.setFrameFormat(frameFormat);
    if (frameCount) gbn.setFrameCount(frameCount);
    gbn.setGradientScalingMode(gradientScalingMode);
    if (noticeableDisplacement) {
        gbn.setNoticeableDisplacement(noticeableDisplacement);
    }
    if (W) gbn.setFrameSize(W, H);
    if (margin) gbn.setMargin(margin);
    clock_t t0 = clock();
    gbn.optimize(iterations, terminateThreshold);
    clock_t t1 = clock();
    double totalTime = (double)(t1 - t0) / CLOCKS_PER_SEC;
    fprintf(stderr, "\ndone! Total time = %.6fs\n", totalTime);
    // =========================================================================
    // Save point sets
    // =========================================================================
    while (optind < argc) { gbn.save(argv[optind++]); }

}
