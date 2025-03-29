/*
 * Base class for GPU optimization of 2D point sets
 * Contains utility routines such as loading and saving
 *
 * 2022-09-13: Created by Abdalla Ahmed from older files
 */

#ifndef GPU_Point_Set_H
#define GPU_Point_Set_H

#include "common.h"
#include "float.h"
#include "aggregate.h"
#include <string>
#include <fstream>
#include <cstdio>
#include <cctype> // for std::tolower

//#include <algorithm>                                                            // this was needed for transform, but is causing compilation errors with nvcc

// =============================================================================
// GPU Global Functions
// =============================================================================

// -----------------------------------------------------------------------------
// Compute Distance Between Two Sets
// -----------------------------------------------------------------------------

__global__
void computeDistSq(int N, Point *p, Point *ref, Float *rr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Float dx = p[i].x - ref[i].x;
    Float dy = p[i].y - ref[i].y;
    rr[i] = dx * dx + dy * dy;
}

__global__
void computeDistSq(int N, Point *p, Point *ref, Float *rr, Float one) {         // Toroidal version
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    Float dx = fabsF(p[i].x - ref[i].x);
    if (dx >= Float(0.5) * one) dx = one - dx;
    Float dy = fabsF(p[i].y - ref[i].y);
    if (dy >= Float(0.5) * one) dy = one - dy;
    rr[i] = dx * dx + dy * dy;
}

// =============================================================================
// Base Point Set Class
// =============================================================================

class GPUPointSet {
protected:
    int N;                                                                      // Number of points
    std::vector<Point> p;                                                       // List of points
    Point *p1, *p2, *pRef;                                                      // Three buffers for holding and processing the points
    Float *tmp1, *tmp2;                                                         // Two Float buffers for processing
    Float s;                                                                    // Scaling factor of gradient steps
    int NB;                                                                     // Number of cuda thread blocks
    int w, h;                                                                   // Width and height of domain
    int n;                                                                      // Size, max(w, h), needed to maintain aspect ratio
    void init();                                                                // Initialize parameters and buffers
    bool toroidal;                                                              // Is the domain toroidal?
    // -------------------------------------------------------------------------
    // Recording parameters
    // -------------------------------------------------------------------------
    Float outBlackRatio;                                                        // Ratio of black in produced renderings; 1 is full black
    std::string frameBaseName;                                                  // For saving frames during optimization
    int snapshotCondition = 1;                                                  // 1: visible change, 2: powers of 2 iteration index, otherwise: all iterations
    int frameFormat = 0;                                                        // 1: txt, 2: eps, 4: png, 8: pdf
    Float noticeableDisplacement = 1;                                           // Record a frame if this threshold is exceeded
    Float outputResolution;                                                     // For deciding if there is a noticeable difference
    int frameIndex;                                                             // Index of output frame
    int frameCount;                                                             // If set, frames would cycle through this loop size, rewriting older ones
    int accumIterations;                                                        // Statistics
    int W, H;                                                                   // Width and height of recorded frames
    Float margin;                                                               // Margin in plots, given as a ratio to domain size
    Float unit;                                                                 // Unit of recorded frames
    Float maxDistance(Point *p1, Point *pRef);                                  // Return max distance between two states
public:
    static const int BS = 256;                                                  // CUDA thread block size
    GPUPointSet(int N, int w = 1, int h = 1);
    GPUPointSet(char *txtFileName, int w = 1, int h = 1);                       // Load a given point set
    void setTimeStep(Float v) {s = v;};                                         // Set time step relative to default
    void setOutBlackRatio(Float v) { outBlackRatio = v; };
    void setFrameBaseName(char *v) { frameBaseName = v; };
    void setSnaptshotCondition(int v) { snapshotCondition = v; };
    void setFrameFormat(int v) { frameFormat = v; };
    void setFrameCount(int v) { frameCount = v; };
    void setMargin(Float v) { margin = v; };
    void setToroidal(bool value) { toroidal = value; }
    void setFrameSize(int W, int H);
    void setNoticeableDisplacement(Float v) { noticeableDisplacement = v; }
    void saveEPS(std::string fileName);
    void saveTXT(std::string fileName);
    void savePNG(std::string fileName);
    void savePDF(std::string fileName);
    void save(std::string fileName);
    void snapshot(int iteration);
    ~GPUPointSet();
};

// -----------------------------------------------------------------------------
// Construction and Destruction:
// -----------------------------------------------------------------------------

GPUPointSet::GPUPointSet(int N, int w, int h) : N(N), w(w), h(h) {
    init();
};

GPUPointSet::GPUPointSet(char *txtFileName, int w, int h) : w(w), h(h) {
    std::fstream file(txtFileName);
    file >> N;
    init();                                                                     // Now that N is known
    Float x, y;
    for (int i = 0; i < N; i++) {
        file >> x >> y;
        if (file.eof()) {
            fprintf(stderr, "Failed to load all points\n");
            exit(1);
        }
        p[i] = {x * n, y * n};
    }
    file.close();
}

void GPUPointSet::init() {
    p.resize(N);
    cudaMalloc(&p1  , N * sizeof(Point));                                       // Allocate memory for points
    cudaMalloc(&p2  , N * sizeof(Point));                                       // Allocate memory for output points
    cudaMalloc(&pRef, N * sizeof(Point));                                       // Allocate memory for reference frame
    cudaMalloc(&tmp1, N * sizeof(Float));                                       // Allocate memory for amplitudes
    cudaMalloc(&tmp2, N * sizeof(Float));                                       // Allocate memory for amplitudes
    NB = (N + BS - 1) / BS;                                                     // Number of thread blocks
    setTimeStep(1);
    // -------------------------------------------------------------------------
    // Recording parameters
    // -------------------------------------------------------------------------
    n = std::max(w, h);                                                         // Square domain size to preserve aspect ratio
    unit = 1. / n;
    setFrameSize(w, h);                                                         // Default saved frame size to input image size
    setFrameFormat(4);                                                          // Default frame saving format to png
    accumIterations = 0;                                                        // Reset recording parameters
    frameIndex = 0;
    frameCount = 0;                                                             // Use iteration number for indexing
    margin = 0;
    toroidal = false;                                                           // Default
}

GPUPointSet::~GPUPointSet() {
    cudaFree(p1);
    cudaFree(p2);
    cudaFree(pRef);
    cudaFree(tmp1);
    cudaFree(tmp2);
}

// -----------------------------------------------------------------------------
// Return travelled distance between two states
// -----------------------------------------------------------------------------

Float GPUPointSet::maxDistance(Point *p1, Point *pRef) {
    Float rrMax;
    if (toroidal) {                                                             // Toroidal?
        computeDistSq<<<NB,BS>>>(N, p1, pRef, tmp1, n);
    }
    else {
        computeDistSq<<<NB,BS>>>(N, p1, pRef, tmp1);
    }
    Float *rrMaxGPU = aggregateMax(tmp1, N, tmp2);
    cudaMemcpy(&rrMax, rrMaxGPU, sizeof(Float), cudaMemcpyDeviceToHost);
    return sqrt(rrMax) / outputResolution;
}

// -----------------------------------------------------------------------------
// Record frame if condition is met
// -----------------------------------------------------------------------------

void GPUPointSet::setFrameSize(int W, int H) {
    this->W = W;
    this->H = H;
    outputResolution = (Float(n)/max(W, H));                                        // Pixel in output resolution
}

void GPUPointSet::snapshot(int iteration) {
    accumIterations++;
    switch (snapshotCondition) {
        case 1: {                                                               // Visible
            Float dmax = maxDistance(p1, pRef);
            fprintf(
                stderr,
                ", Frame %5d: accum max displacement = %19.17f in %4d iterations",
                frameIndex, dmax, accumIterations
            );
            if (iteration && dmax < noticeableDisplacement) return;                              // No visible difference to recorded
            // -----------------------------------------------------------------
            // Now we have a new frame:
            // -----------------------------------------------------------------
            cudaMemcpy(pRef, p1, N * sizeof(Point), cudaMemcpyDeviceToDevice);  // Update reference
            accumIterations = 0;                                                // Reset iterations counter
            break;
        }
        case 2: if (iteration & (iteration - 1)) return;                        // Not a power of two
    }
    if (frameFormat) {                                                          // Otherwise the function may be called just for statistics.
        cudaMemcpy(p.data(), p1, N * sizeof(Point), cudaMemcpyDeviceToHost);    // Copy points to CPU memory
        char serial[10];
        int seqNo = frameCount ? frameIndex % frameCount : iteration;
        sprintf(serial, "%08d", seqNo);
        if (frameFormat & 1) {
            saveTXT(frameBaseName + serial + ".txt");
        }
        if (frameFormat & 2) {
            saveEPS(frameBaseName + serial + ".eps");
        }
        if (frameFormat & 4) {
            savePNG(frameBaseName + serial + ".png");
        }
        if (frameFormat & 8) {
            savePDF(frameBaseName + serial + ".pdf");
        }
        if (frameFormat & 16) {
            savePNG("-");
        }
    }
    frameIndex++;
}

// -----------------------------------------------------------------------------
// Saving Point Sets
// -----------------------------------------------------------------------------

void GPUPointSet::save(std::string fileName) {
    std::string extension = fileName.substr(fileName.length() - 4, 4);
//     transform(
//         extension.begin(), extension.end(), extension.begin(), ::tolower
//     );
    for (char& c : extension) {
        c = std::tolower(c);
    }
    if      (extension == ".txt") { saveTXT(fileName.c_str()); }
    else if (extension == ".eps") { saveEPS(fileName.c_str()); }
    else if (extension == ".pdf") { savePDF(fileName.c_str()); }
    else if (extension == ".png") { savePNG(fileName.c_str()); }
    else {
        fprintf(stderr, "Unknown file extension %s\n", extension.c_str());
    }
}

void GPUPointSet::saveTXT(std::string fileName) {
    FILE *file = fopen(fileName.c_str(), "w");
    fprintf(file, "%d\n", N);
    for (int i = 0; i < N; i++) {
        fprintf(file, "%0.32f %0.32f\n", unit * p[i].x, unit * p[i].y);
    }
    fclose(file);
}

void GPUPointSet::saveEPS (std::string fileName) {
    FILE *file = fopen(fileName.c_str(), "w");
    fprintf(
        file,
        "%%!PS-Adobe-3.0 EPSF-3.0\n"
        "%%%%BoundingBox: 0 0 %d %d\n"
        "/N %d def\n"
        "/w %d def\n"
        "/h %d def\n"
        "/n %d def\n"
        "/blackRatio %f def\n"
        "/margin %f def\n"
        "/PI 3.141592654 def\n"
        "/r w h mul n n mul div N div blackRatio mul PI div sqrt def\n"
        "/p {r 0 360 arc fill} def\n"
        "0 setlinewidth\n"
        "%d 10 mul dup scale\n"
        "1 1 margin 2 mul add div dup scale margin dup translate\n",
        10 * W, 10 * H, N, w, h, n, outBlackRatio, margin, max(W, H)
    );
    for (int i = 0; i < N; i++) {
        fprintf(file, "%0.17f %0.17f p\n", unit * p[i].x, unit * p[i].y);
    }
    fprintf(file, "showpage\n");
    fclose(file);
}

void GPUPointSet::savePNG (std::string fileName) {
    cairo_surface_t *surface = cairo_image_surface_create(
        CAIRO_FORMAT_RGB24, W, H
    );
    cairo_t *cr = cairo_create(surface);
    int scale = max(H, W);
    Float area = min(w, h) / Float(n);
    const double r = sqrt(area * outBlackRatio / N / M_PI);
    cairo_identity_matrix(cr);
    cairo_translate(cr, 0, H);
    cairo_scale(cr, scale, -scale);
    cairo_scale(cr, 1./(1 + 2 * margin), 1./(1 + 2 * margin));
    cairo_translate(cr, margin, margin);
    cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
    //cairo_set_source_rgba(cr, 0, 0, 0, 0);
    cairo_paint(cr);
    //cairo_set_source_rgba(cr, 0, 0, 0, 1);
    cairo_set_source_rgb(cr, 0, 0, 0);
    for (int i = 0; i < N; ++i) {
        cairo_arc(cr, unit * p[i].x, unit * p[i].y, r, 0, 2 * M_PI);
        cairo_fill(cr);
    }
    cairo_destroy (cr);
    if (fileName == "-") {
        cairo_surface_write_to_png_stream(surface, &png2stdout, NULL);
    }
    else {
        cairo_surface_write_to_png(surface, fileName.c_str());
    }
    cairo_surface_destroy(surface);
}

void GPUPointSet::savePDF (std::string fileName) {
    cairo_surface_t *surface = cairo_pdf_surface_create(
        fileName.c_str(), 4000, 4000
    );
    cairo_t *cr = cairo_create(surface);
    int scale = max(H, W);
    Float area = min(w, h) / Float(n);
    const double r = sqrt(area * outBlackRatio / N / M_PI);
    cairo_identity_matrix(cr);
    cairo_translate(cr, 0, 4000);
    cairo_scale(cr, 4000, -4000);
    cairo_scale(cr, 1./(1 + 2 * margin), 1./(1 + 2 * margin));
    cairo_translate(cr, margin, margin);
    cairo_set_source_rgba(cr, 0, 0, 0, 1);
    for (int i = 0; i < N; ++i) {
        cairo_arc(cr, unit * p[i].x, unit * p[i].y, r, 0, 2 * M_PI);
        cairo_fill(cr);
    }
    cairo_show_page(cr);
    cairo_destroy (cr);
    cairo_surface_destroy (surface);
}


#endif                                                                          // GPU_Point_Set_H
