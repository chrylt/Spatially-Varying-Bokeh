/*
 * Common data types and parameters
 * 2023-07-30: Created by Abdalla Ahmed
 */

#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "cairo/cairo.h"
#include "cairo/cairo-pdf.h"
#include <random>

// Comment/uncomment this to use single/double precision:
// #define USE_DOUBLE
#ifdef USE_DOUBLE
    #define Float double
#else
    #define Float float
#endif

// =============================================================================
// Utils
// =============================================================================

std::random_device rd;                                                          // Used to obtain a seed for the random number engine
std::mt19937 rnd(rd());                                                         // Initialize random number generator
Float rand(Float x) {                                                           // Return a random number in [0, x)
    const Float res = 1. / (1ull << 32);
    return res * rnd() * x;
}

cairo_status_t png2stdout(
    void *closure, const unsigned char* data, unsigned int length
) {
    return (
        length == std::fwrite(data, sizeof(data[0]), length, stdout) ?
        CAIRO_STATUS_SUCCESS :
        CAIRO_STATUS_WRITE_ERROR
    );
}

// =============================================================================
// Data Structures
// =============================================================================

struct Vector {
    Float x, y;
    Float sq() { return x * x + y * y; };
    inline Vector operator+(Vector v) { return {x + v.x, y + v.y}; };
    inline void operator*=(Float s) { x *= s; y *= s; };
    inline void operator+=(Vector v) { x += v.x; y += v.y; };
};

inline Vector operator*(Vector v, Float s) { return {v.x * s, v.y * s}; };
inline Vector operator*(Float s, Vector v) { return {v.x * s, v.y * s}; };

typedef Vector Point;
typedef std::vector<Point> Points;

inline bool isInside(const Point &p, const Point &bl, const Point &tr) {
    return p.x > bl.x && p.y > bl.y && p.x < tr.x && p.y < tr.y;
}


// =============================================================================
// Image handling
// =============================================================================

struct Image {                                                                  // Data structure for holding image information
    int w, h;                                                                   // Width and height
    std::vector<Float> pixels;                                                  // Image data
    Float& operator[](int i) { return pixels[i]; };                             // Cast to pixel array
};

Image loadPGM(const char *fileName, Float power = 1) {
    FILE *pgmfile = fopen(fileName, "r");
    if (!pgmfile) {
        fprintf(stderr, "Failed to open file %s\n", fileName);
        exit(1);
    }
    Image img;
    int clrMax;
    fscanf(pgmfile, "P2 %d %d %d", &img.w, &img.h, &clrMax);
    Float clrScl = 1. / clrMax;
    clrScl = std::pow(clrScl, power);
    img.pixels.resize(img.w * img.h);
    for (int y = 0; y < img.h; y++) {
        for (int x = 0; x < img.w; x++) {
            int clr;
            fscanf(pgmfile, " %d", &clr);
            if (feof(pgmfile)) {
                fprintf(stderr,
                        "Sorry, failed to read image information\n");
                exit(2);
            }
            clr = clrMax - clr;
            clr = (int)std::pow(clr, power);
            img[(img.h - 1 - y) * img.w + x] = clrScl * clr;
        }
    }
    return img;
}



#endif

