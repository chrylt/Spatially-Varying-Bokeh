/*
 * Generate samples by recursive subdivision.
 * 2022-12-01: Adapted from 2048 by Abdalla Ahmed
 */

#include "common.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "getopt/getopt.h"
#include <string.h>
#include <ctime>
#include <vector>

inline Float frac(Float x) { return x - std::floor(x); }

struct Rect {                                                                   // Rectangle
    Float xl, xr, yb, yt;                                                       // left, right, bottom, top
    Float width() { return xr - xl; };
    Float height() { return yt - yb; };
};

class C2048 {
private:
    int w, h;                                                                   // Dimensions of accumulated mass buffer
    std::vector<Float> accumMass;                                               // Mass of interval between origin and a given integer point.
    inline Float m(int x, int y) { return accumMass[y * w + x]; }               // Shorthand
    inline Float mass(Float x, Float y);                                        // Mass of of interval between origin and a given point.
    inline Float mass(Float xl, Float xr, Float yb, Float yt) {                 // mass of delimited rectangle
        return mass(xr, yt) - mass(xl, yt) - mass(xr, yb) + mass(xl, yb);
    }
    inline Float mass(Rect r) { return mass(r.xl, r.xr, r.yb, r.yt); };
    Float hSect(Float xl, Float xr, Float yb, Float yt, Float ratio = 0.5);     // Find vertical median line
    Float vSect(Float xl, Float xr, Float yb, Float yt, Float ratio = 0.5);     // Find horizontal median line
    Point centroid(Float xl, Float xr, Float yb, Float yt);                     // Return center of mass of given rectangle
    Point centroid(Rect r) { return centroid(r.xl, r.xr, r.yb, r.yt); } ;
public:
    C2048(const Image &img);
    Points slice(int N);
};

// -----------------------------------------------------------------------------
// Constructor: initilaize accumulated mass buffer
// -----------------------------------------------------------------------------

C2048::C2048(const Image &img) : w(img.w + 1), h(img.h + 1) {
    accumMass.resize(h * w);
    for (int x = 0; x < w; x++) accumMass[x] = 0;                               // Accum mass is essentially zero on the bottom edge
    for (int y = 1; y < h; y++) {
        accumMass[y * w] = 0;                                                   // Accum mass is zero on the left edge
        for (int x = 1; x < w; x++) {
            accumMass[y * w + x] = (
                        img.pixels[(y - 1) * img.w + (x - 1)]                          // Pixel at origin of pixel area
                + accumMass[(y - 1) *     w + (x    )]                          // Accumulated mass one-row below
                + accumMass[(y    ) *     w + (x - 1)]                          // Accumulated mass one-column to the left
                - accumMass[(y - 1) *     w + (x - 1)]                          // Accumulated mass at origin, subtracted because it's added twice
            );
        }
    }
}

// -----------------------------------------------------------------------------
// Compute mass at a floating point
// -----------------------------------------------------------------------------

inline Float C2048::mass(Float x, Float y) {                                    // mass of area to the left and below this point
    return (
        m(x, y)                                                                 // Mass of rectangle up to origin of containing pixel of the point
        + (m(x, ceil(y)) - m(x, y)) * frac(y)                                   // Mass of horizontal stripe above
        + (m(ceil(x), y) - m(x, y)) * frac(x)                                   // Mass of vertical stripe to the right
        + (m(ceil(x), ceil(y)) - m(x, ceil(y)) - m(ceil(x), y) + m(x, y))       // Mass of the containing pixel
            * frac(y) * frac(x)                                                 // Area of pixel below and to the left of the point
    );
}

// -----------------------------------------------------------------------------
// Section rectangle horizontally by a given ratio and return position
// of section line:
// -----------------------------------------------------------------------------

Float C2048::hSect(Float xl, Float xr, Float yb, Float yt, Float ratio) {
    Float trgt = mass(xl, xr, yb, yt) * ratio;                                  // Target mass
    Float x, mLeft, lowerBound(xl), upperBound(xr);
    while (upperBound - lowerBound > 1) {                                       // Binary search until at most one pixel boundary between ends
        x = 0.5 * (upperBound + lowerBound);                                    // Cut at midpoint
        mLeft = mass(xl, x, yb, yt);                                            // Mass of left half
        if (mLeft < trgt) lowerBound = x;                                       // If too small, section is in [x, upperBound]
        else           upperBound = x;                                          // Otherwise it is in [lowerBound, x]
    }
    lowerBound = floor(                                                         // We seek containing pixel origin
        mass(xl, floor(upperBound), yb, yt) < trgt ? upperBound : lowerBound    // Either the one containing upperBound or lowerBound
    );
    mLeft = mass(xl, lowerBound, yb, yt);                                       // Compute mass up to that edge, which is <= target mass
    x = lowerBound + (trgt - mLeft) / mass(lowerBound, lowerBound+1, yb, yt);   // Add fraction needed to reach target
    return x;
}

// -----------------------------------------------------------------------------
// Section given area vertically by given ratio and return position
// of section line:
// -----------------------------------------------------------------------------

Float C2048::vSect (Float xl, Float xr, Float yb, Float yt, Float ratio) {      // Analogous to hSect
    Float trgt = mass(xl, xr, yb, yt) * ratio;
    Float y, mBelow, lowerBound(yb), upperBound(yt);
    while (upperBound - lowerBound > 1) {
        y = 0.5 * (upperBound + lowerBound);
        mBelow = mass(xl, xr, yb, y);
        if (mBelow < trgt) lowerBound = y;
        else            upperBound = y;
    }
    lowerBound = floor(
        mass(xl, xr, yb, floor(upperBound)) < trgt ? upperBound : lowerBound
    );
    mBelow = mass(xl, xr, yb, lowerBound);
    y = lowerBound + (trgt - mBelow) / mass(xl, xr, lowerBound, lowerBound+1);
    return y;
}


// -----------------------------------------------------------------------------
// Compute centroid of a rectangle
// -----------------------------------------------------------------------------

Point C2048::centroid(Float xl, Float xr, Float yb, Float yt) {
    Float sumWeight, sum;
    Point c;
    sumWeight = mass(xl, xr, yb, yt);
    if (int(xl) == int(xr)) {                                                   // All within one pixel?
        c.x = 0.5 * (xl + xr);                                                  // Then centroid is just at midpoint
    }
    else {
        sum = (
            mass(xl, ceil(xl), yb, yt) * (0.5 * (xl + ceil(xl))) +              // Stripe on the left
            mass(floor(xr), xr, yb, yt) * (0.5 * (xr + floor(xr)))              // Stripe on the right
        );
        for (int x = ceil(xl); x < floor(xr); x++) {                            // Iterate through pixels in between
            sum += mass(x, x + 1, yb, yt) * (x + 0.5);                          // Contribution of pixel column
        }
        c.x = sum / sumWeight;
    }
    if (int(yb) == int(yt)) {                                                   // All within one pixel?
        c.y = 0.5 * (yb + yt);                                                  // Then centroid is just at midpoint
    }
    else {
        sum = (
            mass(xl, xr, yb, ceil(yb)) * (0.5 * (yb + ceil(yb))) +              // Stripe at the bottom
            mass(xl, xr, floor(yt), yt) * (0.5 * (yt + floor(yt)))              // Stripe on top
        );
        for (int y = ceil(yb); y < floor(yt); y++) {                            // Iterate through pixels in between
            sum += mass(xl, xr, y, y + 1) * (y + 0.5);                          // Contribution of pixel row
        }
        c.y = sum / sumWeight;
    }
    return c;
}

// -----------------------------------------------------------------------------
// Slice recursively into equal-mass rectangles and place a samples
// in the centroid of each rectangle
// -----------------------------------------------------------------------------

Points C2048::slice(int N) {
    Points p;                                                                   // To store list of points
    if (N <= 0) return p;                                                       // Safety measure
    struct Record {                                                             // Used for stack-based recursion
        Rect rect;                                                              // Region
        int n;                                                                  // Number of points
    };
    std::vector<Record> stack;                                                  // LIFO, hence depth first
    Rect rect = {0, w - 1, 0, h - 1};                                           // Initial region is whole image
    stack.push_back({rect, N});                                                 // And whole number of points
    while (!stack.empty()) {
        Rect rect = stack.back().rect;                                          // Retrieve topmost region
        int n = stack.back().n;                                                 // And number of points
        stack.pop_back();                                                       // Remove from stack
        if (n == 1) {                                                           // Only one point?
            p.push_back(centroid(rect));                                        // Place sample at centroid, and we are done
        }
        else {                                                                  // More than one point
            int n1(n/2), n2(n - n1);                                            // Split roughly at halves
            Float ratio = Float(n1) / n;                                        // Splitting ratio, which is between 1/3 and 1/2
            if (rect.width() > rect.height()) {                                 // If landscape, slice horizontally
                Float x = hSect(rect.xl, rect.xr, rect.yb, rect.yt, ratio);     // Section point
                stack.push_back({{rect.xl, x, rect.yb, rect.yt}, n1});          // Left half
                stack.push_back({{x, rect.xr, rect.yb, rect.yt}, n2});          // Right half
            }
            else {                                                              // Otherwise slice vertically
                Float y = vSect(rect.xl, rect.xr, rect.yb, rect.yt, ratio);     // Section point
                stack.push_back({{rect.xl, rect.xr, rect.yb, y}, n1});          // Bottom half
                stack.push_back({{rect.xl, rect.xr, y, rect.yt}, n2});          // Top half
            }
        }
    }
    return p;
}



