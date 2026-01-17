#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

// ----------------------------
// Sobel Edge Detection (Sequential vs OpenMP)
// - Generates a synthetic grayscale image
// - Runs sequential Sobel
// - Runs OpenMP-parallel Sobel
// - Verifies outputs match
// - Prints timings + (optionally) the matrices for small sizes
// ----------------------------

static int Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1},
};

static int Gy[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1},
};

static void print_image(const std::vector<std::vector<int>>& img) {
    const int h = (int)img.size();
    const int w = h ? (int)img[0].size() : 0;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            std::cout << img[i][j] << "\t";
        }
        std::cout << "\n";
    }
}

static std::vector<std::vector<int>> make_input(int height, int width) {
    std::vector<std::vector<int>> image(height, std::vector<int>(width));

    // Synthetic input (fast, deterministic)
    // A simple gradient-like pattern with some periodic variation.
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int base = (i + j) * 10;
            int wiggle = (int)(20 * std::sin(i * 0.05) + 20 * std::cos(j * 0.05));
            image[i][j] = base + wiggle;
        }
    }
    return image;
}

static void sobel_sequential(
    const std::vector<std::vector<int>>& image,
    std::vector<std::vector<int>>& output
) {
    const int height = (int)image.size();
    const int width  = height ? (int)image[0].size() : 0;

    // borders remain 0
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int sx = 0, sy = 0;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sx += image[i + ki][j + kj] * Gx[ki + 1][kj + 1];
                    sy += image[i + ki][j + kj] * Gy[ki + 1][kj + 1];
                }
            }
            output[i][j] = (int)std::sqrt((double)sx * sx + (double)sy * sy);
        }
    }
}

static void sobel_openmp(
    const std::vector<std::vector<int>>& image,
    std::vector<std::vector<int>>& output
) {
    const int height = (int)image.size();
    const int width  = height ? (int)image[0].size() : 0;

    // Parallelize the independent per-pixel computation.
    // collapse(2) flattens the 2 loops to spread work better.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int sx = 0, sy = 0;
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sx += image[i + ki][j + kj] * Gx[ki + 1][kj + 1];
                    sy += image[i + ki][j + kj] * Gy[ki + 1][kj + 1];
                }
            }
            output[i][j] = (int)std::sqrt((double)sx * sx + (double)sy * sy);
        }
    }
}

static bool outputs_equal(
    const std::vector<std::vector<int>>& a,
    const std::vector<std::vector<int>>& b,
    int& first_i,
    int& first_j
) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].size() != b[i].size()) return false;
        for (size_t j = 0; j < a[i].size(); j++) {
            if (a[i][j] != b[i][j]) {
                first_i = (int)i;
                first_j = (int)j;
                return false;
            }
        }
    }
    return true;
}

static int parse_int_arg(const char* s, const char* name) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') {
        throw std::runtime_error(std::string("Invalid ") + name + ": " + s);
    }
    if (v < 3 || v > 20000) {
        throw std::runtime_error(std::string(name) + " must be between 3 and 20000");
    }
    return (int)v;
}


int main(int argc, char** argv) {
    // Defaults chosen so speedup is visible.
    // You can pass custom size: sobel_compare.exe <height> <width>
    int height = 2048;
    int width  = 2048;

    if (argc >= 2) height = parse_int_arg(argv[1], "height");
    if (argc >= 3) width  = parse_int_arg(argv[2], "width");

    // FORCE EXACTLY 2 THREADS
    omp_set_num_threads(2);

    const bool should_print = (height <= 16 && width <= 16);

    std::vector<std::vector<int>> image(height, std::vector<int>(width));

  
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int base = (i + j) * 10;
            int wiggle = (int)(20 * sin(i * 0.05) + 20 * cos(j * 0.05));
            image[i][j] = base + wiggle;
        }
    }

    std::vector<std::vector<int>> out_seq(height, std::vector<int>(width, 0));
    std::vector<std::vector<int>> out_omp(height, std::vector<int>(width, 0));

    if (should_print) {
        std::cout << "INPUT IMAGE (" << height << "x" << width << "):\n";
        print_image(image);
        std::cout << "\n";
    } else {
        std::cout << "INPUT IMAGE size: " << height << "x" << width << "\n";
    }


    double t0 = omp_get_wtime();
    sobel_sequential(image, out_seq);
    double t1 = omp_get_wtime();


    double t2 = omp_get_wtime();
    sobel_openmp(image, out_omp);
    double t3 = omp_get_wtime();


    int bi = -1, bj = -1;
    bool ok = outputs_equal(out_seq, out_omp, bi, bj);

    if (should_print) {
        std::cout << "OUTPUT (Sequential):\n";
        print_image(out_seq);
        std::cout << "\nOUTPUT (OpenMP):\n";
        print_image(out_omp);
        std::cout << "\n";
    }

    std::cout << std::fixed << std::setprecision(6);
    const double seq_s = (t1 - t0);
    const double omp_s = (t3 - t2);

    std::cout << "Sequential time: " << seq_s << " s\n";
    std::cout << "OpenMP time:     " << omp_s << " s\n";

    if (omp_s > 0) {
        std::cout << "Speedup (seq/omp): " << (seq_s / omp_s) << "x\n";
    }

    if (!ok) {
        std::cerr << "ERROR: outputs do NOT match. First mismatch at ("
                  << bi << ", " << bj << ")\n";
        return 1;
    }

    std::cout << "Correctness check: PASS (outputs match)\n";

    return 0;
}

