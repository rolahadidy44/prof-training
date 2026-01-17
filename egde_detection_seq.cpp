#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

const int WIDTH = 8;
const int HEIGHT = 8;

// Sobel kernels
int Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

int Gy[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

void print_image(const vector<vector<int>>& img) {
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            cout << img[i][j] << "\t";
        }
        cout << endl;
    }
}

int main() {
    vector<vector<int>> image(HEIGHT, vector<int>(WIDTH));
    vector<vector<int>> output(HEIGHT, vector<int>(WIDTH, 0));

    // Visible INPUT
    for (int i = 0; i < HEIGHT; i++)
        for (int j = 0; j < WIDTH; j++)
            image[i][j] = (i + j) * 10;

    cout << "INPUT IMAGE:\n";
    print_image(image);

    double start = omp_get_wtime();

    // SEQUENTIAL Sobel
    for (int i = 1; i < HEIGHT - 1; i++) {
        for (int j = 1; j < WIDTH - 1; j++) {
            int sx = 0, sy = 0;

            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sx += image[i + ki][j + kj] * Gx[ki + 1][kj + 1];
                    sy += image[i + ki][j + kj] * Gy[ki + 1][kj + 1];
                }
            }

            output[i][j] = sqrt(sx * sx + sy * sy);
        }
    }

    double end = omp_get_wtime();

    cout << "\nOUTPUT IMAGE (EDGES):\n";
    print_image(output);

    cout << "\nSequential Execution Time: "
         << end - start << " seconds\n";

    return 0;
}
