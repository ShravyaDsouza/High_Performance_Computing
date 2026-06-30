#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iomanip>
using namespace std;

void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1, n2 = right - mid;
    vector<int> L(n1), R(n2);
    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void mergeSortRecursiveSerial(vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    mergeSortRecursiveSerial(arr, left, mid);
    mergeSortRecursiveSerial(arr, mid + 1, right);
    merge(arr, left, mid, right);
}

void mergeSortRecursiveParallel(vector<int>& arr, int left, int right) {
    if (left >= right) return;
    int mid = left + (right - left) / 2;
    #pragma omp task shared(arr)
    mergeSortRecursiveParallel(arr, left, mid);
    #pragma omp task shared(arr)
    mergeSortRecursiveParallel(arr, mid + 1, right);
    #pragma omp taskwait
    merge(arr, left, mid, right);
}

void mergeSortIterativeSerial(vector<int>& arr) {
    int n = arr.size();
    int curr_size = 1;
    while (curr_size < n) {
        for (int left_start = 0; left_start < n - 1; left_start += 2 * curr_size) {
            int mid = min(left_start + curr_size - 1, n - 1);
            int right_end = min(left_start + 2 * curr_size - 1, n - 1);
            merge(arr, left_start, mid, right_end);
        }
        curr_size++;
    }
}

void mergeSortIterativeParallel(vector<int>& arr) {
    int n = arr.size();
    int curr_size = 1;
    while (curr_size < n) {
        #pragma omp parallel for
        for (int left_start = 0; left_start < n - 1; left_start += 2 * curr_size) {
            int mid = min(left_start + curr_size - 1, n - 1);
            int right_end = min(left_start + 2 * curr_size - 1, n - 1);
            merge(arr, left_start, mid, right_end);
        }
        curr_size++;
    }
}

vector<int> generateRandomArray(int size) {
    vector<int> arr(size);
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100000;
    }
    return arr;
}

int main() {
    srand(time(0));
    vector<int> sizes = {500, 1000, 10000, 30000, 50000, 70000, 100000};
    int processors = 12;
    omp_set_num_threads(processors); // Set the number of threads for OpenMP

    cout << "Using " << processors << " processor cores\n";
    cout << left << setw(10) << "DataPts"
         << setw(20) << "Serial Rec Time (s)"
         << setw(20) << "Parallel Rec Time (s)"
         << setw(15) << "Speedup"
         << setw(15) << "Efficiency"
         << setw(15) << "Total Cost\n";
    cout << string(95, '-') << endl;

    for (int size : sizes) {
        vector<int> base = generateRandomArray(size);
        vector<int> A = base;
        vector<int> B = base;

        double t1 = omp_get_wtime();
        mergeSortRecursiveSerial(A, 0, size - 1);
        double t2 = omp_get_wtime();

        double t3 = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            mergeSortRecursiveParallel(B, 0, size - 1);
        }
        double t4 = omp_get_wtime();

        // Calculations
        double Ts = t2 - t1;
        double Tp = t4 - t3;
        double speedup = Ts / Tp;
        double efficiency = speedup / processors;
        int totalCost = size * processors;

        cout << left << setw(10) << size
             << setw(20) << fixed << setprecision(6) << Ts
             << setw(20) << fixed << setprecision(6) << Tp
             << setw(15) << fixed << setprecision(6) << speedup
             << setw(15) << fixed << setprecision(6) << efficiency
             << setw(15) << totalCost << endl;
    }
    return 0;
}

