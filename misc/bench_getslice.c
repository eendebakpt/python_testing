#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <time.h>

#define ITERATIONS 1000000
#define SIZE 2000

static double
bench_getslice(PyObject *seq, Py_ssize_t i1, Py_ssize_t i2, int iterations)
{
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        PyObject *slice = PySequence_GetSlice(seq, i1, i2);
        if (slice == NULL) {
            PyErr_Print();
            return -1.0;
        }
        Py_DECREF(slice);
    }
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

int
main(int argc, char *argv[])
{
    Py_Initialize();

    /* Benchmark with a list */
    PyObject *list = PyList_New(SIZE);
    if (list == NULL) {
        PyErr_Print();
        return 1;
    }
    for (Py_ssize_t i = 0; i < SIZE; i++) {
        PyObject *val = PyLong_FromSsize_t(i);
        if (val == NULL) {
            PyErr_Print();
            return 1;
        }
        PyList_SET_ITEM(list, i, val);  /* steals reference */
    }

    printf("PySequence_GetSlice benchmark (%d iterations)\n", ITERATIONS);
    printf("==============================================\n\n");

    /* Small slice */
    double t = bench_getslice(list, 0, 5, ITERATIONS);
    printf("list[0:5]      : %.4f s  (%.1f ns/call)\n",
           t, t / ITERATIONS * 1e9);

    /* Medium slice */
    t = bench_getslice(list, 1500, 1800, ITERATIONS);
    printf("list[1500:1800]: %.4f s  (%.1f ns/call)\n",
           t, t / ITERATIONS * 1e9);

    /* Full slice */
    t = bench_getslice(list, 0, 2000, ITERATIONS);
    printf("list[0:2000]   : %.4f s  (%.1f ns/call)\n",
           t, t / ITERATIONS * 1e9);

    /* Empty slice */
    t = bench_getslice(list, 1000, 1000, ITERATIONS);
    printf("list[1000:1000]: %.4f s  (%.1f ns/call)\n",
           t, t / ITERATIONS * 1e9);

    Py_DECREF(list);

    /* Benchmark with a tuple */
    PyObject *tuple = PyTuple_New(SIZE);
    if (tuple == NULL) {
        PyErr_Print();
        return 1;
    }
    for (Py_ssize_t i = 0; i < SIZE; i++) {
        PyObject *val = PyLong_FromSsize_t(i);
        if (val == NULL) {
            PyErr_Print();
            return 1;
        }
        PyTuple_SET_ITEM(tuple, i, val);  /* steals reference */
    }

    printf("\n");

    t = bench_getslice(tuple, 0, 5, ITERATIONS);
    printf("tuple[0:5]     : %.4f s  (%.1f ns/call)\n",
           t, t / ITERATIONS * 1e9);

    t = bench_getslice(tuple, 1500, 1800, ITERATIONS);
    printf("tuple[1500:1800]: %.4f s  (%.1f ns/call)\n",
           t, t / ITERATIONS * 1e9);

    t = bench_getslice(tuple, 0, 2000, ITERATIONS);
    printf("tuple[0:2000]  : %.4f s  (%.1f ns/call)\n",
           t, t / ITERATIONS * 1e9);

    Py_DECREF(tuple);

    /* Benchmark with a bytes object */
    PyObject *bytes = PyBytes_FromStringAndSize(NULL, SIZE);
    if (bytes == NULL) {
        PyErr_Print();
        return 1;
    }
    char *buf = PyBytes_AS_STRING(bytes);
    for (int i = 0; i < SIZE; i++) {
        buf[i] = (char)i;
    }

    printf("\n");

    t = bench_getslice(bytes, 0, 5, ITERATIONS);
    printf("bytes[0:5]     : %.4f s  (%.1f ns/call)\n",
           t, t / ITERATIONS * 1e9);

    t = bench_getslice(bytes, 1500, 1800, ITERATIONS);
    printf("bytes[1500:1800]: %.4f s  (%.1f ns/call)\n",
           t, t / ITERATIONS * 1e9);

    t = bench_getslice(bytes, 0, 2000, ITERATIONS);
    printf("bytes[0:2000]  : %.4f s  (%.1f ns/call)\n",
           t, t / ITERATIONS * 1e9);

    Py_DECREF(bytes);

    if (Py_FinalizeEx() < 0) {
        return 120;
    }
    return 0;
}

/*

# From the CPython repo root
cl /I Include /I Include/internal /I PC bench_getslice.c /link /LIBPATH:PCbuild/amd64 python315.lib
# Then run with python315.dll on PATH:
set PATH=PCbuild\amd64;%PATH%
bench_getslice.exe

cl /I Include /I Include/internal /I PC /Zi bench_getslice.c /link /LIBPATH:PCbuild/amd64 python315_d.lib
set PATH=PCbuild\amd64;%PATH%
bench_getslice.exe

Search the Start Menu for "Developer Command Prompt for VS 2022

PCbuild\find_msbuild.bat

# For 64-bit (most common)
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64

set PATH=PCbuild\amd64;%PATH%


*/


/* PR

PySequence_GetSlice benchmark (1000000 iterations)
==============================================

list[0:5]    : 0.0520 s  (52.0 ns/call)
list[10:50]  : 0.0940 s  (94.0 ns/call)
list[0:100]  : 0.1530 s  (153.0 ns/call)
list[50:50]  : 0.0370 s  (37.0 ns/call)

tuple[0:5]   : 0.0450 s  (45.0 ns/call)
tuple[10:50] : 0.1180 s  (118.0 ns/call)
tuple[0:100] : 0.0560 s  (56.0 ns/call)

bytes[0:5]   : 0.0520 s  (52.0 ns/call)
bytes[10:50] : 0.0450 s  (45.0 ns/call)
bytes[0:100] : 0.0320 s  (32.0 ns/call)


PySequence_GetSlice benchmark (1000000 iterations)
==============================================

list[0:5]    : 0.0490 s  (49.0 ns/call)
list[10:50]  : 0.0950 s  (95.0 ns/call)
list[0:100]  : 0.1550 s  (155.0 ns/call)
list[50:50]  : 0.0350 s  (35.0 ns/call)

tuple[0:5]   : 0.0440 s  (44.0 ns/call)
tuple[10:50] : 0.0950 s  (95.0 ns/call)
tuple[0:100] : 0.0260 s  (26.0 ns/call)

bytes[0:5]   : 0.0390 s  (39.0 ns/call)
bytes[10:50] : 0.0410 s  (41.0 ns/call)
bytes[0:100] : 0.0280 s  (28.0 ns/call)

*/