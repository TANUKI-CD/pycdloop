#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * Compiled kernels of pycdloop: the Walsh--Hadamard transform and the change
 * of coordinates between the loop basis and the split form.
 *
 * The unnormalized transform H of length N = 2^n is applied in place by
 * butterflies (u, v) -> (u + v, u - v) in stages of increasing length, as in
 * the reference implementation in _transform.py. A coefficient may occupy
 * several adjacent storage words, called lanes: two doubles for a complex
 * coefficient, and deg residues for an element of an extension field. A
 * stage of length L then combines two adjacent blocks of L * lanes words,
 * word by word, so one routine serves every coefficient type.
 *
 * Two changes to the order of the loops make the transform faster without
 * changing a single arithmetic operation, so results are bit-identical to
 * the stage-by-stage evaluation. With one lane, the first three stages are
 * fused over tiles of eight words held in registers. After that, stages are
 * taken two at a time over four blocks, which halves the passes over the
 * record.
 *
 * The modular transform works on signed 64-bit integers that hold residues
 * of a prime below 2^31, or sums of two of them. A stage at most doubles the
 * magnitude of an entry, so entries are reduced after every WHT_UNREDUCED
 * stages and once at the end, which keeps every intermediate value inside
 * the 64-bit range.
 *
 * A loop-basis record holds the coefficients a of the basis elements [e_t]
 * followed by the coefficients b of [-e_t]. Its split form is the pair
 * (H(a + b), a - b), and the inverse map sends (s, g) to
 * a = (H s / N + g) / 2 and b = (H s / N - g) / 2. The routines below make
 * one pass over each record, which stays in the first-level cache, where
 * the same steps written as whole-array operations pass over a batch about
 * a dozen times.
 */

#define WHT_UNREDUCED 30
#define WHT_RELEASE_WORDS ((Py_ssize_t)1 << 15)

// Apply the first three stages to one tile of eight words
static inline void tile_f64(double* restrict a) {
    double a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    double a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
    double b0 = a0 + a1, b1 = a0 - a1, b2 = a2 + a3, b3 = a2 - a3;
    double b4 = a4 + a5, b5 = a4 - a5, b6 = a6 + a7, b7 = a6 - a7;
    double c0 = b0 + b2, c1 = b1 + b3, c2 = b0 - b2, c3 = b1 - b3;
    double c4 = b4 + b6, c5 = b5 + b7, c6 = b4 - b6, c7 = b5 - b7;

    a[0] = c0 + c4;
    a[1] = c1 + c5;
    a[2] = c2 + c6;
    a[3] = c3 + c7;
    a[4] = c0 - c4;
    a[5] = c1 - c5;
    a[6] = c2 - c6;
    a[7] = c3 - c7;
}

// Apply two consecutive stages to four adjacent blocks of doubles
static inline void quad_f64(double* restrict a, size_t width) {
    double* restrict p0 = a;
    double* restrict p1 = p0 + width;
    double* restrict p2 = p1 + width;
    double* restrict p3 = p2 + width;

    for (size_t j = 0; j < width; j++) {
        double x0 = p0[j], x1 = p1[j], x2 = p2[j], x3 = p3[j];
        double s0 = x0 + x1, d0 = x0 - x1, s1 = x2 + x3, d1 = x2 - x3;

        p0[j] = s0 + s1;
        p1[j] = d0 + d1;
        p2[j] = s0 - s1;
        p3[j] = d0 - d1;
    }
}

// Apply one stage to two adjacent blocks of doubles
static inline void pair_f64(double* restrict a, size_t width) {
    double* restrict p0 = a;
    double* restrict p1 = p0 + width;

    for (size_t j = 0; j < width; j++) {
        double x0 = p0[j], x1 = p1[j];

        p0[j] = x0 + x1;
        p1[j] = x0 - x1;
    }
}

// Transform one record of length words, whose coefficients span lanes words
static void record_f64(double* restrict a, size_t length, size_t lanes) {
    size_t width = lanes;

    if (lanes == 1 && length >= 8) {
        for (size_t base = 0; base < length; base += 8) tile_f64(a + base);
        width = 8;
    }
    for (; 4 * width <= length; width *= 4) {
        for (size_t base = 0; base < length; base += 4 * width) {
            quad_f64(a + base, width);
        }
    }
    for (; width < length; width *= 2) {
        for (size_t base = 0; base < length; base += 2 * width) {
            pair_f64(a + base, width);
        }
    }
}

// Apply the first three stages to one tile of eight integers
static inline void tile_i64(int64_t* restrict a) {
    int64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    int64_t a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
    int64_t b0 = a0 + a1, b1 = a0 - a1, b2 = a2 + a3, b3 = a2 - a3;
    int64_t b4 = a4 + a5, b5 = a4 - a5, b6 = a6 + a7, b7 = a6 - a7;
    int64_t c0 = b0 + b2, c1 = b1 + b3, c2 = b0 - b2, c3 = b1 - b3;
    int64_t c4 = b4 + b6, c5 = b5 + b7, c6 = b4 - b6, c7 = b5 - b7;

    a[0] = c0 + c4;
    a[1] = c1 + c5;
    a[2] = c2 + c6;
    a[3] = c3 + c7;
    a[4] = c0 - c4;
    a[5] = c1 - c5;
    a[6] = c2 - c6;
    a[7] = c3 - c7;
}

// Apply two consecutive stages to four adjacent blocks of integers
static inline void quad_i64(int64_t* restrict a, size_t width) {
    int64_t* restrict p0 = a;
    int64_t* restrict p1 = p0 + width;
    int64_t* restrict p2 = p1 + width;
    int64_t* restrict p3 = p2 + width;

    for (size_t j = 0; j < width; j++) {
        int64_t x0 = p0[j], x1 = p1[j], x2 = p2[j], x3 = p3[j];
        int64_t s0 = x0 + x1, d0 = x0 - x1, s1 = x2 + x3, d1 = x2 - x3;

        p0[j] = s0 + s1;
        p1[j] = d0 + d1;
        p2[j] = s0 - s1;
        p3[j] = d0 - d1;
    }
}

// Apply one stage to two adjacent blocks of integers
static inline void pair_i64(int64_t* restrict a, size_t width) {
    int64_t* restrict p0 = a;
    int64_t* restrict p1 = p0 + width;

    for (size_t j = 0; j < width; j++) {
        int64_t x0 = p0[j], x1 = p1[j];

        p0[j] = x0 + x1;
        p1[j] = x0 - x1;
    }
}

// Reduce every word of a record to its canonical residue
static void reduce_i64(int64_t* restrict a, size_t length, int64_t modulus) {
    for (size_t i = 0; i < length; i++) {
        int64_t residue = a[i] % modulus;

        a[i] = residue < 0 ? residue + modulus : residue;
    }
}

// Transform one record of residues, reducing after every interval stages
static void record_i64(int64_t* restrict a, size_t length, size_t lanes,
        int64_t modulus, size_t interval) {
    size_t width = lanes;
    size_t pending = 0;

    /* Stages are fused only while all of them fit before the next reduction,
     * so no entry grows by more than 2^interval between reductions. */
    if (lanes == 1 && length >= 8 && interval >= 3) {
        for (size_t base = 0; base < length; base += 8) tile_i64(a + base);
        width = 8;
        pending = 3;
    }
    if (pending >= interval) {
        reduce_i64(a, length, modulus);
        pending = 0;
    }
    while (width < length) {
        if (4 * width <= length && pending + 2 <= interval) {
            for (size_t base = 0; base < length; base += 4 * width) {
                quad_i64(a + base, width);
            }
            width *= 4;
            pending += 2;
        } else {
            for (size_t base = 0; base < length; base += 2 * width) {
                pair_i64(a + base, width);
            }
            width *= 2;
            pending += 1;
        }
        if (pending >= interval) {
            reduce_i64(a, length, modulus);
            pending = 0;
        }
    }
    reduce_i64(a, length, modulus);
}

// Compute the split form of one loop-basis record of doubles
static void split_f64(const double* restrict loop, double* restrict spectral,
        double* restrict signed_part, size_t length, size_t lanes) {
    const double* restrict plus = loop;
    const double* restrict minus = loop + length;

    for (size_t i = 0; i < length; i++) {
        double a = plus[i], b = minus[i];

        spectral[i] = a + b;
        signed_part[i] = a - b;
    }
    record_f64(spectral, length, lanes);
}

// Compute one loop-basis record of doubles from its split form
static void unsplit_f64(const double* restrict spectral,
        const double* restrict signed_part, double* restrict loop, size_t length,
        size_t lanes, double dimension) {
    double* restrict plus = loop;
    double* restrict minus = loop + length;

    // The first row of the output serves as the workspace of the transform
    memcpy(plus, spectral, length * sizeof(double));
    record_f64(plus, length, lanes);
    for (size_t i = 0; i < length; i++) {
        double group = plus[i] / dimension;
        double g = signed_part[i];

        plus[i] = (group + g) * 0.5;
        minus[i] = (group - g) * 0.5;
    }
}

// Compute the split form of one loop-basis record of residues
static void split_u32(const uint32_t* restrict loop, uint32_t* restrict spectral,
        uint32_t* restrict signed_part, int64_t* restrict work, size_t length,
        size_t lanes, int64_t modulus) {
    const uint32_t* restrict plus = loop;
    const uint32_t* restrict minus = loop + length;

    for (size_t i = 0; i < length; i++) {
        int64_t a = plus[i], b = minus[i];
        int64_t difference = a - b;

        work[i] = a + b;
        signed_part[i] = (uint32_t)(difference < 0 ? difference + modulus : difference);
    }
    /* The sums are below 2p < 2^32, so 30 unreduced stages stay below 2^62. */
    record_i64(work, length, lanes, modulus, WHT_UNREDUCED);
    for (size_t i = 0; i < length; i++) spectral[i] = (uint32_t)work[i];
}

// Compute one loop-basis record of residues from its split form
static void unsplit_u32(const uint32_t* restrict spectral,
        const uint32_t* restrict signed_part, uint32_t* restrict loop,
        int64_t* restrict work, size_t length, size_t lanes, int64_t modulus,
        uint64_t inverse_dimension) {
    uint32_t* restrict plus = loop;
    uint32_t* restrict minus = loop + length;
    uint64_t p = (uint64_t)modulus;
    uint64_t half = (p + 1) / 2;

    for (size_t i = 0; i < length; i++) work[i] = spectral[i];
    record_i64(work, length, lanes, modulus, WHT_UNREDUCED);
    for (size_t i = 0; i < length; i++) {
        // Every factor is below 2^31, so each product is below 2^62
        uint64_t group = (uint64_t)work[i] * inverse_dimension % p;
        uint64_t g = signed_part[i];
        uint64_t sum = group + g;
        uint64_t difference = group + p - g;

        if (sum >= p) sum -= p;
        if (difference >= p) difference -= p;
        plus[i] = (uint32_t)(sum * half % p);
        minus[i] = (uint32_t)(difference * half % p);
    }
}

// Return whether a positive value is a power of two
static int is_power_of_two(Py_ssize_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

enum { WORD_F64, WORD_I64, WORD_U32 };

/*
 * Acquire a C-contiguous buffer and check its element type, so that the
 * kernels never reinterpret storage of another type. A buffer of float64
 * words may also hold complex128 elements, which are pairs of them. Returns
 * the size of one word, or 0 with an exception set.
 */
// Acquire a buffer of the given word type
static Py_ssize_t acquire(PyObject* object, Py_buffer* view, int writable, int kind) {
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | (writable ? PyBUF_WRITABLE : 0);
    const char* format;
    Py_ssize_t word = 0;

    if (PyObject_GetBuffer(object, view, flags) != 0) return 0;
    format = view->format == NULL ? "B" : view->format;
    if (*format == '@' || *format == '=') format++;
    if (kind == WORD_F64) {
        if ((strcmp(format, "d") == 0 && view->itemsize == 8)
                || (strcmp(format, "Zd") == 0 && view->itemsize == 16)) {
            word = 8;
        }
    } else if (kind == WORD_I64) {
        if ((strcmp(format, "l") == 0 || strcmp(format, "q") == 0)
                && view->itemsize == 8) {
            word = 8;
        }
    } else if ((strcmp(format, "I") == 0 || strcmp(format, "L") == 0)
            && view->itemsize == 4) {
        word = 4;
    }
    if (word == 0) {
        PyBuffer_Release(view);
        PyErr_SetString(PyExc_TypeError,
                "buffer does not hold native words of the required type");
        return 0;
    }
    if (((uintptr_t)view->buf % (size_t)word) != 0) {
        PyBuffer_Release(view);
        PyErr_SetString(PyExc_ValueError, "buffer is not aligned");
        return 0;
    }
    return word;
}

// Check a buffer of whole records and return their number, or -1 with an error set
static Py_ssize_t record_count(const Py_buffer* view, Py_ssize_t dimension,
        Py_ssize_t lanes, Py_ssize_t word) {
    Py_ssize_t record;

    if (!is_power_of_two(dimension)) {
        PyErr_SetString(PyExc_ValueError, "dimension must be a power of two");
        return -1;
    }
    if (lanes < 1) {
        PyErr_SetString(PyExc_ValueError, "lanes must be positive");
        return -1;
    }
    if (dimension > PY_SSIZE_T_MAX / lanes / word) {
        PyErr_SetString(PyExc_OverflowError, "record is too large");
        return -1;
    }
    record = dimension * lanes * word;
    if (view->len % record != 0) {
        PyErr_SetString(PyExc_ValueError, "buffer must hold whole records");
        return -1;
    }
    return view->len / record;
}

// transform(buffer, dimension, lanes): transform records of doubles in place
static PyObject* core_transform(PyObject* module, PyObject* args) {
    PyObject* object;
    Py_buffer view;
    Py_ssize_t dimension, lanes, count, length, word;
    double* data;

    (void)module;
    if (!PyArg_ParseTuple(args, "Onn", &object, &dimension, &lanes)) return NULL;
    word = acquire(object, &view, 1, WORD_F64);
    if (word == 0) return NULL;
    count = record_count(&view, dimension, lanes, word);
    if (count < 0) {
        PyBuffer_Release(&view);
        return NULL;
    }
    data = (double*)view.buf;
    length = dimension * lanes;
    if (count * length >= WHT_RELEASE_WORDS) {
        Py_BEGIN_ALLOW_THREADS
        for (Py_ssize_t r = 0; r < count; r++) {
            record_f64(data + r * length, (size_t)length, (size_t)lanes);
        }
        Py_END_ALLOW_THREADS
    } else {
        for (Py_ssize_t r = 0; r < count; r++) {
            record_f64(data + r * length, (size_t)length, (size_t)lanes);
        }
    }
    PyBuffer_Release(&view);
    Py_RETURN_NONE;
}

// Check that a modulus is an odd number that fits the residue arithmetic
static int check_modulus(long long modulus) {
    if (modulus < 3 || modulus > INT32_MAX || modulus % 2 == 0) {
        PyErr_SetString(PyExc_ValueError,
                "modulus must be odd and lie in 3..2**31-1");
        return -1;
    }
    return 0;
}

// transform_mod(buffer, dimension, lanes, modulus[, interval]): residues in place
static PyObject* core_transform_mod(PyObject* module, PyObject* args) {
    PyObject* object;
    Py_buffer view;
    Py_ssize_t dimension, lanes, count, length, word;
    Py_ssize_t interval = WHT_UNREDUCED;
    long long modulus;
    int64_t* data;

    (void)module;
    if (!PyArg_ParseTuple(args, "OnnL|n", &object, &dimension, &lanes, &modulus,
            &interval)) {
        return NULL;
    }
    if (check_modulus(modulus) != 0) return NULL;
    if (interval < 1 || interval > WHT_UNREDUCED) {
        PyErr_SetString(PyExc_ValueError, "interval must lie in 1..30");
        return NULL;
    }
    word = acquire(object, &view, 1, WORD_I64);
    if (word == 0) return NULL;
    count = record_count(&view, dimension, lanes, word);
    if (count < 0) {
        PyBuffer_Release(&view);
        return NULL;
    }
    data = (int64_t*)view.buf;
    length = dimension * lanes;
    if (count * length >= WHT_RELEASE_WORDS) {
        Py_BEGIN_ALLOW_THREADS
        for (Py_ssize_t r = 0; r < count; r++) {
            record_i64(data + r * length, (size_t)length, (size_t)lanes,
                    (int64_t)modulus, (size_t)interval);
        }
        Py_END_ALLOW_THREADS
    } else {
        for (Py_ssize_t r = 0; r < count; r++) {
            record_i64(data + r * length, (size_t)length, (size_t)lanes,
                    (int64_t)modulus, (size_t)interval);
        }
    }
    PyBuffer_Release(&view);
    Py_RETURN_NONE;
}

typedef struct Coordinates {
    Py_buffer loop;
    Py_buffer spectral;
    Py_buffer signed_part;
    Py_ssize_t count;
    Py_ssize_t length;
} Coordinates;

// Release the three buffers of a coordinate change
static void coordinates_release(Coordinates* buffers) {
    PyBuffer_Release(&buffers->loop);
    PyBuffer_Release(&buffers->spectral);
    PyBuffer_Release(&buffers->signed_part);
}

/*
 * Acquire the three buffers of a coordinate change. The loop buffer is the
 * output when to_loop is nonzero and the input otherwise; the other two have
 * the opposite role. The loop buffer must hold two rows for each record of
 * the spectral and signed buffers. Returns 0, or -1 with an exception set.
 */
// Acquire and check the buffers of a coordinate change
static int coordinates_acquire(Coordinates* buffers, PyObject* loop,
        PyObject* spectral, PyObject* signed_part, int to_loop, int kind,
        Py_ssize_t dimension, Py_ssize_t lanes) {
    Py_ssize_t word = acquire(loop, &buffers->loop, to_loop, kind);

    if (word == 0) return -1;
    if (acquire(spectral, &buffers->spectral, !to_loop, kind) == 0) {
        PyBuffer_Release(&buffers->loop);
        return -1;
    }
    if (acquire(signed_part, &buffers->signed_part, !to_loop, kind) == 0) {
        PyBuffer_Release(&buffers->loop);
        PyBuffer_Release(&buffers->spectral);
        return -1;
    }
    buffers->count = record_count(&buffers->spectral, dimension, lanes, word);
    buffers->length = dimension * lanes;
    if (buffers->count >= 0
            && (buffers->signed_part.len != buffers->spectral.len
                    || buffers->loop.len != 2 * buffers->spectral.len)) {
        PyErr_SetString(PyExc_ValueError,
                "the loop buffer must hold two rows for each record of the "
                "spectral and signed buffers");
        buffers->count = -1;
    }
    if (buffers->count < 0) {
        coordinates_release(buffers);
        return -1;
    }
    return 0;
}

// split(loop, spectral, signed, dimension, lanes): doubles to split form
static PyObject* core_split(PyObject* module, PyObject* args) {
    PyObject* loop;
    PyObject* spectral;
    PyObject* signed_part;
    Coordinates buffers;
    Py_ssize_t dimension, lanes, length;

    (void)module;
    if (!PyArg_ParseTuple(args, "OOOnn", &loop, &spectral, &signed_part,
            &dimension, &lanes)) {
        return NULL;
    }
    if (coordinates_acquire(&buffers, loop, spectral, signed_part, 0, WORD_F64,
            dimension, lanes) != 0) {
        return NULL;
    }
    length = buffers.length;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t r = 0; r < buffers.count; r++) {
        split_f64((const double*)buffers.loop.buf + 2 * r * length,
                (double*)buffers.spectral.buf + r * length,
                (double*)buffers.signed_part.buf + r * length, (size_t)length,
                (size_t)lanes);
    }
    Py_END_ALLOW_THREADS
    coordinates_release(&buffers);
    Py_RETURN_NONE;
}

// unsplit(spectral, signed, loop, dimension, lanes): doubles to the loop basis
static PyObject* core_unsplit(PyObject* module, PyObject* args) {
    PyObject* loop;
    PyObject* spectral;
    PyObject* signed_part;
    Coordinates buffers;
    Py_ssize_t dimension, lanes, length;

    (void)module;
    if (!PyArg_ParseTuple(args, "OOOnn", &spectral, &signed_part, &loop,
            &dimension, &lanes)) {
        return NULL;
    }
    if (coordinates_acquire(&buffers, loop, spectral, signed_part, 1, WORD_F64,
            dimension, lanes) != 0) {
        return NULL;
    }
    length = buffers.length;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t r = 0; r < buffers.count; r++) {
        unsplit_f64((const double*)buffers.spectral.buf + r * length,
                (const double*)buffers.signed_part.buf + r * length,
                (double*)buffers.loop.buf + 2 * r * length, (size_t)length,
                (size_t)lanes, (double)dimension);
    }
    Py_END_ALLOW_THREADS
    coordinates_release(&buffers);
    Py_RETURN_NONE;
}

// Allocate the workspace of one record of residues
static int64_t* residue_workspace(Py_ssize_t length) {
    int64_t* work = PyMem_Malloc((size_t)(length > 0 ? length : 1) * sizeof(int64_t));

    if (work == NULL) PyErr_NoMemory();
    return work;
}

// split_mod(loop, spectral, signed, dimension, lanes, modulus): residues
static PyObject* core_split_mod(PyObject* module, PyObject* args) {
    PyObject* loop;
    PyObject* spectral;
    PyObject* signed_part;
    Coordinates buffers;
    Py_ssize_t dimension, lanes, length;
    long long modulus;
    int64_t* work;

    (void)module;
    if (!PyArg_ParseTuple(args, "OOOnnL", &loop, &spectral, &signed_part,
            &dimension, &lanes, &modulus)) {
        return NULL;
    }
    if (check_modulus(modulus) != 0) return NULL;
    if (coordinates_acquire(&buffers, loop, spectral, signed_part, 0, WORD_U32,
            dimension, lanes) != 0) {
        return NULL;
    }
    length = buffers.length;
    work = residue_workspace(length);
    if (work == NULL) {
        coordinates_release(&buffers);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t r = 0; r < buffers.count; r++) {
        split_u32((const uint32_t*)buffers.loop.buf + 2 * r * length,
                (uint32_t*)buffers.spectral.buf + r * length,
                (uint32_t*)buffers.signed_part.buf + r * length, work,
                (size_t)length, (size_t)lanes, (int64_t)modulus);
    }
    Py_END_ALLOW_THREADS
    PyMem_Free(work);
    coordinates_release(&buffers);
    Py_RETURN_NONE;
}

// unsplit_mod(spectral, signed, loop, dimension, lanes, modulus, inverse)
static PyObject* core_unsplit_mod(PyObject* module, PyObject* args) {
    PyObject* loop;
    PyObject* spectral;
    PyObject* signed_part;
    Coordinates buffers;
    Py_ssize_t dimension, lanes, length;
    long long modulus, inverse;
    int64_t* work;

    (void)module;
    if (!PyArg_ParseTuple(args, "OOOnnLL", &spectral, &signed_part, &loop,
            &dimension, &lanes, &modulus, &inverse)) {
        return NULL;
    }
    if (check_modulus(modulus) != 0) return NULL;
    if (inverse < 1 || inverse >= modulus) {
        PyErr_SetString(PyExc_ValueError, "inverse must be a nonzero residue");
        return NULL;
    }
    if (coordinates_acquire(&buffers, loop, spectral, signed_part, 1, WORD_U32,
            dimension, lanes) != 0) {
        return NULL;
    }
    length = buffers.length;
    work = residue_workspace(length);
    if (work == NULL) {
        coordinates_release(&buffers);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t r = 0; r < buffers.count; r++) {
        unsplit_u32((const uint32_t*)buffers.spectral.buf + r * length,
                (const uint32_t*)buffers.signed_part.buf + r * length,
                (uint32_t*)buffers.loop.buf + 2 * r * length, work, (size_t)length,
                (size_t)lanes, (int64_t)modulus, (uint64_t)inverse);
    }
    Py_END_ALLOW_THREADS
    PyMem_Free(work);
    coordinates_release(&buffers);
    Py_RETURN_NONE;
}

static PyMethodDef core_methods[] = {
    {"transform", core_transform, METH_VARARGS,
            "transform(buffer, dimension, lanes)\n\n"
            "Apply the Walsh--Hadamard transform in place to every record of a\n"
            "writable buffer of float64 words."},
    {"transform_mod", core_transform_mod, METH_VARARGS,
            "transform_mod(buffer, dimension, lanes, modulus, interval=30)\n\n"
            "Apply the Walsh--Hadamard transform in place to every record of a\n"
            "writable buffer of int64 residues, leaving canonical residues."},
    {"split", core_split, METH_VARARGS,
            "split(loop, spectral, signed, dimension, lanes)\n\n"
            "Write the split form of every loop-basis record of float64 words."},
    {"unsplit", core_unsplit, METH_VARARGS,
            "unsplit(spectral, signed, loop, dimension, lanes)\n\n"
            "Write the loop-basis record of every split form of float64 words."},
    {"split_mod", core_split_mod, METH_VARARGS,
            "split_mod(loop, spectral, signed, dimension, lanes, modulus)\n\n"
            "Write the split form of every loop-basis record of uint32 residues."},
    {"unsplit_mod", core_unsplit_mod, METH_VARARGS,
            "unsplit_mod(spectral, signed, loop, dimension, lanes, modulus, inverse)\n\n"
            "Write the loop-basis record of every split form of uint32 residues;\n"
            "inverse is the inverse of the dimension modulo the modulus."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef core_module = {
    PyModuleDef_HEAD_INIT,
    "pycdloop._core",
    "Compiled kernels of pycdloop.",
    -1,
    core_methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&core_module);
}
