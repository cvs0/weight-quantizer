// y[b, m] = sum_k scale[m, g] * (code[m, k] - zero[m, g]) * channel[k] * x[b, k]
// g = k / group_size.
// int8: codes are uint8 [M, K].
// int4: packed uint8 [M, ceil(K/2)], low nibble is code[2j], high nibble is code[2j+1].
// One OpenMP worker owns a fixed subset of rows. Each row is dequantized into a
// private fp32 buffer, then dotted with every batch row (AVX2 FMA when enabled).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace py = pybind11;

namespace {

inline float at2(const float* a, int rows, int cols, int m, int g) {
    const int r = rows == 1 ? 0 : m;
    const int c = cols == 1 ? 0 : g;
    return a[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(c)];
}

#if defined(__AVX2__)
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, lo);
    lo = _mm_add_ss(lo, sh);
    return _mm_cvtss_f32(lo);
}
#endif

inline int code_at(const std::uint8_t* row, int k, int bits) {
    if (bits == 8) {
        return static_cast<int>(row[k]);
    }
    const std::uint8_t byte = row[k >> 1];
    return (k & 1) == 0 ? static_cast<int>(byte & 0x0F) : static_cast<int>(byte >> 4);
}

void dequant_row(
    const std::uint8_t* row,
    int m,
    int K,
    int bits,
    const float* scales,
    const float* zeros,
    int scale_rows,
    int groups,
    int group_size,
    const float* channel,
    float* dst) {
    const int n_groups = (K + group_size - 1) / group_size;
    for (int g = 0; g < n_groups; ++g) {
        const int k0 = g * group_size;
        const int k1 = std::min(k0 + group_size, K);
        const float scale = at2(scales, scale_rows, groups, m, g);
        const float zero = at2(zeros, scale_rows, groups, m, g);
        int k = k0;
#if defined(__AVX2__)
        if (bits == 8) {
            const __m256 z = _mm256_set1_ps(zero);
            const __m256 s = _mm256_set1_ps(scale);
            for (; k + 8 <= k1; k += 8) {
                const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + k));
                const __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
                __m256 f = _mm256_mul_ps(_mm256_sub_ps(f0, z), s);
                if (channel != nullptr) {
                    f = _mm256_mul_ps(f, _mm256_loadu_ps(channel + k));
                }
                _mm256_storeu_ps(dst + k, f);
            }
        } else if ((k0 & 1) == 0) {
            const __m256 z = _mm256_set1_ps(zero);
            const __m256 s = _mm256_set1_ps(scale);
            for (; k + 16 <= k1; k += 16) {
                const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + (k >> 1)));
                const __m128i low = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
                const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), _mm_set1_epi8(0x0F));
                const __m128i nib = _mm_unpacklo_epi8(low, high);
                __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(nib));
                __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(nib, 8)));
                f0 = _mm256_mul_ps(_mm256_sub_ps(f0, z), s);
                f1 = _mm256_mul_ps(_mm256_sub_ps(f1, z), s);
                if (channel != nullptr) {
                    f0 = _mm256_mul_ps(f0, _mm256_loadu_ps(channel + k));
                    f1 = _mm256_mul_ps(f1, _mm256_loadu_ps(channel + k + 8));
                }
                _mm256_storeu_ps(dst + k, f0);
                _mm256_storeu_ps(dst + k + 8, f1);
            }
        }
#else
        (void)bits;
#endif
        for (; k < k1; ++k) {
            float w = (static_cast<float>(code_at(row, k, bits)) - zero) * scale;
            if (channel != nullptr) {
                w *= channel[k];
            }
            dst[k] = w;
        }
    }
}

float dot_scalar(const float* a, const float* b, int K) {
    float s = 0.F;
    int k = 0;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; k + 8 <= K; k += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc);
    }
    s = hsum256(acc);
#endif
    for (; k < K; ++k) {
        s += a[k] * b[k];
    }
    return s;
}

void dots_into(const float* row, const float* x, float* y, int B, int M, int K, int m) {
    int b = 0;
#if defined(__AVX2__)
    for (; b + 4 <= B; b += 4) {
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();
        int k = 0;
        for (; k + 8 <= K; k += 8) {
            const __m256 w = _mm256_loadu_ps(row + k);
            a0 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + static_cast<std::size_t>(b + 0) * K + k), a0);
            a1 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + static_cast<std::size_t>(b + 1) * K + k), a1);
            a2 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + static_cast<std::size_t>(b + 2) * K + k), a2);
            a3 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + static_cast<std::size_t>(b + 3) * K + k), a3);
        }
        float s0 = hsum256(a0);
        float s1 = hsum256(a1);
        float s2 = hsum256(a2);
        float s3 = hsum256(a3);
        for (; k < K; ++k) {
            const float w = row[k];
            s0 += w * x[static_cast<std::size_t>(b + 0) * K + k];
            s1 += w * x[static_cast<std::size_t>(b + 1) * K + k];
            s2 += w * x[static_cast<std::size_t>(b + 2) * K + k];
            s3 += w * x[static_cast<std::size_t>(b + 3) * K + k];
        }
        y[static_cast<std::size_t>(b + 0) * M + m] = s0;
        y[static_cast<std::size_t>(b + 1) * M + m] = s1;
        y[static_cast<std::size_t>(b + 2) * M + m] = s2;
        y[static_cast<std::size_t>(b + 3) * M + m] = s3;
    }
#endif
    for (; b < B; ++b) {
        y[static_cast<std::size_t>(b) * M + m] = dot_scalar(row, x + static_cast<std::size_t>(b) * K, K);
    }
}

void run_gemm(
    const float* x,
    const std::uint8_t* q,
    const float* scales,
    const float* zeros,
    const float* channel,
    float* y,
    int B,
    int M,
    int K,
    int scale_rows,
    int groups,
    int group_size,
    int bits,
    int nthreads) {
    const int packed_cols = bits == 4 ? (K + 1) / 2 : K;
#ifdef _OPENMP
    if (nthreads < 1) {
        nthreads = omp_get_max_threads();
    }
#pragma omp parallel num_threads(nthreads)
    {
        std::vector<float> row(static_cast<std::size_t>(std::max(K, 0)));
#pragma omp for schedule(static)
        for (int m = 0; m < M; ++m) {
            const std::uint8_t* qrow = q + static_cast<std::size_t>(m) * static_cast<std::size_t>(packed_cols);
            dequant_row(qrow, m, K, bits, scales, zeros, scale_rows, groups, group_size, channel, row.data());
            dots_into(row.data(), x, y, B, M, K, m);
        }
    }
#else
    if (nthreads < 1) {
        nthreads = 1;
    }
    (void)nthreads;
    std::vector<float> row(static_cast<std::size_t>(std::max(K, 0)));
    for (int m = 0; m < M; ++m) {
        const std::uint8_t* qrow = q + static_cast<std::size_t>(m) * static_cast<std::size_t>(packed_cols);
        dequant_row(qrow, m, K, bits, scales, zeros, scale_rows, groups, group_size, channel, row.data());
        dots_into(row.data(), x, y, B, M, K, m);
    }
#endif
}

}  // namespace

py::array_t<float> dequant_gemm(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> q,
    py::array_t<float, py::array::c_style | py::array::forcecast> scales,
    py::array_t<float, py::array::c_style | py::array::forcecast> zeros,
    py::object channel_scale,
    int group_size,
    int bits,
    int nthreads) {
    auto xr = x.request();
    auto qr = q.request();
    auto sr = scales.request();
    auto zr = zeros.request();
    if (xr.ndim != 2) {
        throw std::invalid_argument("x must have shape [B, K]");
    }
    if (qr.ndim != 2 || sr.ndim != 2 || zr.ndim != 2) {
        throw std::invalid_argument("q, scales, and zeros must be rank 2");
    }
    if (bits != 4 && bits != 8) {
        throw std::invalid_argument("bits must be 4 or 8");
    }
    if (group_size <= 0) {
        throw std::invalid_argument("group_size must be positive");
    }
    const int B = static_cast<int>(xr.shape[0]);
    const int K = static_cast<int>(xr.shape[1]);
    const int M = static_cast<int>(qr.shape[0]);
    const int scale_rows = static_cast<int>(sr.shape[0]);
    const int groups = static_cast<int>(sr.shape[1]);
    if (K < 0 || M < 0 || B < 0) {
        throw std::invalid_argument("negative dimension");
    }
    const int packed_cols = bits == 4 ? (K + 1) / 2 : K;
    if (static_cast<int>(qr.shape[1]) != packed_cols) {
        throw std::invalid_argument("packed code row length does not match K");
    }
    if (zr.shape[0] != sr.shape[0] || zr.shape[1] != sr.shape[1]) {
        throw std::invalid_argument("zeros shape must match scales");
    }
    if (scale_rows != 1 && scale_rows != M) {
        throw std::invalid_argument("scale rows must be 1 or M");
    }
    const int n_groups = K == 0 ? 0 : (K + group_size - 1) / group_size;
    if (K > 0 && groups != n_groups) {
        throw std::invalid_argument("scale groups must equal ceil(K / group_size)");
    }
    const float* channel = nullptr;
    py::array_t<float, py::array::c_style | py::array::forcecast> channel_holder;
    if (!channel_scale.is_none()) {
        channel_holder = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(channel_scale);
        auto cr = channel_holder.request();
        if (cr.ndim != 1 || static_cast<int>(cr.shape[0]) != K) {
            throw std::invalid_argument("channel_scale must have shape [K]");
        }
        channel = static_cast<const float*>(cr.ptr);
    }
    auto y = py::array_t<float>({xr.shape[0], static_cast<py::ssize_t>(M)});
    auto yr = y.request();
    {
        py::gil_scoped_release release;
        run_gemm(
            static_cast<const float*>(xr.ptr),
            static_cast<const std::uint8_t*>(qr.ptr),
            static_cast<const float*>(sr.ptr),
            static_cast<const float*>(zr.ptr),
            channel,
            static_cast<float*>(yr.ptr),
            B,
            M,
            K,
            scale_rows,
            groups,
            group_size,
            bits,
            nthreads);
    }
    return y;
}

py::dict kernel_info() {
    py::dict info;
#if defined(__AVX2__)
    info["avx2"] = true;
#else
    info["avx2"] = false;
#endif
#ifdef _OPENMP
    info["openmp"] = true;
#else
    info["openmp"] = false;
#endif
    return info;
}

PYBIND11_MODULE(_kernels, m) {
    m.doc() = "fp32 activation times int8/int4 dequantized weight";
    m.def(
        "dequant_gemm",
        &dequant_gemm,
        py::arg("x"),
        py::arg("q"),
        py::arg("scales"),
        py::arg("zeros"),
        py::arg("channel_scale"),
        py::arg("group_size"),
        py::arg("bits"),
        py::arg("nthreads"));
    m.def("kernel_info", &kernel_info);
}
