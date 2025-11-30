//
// MIT License
// Copyright (c) 2025 Eviatar
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#pragma once
#include <cmath>
#include <cstdint>
#include <type_traits>

#if defined(_MSC_VER)
# define _SIMD_COMPILER_MSVC
#elif defined(__GNUC__) || defined(__clang__)
# define _SIMD_COMPILER_GCC_CLANG
#else
# define _SIMD_COMPILER_UNKNOWN
#endif

#ifdef _SIMD_COMPILER_MSVC
# include <intrin.h>
#else
# include <cpuid.h>
#endif

#ifdef __FMA__
# define _SIMD_FMA
#endif

#ifdef __SSE__
# define _SIMD_SSE
# ifdef __SSE2__
#  define _SIMD_SSE2
#  ifdef __SSE3__
#   define _SIMD_SSE3
#   ifdef __SSE4_1__
#    define _SIMD_SSE4_1
#    ifdef __SSE4_2__
#     define _SIMD_SSE4_2
#    endif
#   endif
#  endif
# endif
#endif

#ifdef __AVX__
# define _SIMD_AVX
# ifdef __AVX2__
#  define _SIMD_AVX2
#  ifdef __AVX512F__
#   define _SIMD_AVX512F
#  endif
# endif
#endif

namespace simd {
    // ============================================================================
    // Vector Traits
    // ============================================================================

    template<typename VectorT, typename ScalarT, int Lanes>
    struct vector_traits_base {
        using scalar_type = ScalarT;
        using mask_type = VectorT;
        static constexpr size_t lanes = Lanes;
    };

    template<typename T, typename Tag = signed>
    struct vector_traits;

    template<>
    struct vector_traits<int64_t, signed> : vector_traits_base<int64_t, int64_t, 4> {
    };

    template<>
    struct vector_traits<uint64_t, unsigned> : vector_traits_base<uint64_t, uint64_t, 4> {
    };

    template<>
    struct vector_traits<long double> : vector_traits_base<long double, long double, 8> {
    };

    template<>
    struct vector_traits<float> : vector_traits_base<float, float, 4> {
    };

    template<>
    struct vector_traits<__m128i, signed> : vector_traits_base<__m128i, int32_t, 4> {
    };

    template<>
    struct vector_traits<__m128i, unsigned> : vector_traits_base<__m128i, uint32_t, 4> {
    };

    template<>
    struct vector_traits<__m128> : vector_traits_base<__m128, float, 4> {
    };

    template<>
    struct vector_traits<__m128d> : vector_traits_base<__m128d, double, 2> {
    };

    template<>
    struct vector_traits<__m256i, signed> : vector_traits_base<__m256i, int32_t, 8> {
    };

    template<>
    struct vector_traits<__m256i, unsigned> : vector_traits_base<__m256i, uint32_t, 8> {
    };

    template<>
    struct vector_traits<__m256> : vector_traits_base<__m256, float, 8> {
    };

    template<>
    struct vector_traits<__m256d> : vector_traits_base<__m256d, double, 4> {
    };

    template<>
    struct vector_traits<__m512i, signed> : vector_traits_base<__m512i, int32_t, 16> {
    };

    template<>
    struct vector_traits<__m512i, unsigned> : vector_traits_base<__m512i, uint32_t, 16> {
    };

    template<>
    struct vector_traits<__m512> : vector_traits_base<__m512, float, 16> {
    };

    template<>
    struct vector_traits<__m512d> : vector_traits_base<__m512d, double, 8> {
    };

    // ============================================================================
    // Scalar Integer Vector
    // ============================================================================

    template<typename T, typename Tag = signed>
    class vector;

    template<typename Tag>
    class vector<int64_t, Tag> {
        static_assert(std::is_same_v<Tag, signed> || std::is_same_v<Tag, unsigned>,
                      "Tag must be signed or unsigned");

    public:
        using scalar_type = vector_traits<int64_t, Tag>::scalar_type;
        using mask_type = vector_traits<int64_t, Tag>::mask_type;

    private:
        static constexpr size_t num_lanes = vector_traits<scalar_type>::lanes;
        scalar_type m_data[num_lanes]{};

    public:
        // Constructors
        vector() noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] = 0;
            }
        }

        explicit vector(const scalar_type val) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] = val;
            }
        }

        explicit vector(const scalar_type *data) noexcept {
            load(data);
        }

        // Access
        scalar_type *get() noexcept { return m_data; }
        [[nodiscard]] const scalar_type *get() const noexcept { return m_data; }

        // Load/Store
        void load(const scalar_type *data) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] = data[i];
            }
        }

        void store(scalar_type *data) const noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                data[i] = m_data[i];
            }
        }

        void load_aligned(const scalar_type *data) noexcept {
            load(data); // Alignment doesn't matter for scalar
        }

        void store_aligned(scalar_type *data) const noexcept {
            store(data); // Alignment doesn't matter for scalar
        }

        // Arithmetic operators
        vector operator+(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] + other.m_data[i];
            }
            return result;
        }

        vector operator-(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] - other.m_data[i];
            }
            return result;
        }

        vector operator*(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] * other.m_data[i];
            }
            return result;
        }

        vector &operator+=(const vector &other) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] += other.m_data[i];
            }
            return *this;
        }

        vector &operator-=(const vector &other) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] -= other.m_data[i];
            }
            return *this;
        }

        vector &operator*=(const vector &other) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] *= other.m_data[i];
            }
            return *this;
        }

        // Unary operators
        vector operator-() const noexcept requires (std::is_same_v<Tag, signed>) {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = -m_data[i];
            }
            return result;
        }

        // Bitwise operators
        vector operator&(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] & other.m_data[i];
            }
            return result;
        }

        vector operator|(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] | other.m_data[i];
            }
            return result;
        }

        vector operator^(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] ^ other.m_data[i];
            }
            return result;
        }

        vector operator~() const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = ~m_data[i];
            }
            return result;
        }

        vector &operator&=(const vector &other) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] &= other.m_data[i];
            }
            return *this;
        }

        vector &operator|=(const vector &other) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] |= other.m_data[i];
            }
            return *this;
        }

        vector &operator^=(const vector &other) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] ^= other.m_data[i];
            }
            return *this;
        }

        vector operator<<(const int32_t count) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] << count;
            }
            return result;
        }

        vector operator>>(const int32_t count) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                if constexpr (std::is_same_v<Tag, signed>) {
                    result.m_data[i] = m_data[i] >> count; // Arithmetic shift
                } else {
                    result.m_data[i] = static_cast<std::make_unsigned_t<scalar_type>>(m_data[i]) >> count;
                }
            }
            return result;
        }

        vector &operator<<=(const int32_t count) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] <<= count;
            }
            return *this;
        }

        vector &operator>>=(const int32_t count) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                if constexpr (std::is_same_v<Tag, signed>) {
                    m_data[i] >>= count; // Arithmetic shift
                } else {
                    m_data[i] = static_cast<std::make_unsigned_t<scalar_type>>(m_data[i]) >> count;
                }
            }
            return *this;
        }

        // Comparison operators (return mask with all bits set for true)
        [[nodiscard]] vector cmp_eq(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] == other.m_data[i] ? -1 : 0;
            }
            return result;
        }

        [[nodiscard]] vector cmp_ne(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] != other.m_data[i] ? -1 : 0;
            }
            return result;
        }

        [[nodiscard]] vector cmp_lt(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] < other.m_data[i] ? -1 : 0;
            }
            return result;
        }

        [[nodiscard]] vector cmp_le(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] <= other.m_data[i] ? -1 : 0;
            }
            return result;
        }

        [[nodiscard]] vector cmp_gt(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] > other.m_data[i] ? -1 : 0;
            }
            return result;
        }

        [[nodiscard]] vector cmp_ge(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] >= other.m_data[i] ? -1 : 0;
            }
            return result;
        }

        vector operator==(const vector &other) const noexcept { return cmp_eq(other); }
        vector operator!=(const vector &other) const noexcept { return cmp_ne(other); }
        vector operator<(const vector &other) const noexcept { return cmp_lt(other); }
        vector operator<=(const vector &other) const noexcept { return cmp_le(other); }
        vector operator>(const vector &other) const noexcept { return cmp_gt(other); }
        vector operator>=(const vector &other) const noexcept { return cmp_ge(other); }

        // Static functions
        static vector blend(const vector &a, const vector &b, const vector &mask) noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = mask.m_data[i] ? a.m_data[i] : b.m_data[i];
            }
            return result;
        }

        static vector select(const vector &mask, const vector &a, const vector &b) noexcept {
            return blend(a, b, mask);
        }

        static vector min(const vector &a, const vector &b) noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = a.m_data[i] < b.m_data[i] ? a.m_data[i] : b.m_data[i];
            }
            return result;
        }

        static vector max(const vector &a, const vector &b) noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = a.m_data[i] > b.m_data[i] ? a.m_data[i] : b.m_data[i];
            }
            return result;
        }

        static vector abs(const vector &v) noexcept requires (std::is_same_v<Tag, signed>) {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = v.m_data[i] < 0 ? -v.m_data[i] : v.m_data[i];
            }
            return result;
        }

        static vector zero() noexcept {
            return vector(0);
        }

        static vector set1(const scalar_type val) noexcept {
            return vector(val);
        }

        static constexpr size_t size() noexcept { return num_lanes; }

        // Horizontal operations
        [[nodiscard]] scalar_type horizontal_sum() const noexcept {
            scalar_type sum = 0;
            for (size_t i = 0; i < num_lanes; ++i) {
                sum += m_data[i];
            }
            return sum;
        }

        [[nodiscard]] scalar_type horizontal_min() const noexcept {
            scalar_type result = m_data[0];
            for (size_t i = 1; i < num_lanes; ++i) {
                if (m_data[i] < result) result = m_data[i];
            }
            return result;
        }

        [[nodiscard]] scalar_type horizontal_max() const noexcept {
            scalar_type result = m_data[0];
            for (size_t i = 1; i < num_lanes; ++i) {
                if (m_data[i] > result) result = m_data[i];
            }
            return result;
        }
    };

    // ============================================================================
    // Scalar Float Vector
    // ============================================================================

    template<>
    class vector<float> {
    public:
        using scalar_type = vector_traits<float>::scalar_type;
        using mask_type = vector_traits<float>::mask_type;

    private:
        static constexpr size_t num_lanes = vector_traits<scalar_type>::lanes;
        scalar_type m_data[num_lanes]{};

    public:
        // Constructors
        vector() noexcept {
            for (float &i: m_data) {
                i = 0.0f;
            }
        }

        explicit vector(const scalar_type val) noexcept {
            for (float &i: m_data) {
                i = val;
            }
        }

        explicit vector(const scalar_type *data) noexcept {
            load(data);
        }

        // Access
        scalar_type *get() noexcept { return m_data; }
        [[nodiscard]] const scalar_type *get() const noexcept { return m_data; }

        // Load/Store
        void load(const scalar_type *data) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] = data[i];
            }
        }

        void store(scalar_type *data) const noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                data[i] = m_data[i];
            }
        }

        void load_aligned(const scalar_type *data) noexcept {
            load(data); // Alignment doesn't matter for scalar
        }

        void store_aligned(scalar_type *data) const noexcept {
            store(data); // Alignment doesn't matter for scalar
        }

        // Arithmetic operators
        vector operator+(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] + other.m_data[i];
            }
            return result;
        }

        vector operator-(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] - other.m_data[i];
            }
            return result;
        }

        vector operator*(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] * other.m_data[i];
            }
            return result;
        }

        vector operator/(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] / other.m_data[i];
            }
            return result;
        }

        vector &operator+=(const vector &other) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] += other.m_data[i];
            }
            return *this;
        }

        vector &operator-=(const vector &other) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] -= other.m_data[i];
            }
            return *this;
        }

        vector &operator*=(const vector &other) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] *= other.m_data[i];
            }
            return *this;
        }

        vector &operator/=(const vector &other) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] /= other.m_data[i];
            }
            return *this;
        }

        // Unary operators
        vector operator-() const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = -m_data[i];
            }
            return result;
        }

        // Bitwise operators (operate on bit representation)
        vector operator&(const vector &other) const noexcept {
            vector result;
            const auto *a = reinterpret_cast<const uint32_t *>(m_data);
            const auto *b = reinterpret_cast<const uint32_t *>(other.m_data);
            auto *r = reinterpret_cast<uint32_t *>(result.m_data);
            for (size_t i = 0; i < num_lanes; ++i) {
                r[i] = a[i] & b[i];
            }
            return result;
        }

        vector operator|(const vector &other) const noexcept {
            vector result;
            const auto *a = reinterpret_cast<const uint32_t *>(m_data);
            const auto *b = reinterpret_cast<const uint32_t *>(other.m_data);
            auto *r = reinterpret_cast<uint32_t *>(result.m_data);
            for (size_t i = 0; i < num_lanes; ++i) {
                r[i] = a[i] | b[i];
            }
            return result;
        }

        vector operator^(const vector &other) const noexcept {
            vector result;
            const auto *a = reinterpret_cast<const uint32_t *>(m_data);
            const auto *b = reinterpret_cast<const uint32_t *>(other.m_data);
            auto *r = reinterpret_cast<uint32_t *>(result.m_data);
            for (size_t i = 0; i < num_lanes; ++i) {
                r[i] = a[i] ^ b[i];
            }
            return result;
        }

        vector operator~() const noexcept {
            vector result;
            const auto *a = reinterpret_cast<const uint32_t *>(m_data);
            auto *r = reinterpret_cast<uint32_t *>(result.m_data);
            for (size_t i = 0; i < num_lanes; ++i) {
                r[i] = ~a[i];
            }
            return result;
        }

        vector &operator&=(const vector &other) noexcept {
            auto *a = reinterpret_cast<uint32_t *>(m_data);
            const auto *b = reinterpret_cast<const uint32_t *>(other.m_data);
            for (size_t i = 0; i < num_lanes; ++i) {
                a[i] &= b[i];
            }
            return *this;
        }

        vector &operator|=(const vector &other) noexcept {
            auto *a = reinterpret_cast<uint32_t *>(m_data);
            const auto *b = reinterpret_cast<const uint32_t *>(other.m_data);
            for (size_t i = 0; i < num_lanes; ++i) {
                a[i] |= b[i];
            }
            return *this;
        }

        vector &operator^=(const vector &other) noexcept {
            auto *a = reinterpret_cast<uint32_t *>(m_data);
            const auto *b = reinterpret_cast<const uint32_t *>(other.m_data);
            for (size_t i = 0; i < num_lanes; ++i) {
                a[i] ^= b[i];
            }
            return *this;
        }

        // Comparison operators (return mask with all bits set for true)
        [[nodiscard]] vector cmp_eq(const vector &other) const noexcept {
            vector result;
            auto *r = reinterpret_cast<uint32_t *>(result.m_data);
            for (size_t i = 0; i < num_lanes; ++i) {
                constexpr uint32_t true_mask = 0xFFFFFFFF;
                r[i] = (m_data[i] == other.m_data[i]) ? true_mask : 0;
            }
            return result;
        }

        [[nodiscard]] vector cmp_ne(const vector &other) const noexcept {
            vector result;
            auto *r = reinterpret_cast<uint32_t *>(result.m_data);
            for (size_t i = 0; i < num_lanes; ++i) {
                constexpr uint32_t true_mask = 0xFFFFFFFF;
                r[i] = (m_data[i] != other.m_data[i]) ? true_mask : 0;
            }
            return result;
        }

        [[nodiscard]] vector cmp_lt(const vector &other) const noexcept {
            vector result;
            auto *r = reinterpret_cast<uint32_t *>(result.m_data);
            for (size_t i = 0; i < num_lanes; ++i) {
                constexpr uint32_t true_mask = 0xFFFFFFFF;
                r[i] = (m_data[i] < other.m_data[i]) ? true_mask : 0;
            }
            return result;
        }

        [[nodiscard]] vector cmp_le(const vector &other) const noexcept {
            vector result;
            auto *r = reinterpret_cast<uint32_t *>(result.m_data);
            for (size_t i = 0; i < num_lanes; ++i) {
                constexpr uint32_t true_mask = 0xFFFFFFFF;
                r[i] = (m_data[i] <= other.m_data[i]) ? true_mask : 0;
            }
            return result;
        }

        [[nodiscard]] vector cmp_gt(const vector &other) const noexcept {
            vector result;
            auto *r = reinterpret_cast<uint32_t *>(result.m_data);
            for (size_t i = 0; i < num_lanes; ++i) {
                constexpr uint32_t true_mask = 0xFFFFFFFF;
                r[i] = (m_data[i] > other.m_data[i]) ? true_mask : 0;
            }
            return result;
        }

        [[nodiscard]] vector cmp_ge(const vector &other) const noexcept {
            vector result;
            auto *r = reinterpret_cast<uint32_t *>(result.m_data);
            for (size_t i = 0; i < num_lanes; ++i) {
                constexpr uint32_t true_mask = 0xFFFFFFFF;
                r[i] = (m_data[i] >= other.m_data[i]) ? true_mask : 0;
            }
            return result;
        }

        vector operator==(const vector &other) const noexcept { return cmp_eq(other); }
        vector operator!=(const vector &other) const noexcept { return cmp_ne(other); }
        vector operator<(const vector &other) const noexcept { return cmp_lt(other); }
        vector operator<=(const vector &other) const noexcept { return cmp_le(other); }
        vector operator>(const vector &other) const noexcept { return cmp_gt(other); }
        vector operator>=(const vector &other) const noexcept { return cmp_ge(other); }

        // Static functions
        static vector blend(const vector &a, const vector &b, const vector &mask) noexcept {
            vector result;
            const auto *m = reinterpret_cast<const uint32_t *>(mask.m_data);
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m[i] ? a.m_data[i] : b.m_data[i];
            }
            return result;
        }

        static vector select(const vector &mask, const vector &a, const vector &b) noexcept {
            return blend(a, b, mask);
        }

        static vector min(const vector &a, const vector &b) noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = a.m_data[i] < b.m_data[i] ? a.m_data[i] : b.m_data[i];
            }
            return result;
        }

        static vector max(const vector &a, const vector &b) noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = a.m_data[i] > b.m_data[i] ? a.m_data[i] : b.m_data[i];
            }
            return result;
        }

        static vector sqrt(const vector &v) noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = std::sqrt(v.m_data[i]);
            }
            return result;
        }

        static vector abs(const vector &v) noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = std::abs(v.m_data[i]);
            }
            return result;
        }

        static vector fma(const vector &a, const vector &b, const vector &c) noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = a.m_data[i] * b.m_data[i] + c.m_data[i];
            }
            return result;
        }

        static vector zero() noexcept {
            return vector(0.0f);
        }

        static vector set1(const scalar_type val) noexcept {
            return vector(val);
        }

        static constexpr size_t size() noexcept { return num_lanes; }

        // Horizontal operations
        [[nodiscard]] scalar_type horizontal_sum() const noexcept {
            scalar_type sum = 0.0f;
            for (const float i: m_data) {
                sum += i;
            }
            return sum;
        }

        [[nodiscard]] scalar_type horizontal_min() const noexcept {
            scalar_type result = m_data[0];
            for (size_t i = 1; i < num_lanes; ++i) {
                if (m_data[i] < result) result = m_data[i];
            }
            return result;
        }

        [[nodiscard]] scalar_type horizontal_max() const noexcept {
            scalar_type result = m_data[0];
            for (size_t i = 1; i < num_lanes; ++i) {
                if (m_data[i] > result) result = m_data[i];
            }
            return result;
        }
    };

    // ============================================================================
    // Scalar Double Vector
    // ============================================================================

    template<>
    class vector<long double> {
    public:
        using scalar_type = vector_traits<long double>::scalar_type;
        using mask_type = vector_traits<long double>::mask_type;

    private:
        static constexpr size_t num_lanes = vector_traits<scalar_type>::lanes;
        scalar_type m_data[num_lanes]{};

    public:
        // Constructors
        vector() noexcept {
            for (long double &i: m_data) {
                i = 0.0L;
            }
        }

        explicit vector(const scalar_type val) noexcept {
            for (long double &i: m_data) {
                i = val;
            }
        }

        explicit vector(const scalar_type *data) noexcept {
            load(data);
        }

        // Access
        scalar_type *get() noexcept { return m_data; }
        [[nodiscard]] const scalar_type *get() const noexcept { return m_data; }

        // Load/Store
        void load(const scalar_type *data) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] = data[i];
            }
        }

        void store(scalar_type *data) const noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                data[i] = m_data[i];
            }
        }

        void load_aligned(const scalar_type *data) noexcept {
            load(data); // Alignment doesn't matter for scalar
        }

        void store_aligned(scalar_type *data) const noexcept {
            store(data); // Alignment doesn't matter for scalar
        }

        // Arithmetic operators
        vector operator+(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] + other.m_data[i];
            }
            return result;
        }

        vector operator-(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] - other.m_data[i];
            }
            return result;
        }

        vector operator*(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] * other.m_data[i];
            }
            return result;
        }

        vector operator/(const vector &other) const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m_data[i] / other.m_data[i];
            }
            return result;
        }

        vector &operator+=(const vector &other) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] += other.m_data[i];
            }
            return *this;
        }

        vector &operator-=(const vector &other) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] -= other.m_data[i];
            }
            return *this;
        }

        vector &operator*=(const vector &other) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] *= other.m_data[i];
            }
            return *this;
        }

        vector &operator/=(const vector &other) noexcept {
            for (size_t i = 0; i < num_lanes; ++i) {
                m_data[i] /= other.m_data[i];
            }
            return *this;
        }

        // Unary operators
        vector operator-() const noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = -m_data[i];
            }
            return result;
        }

        // Bitwise operators (operate on bit representation)
        vector operator&(const vector &other) const noexcept {
            vector result;
            const auto *a = reinterpret_cast<const uint64_t *>(m_data);
            const auto *b = reinterpret_cast<const uint64_t *>(other.m_data);
            auto *r = reinterpret_cast<uint64_t *>(result.m_data);
            constexpr size_t chunks = num_lanes * sizeof(scalar_type) / sizeof(uint64_t);
            for (size_t i = 0; i < chunks; ++i) {
                r[i] = a[i] & b[i];
            }
            return result;
        }

        vector operator|(const vector &other) const noexcept {
            vector result;
            const auto *a = reinterpret_cast<const uint64_t *>(m_data);
            const auto *b = reinterpret_cast<const uint64_t *>(other.m_data);
            auto *r = reinterpret_cast<uint64_t *>(result.m_data);
            constexpr size_t chunks = num_lanes * sizeof(scalar_type) / sizeof(uint64_t);
            for (size_t i = 0; i < chunks; ++i) {
                r[i] = a[i] | b[i];
            }
            return result;
        }

        vector operator^(const vector &other) const noexcept {
            vector result;
            const auto *a = reinterpret_cast<const uint64_t *>(m_data);
            const auto *b = reinterpret_cast<const uint64_t *>(other.m_data);
            auto *r = reinterpret_cast<uint64_t *>(result.m_data);
            constexpr size_t chunks = num_lanes * sizeof(scalar_type) / sizeof(uint64_t);
            for (size_t i = 0; i < chunks; ++i) {
                r[i] = a[i] ^ b[i];
            }
            return result;
        }

        vector operator~() const noexcept {
            vector result;
            const auto *a = reinterpret_cast<const uint64_t *>(m_data);
            auto *r = reinterpret_cast<uint64_t *>(result.m_data);
            constexpr size_t chunks = num_lanes * sizeof(scalar_type) / sizeof(uint64_t);
            for (size_t i = 0; i < chunks; ++i) {
                r[i] = ~a[i];
            }
            return result;
        }

        vector &operator&=(const vector &other) noexcept {
            auto *a = reinterpret_cast<uint64_t *>(m_data);
            const auto *b = reinterpret_cast<const uint64_t *>(other.m_data);
            constexpr size_t chunks = num_lanes * sizeof(scalar_type) / sizeof(uint64_t);
            for (size_t i = 0; i < chunks; ++i) {
                a[i] &= b[i];
            }
            return *this;
        }

        vector &operator|=(const vector &other) noexcept {
            auto *a = reinterpret_cast<uint64_t *>(m_data);
            const auto *b = reinterpret_cast<const uint64_t *>(other.m_data);
            constexpr size_t chunks = num_lanes * sizeof(scalar_type) / sizeof(uint64_t);
            for (size_t i = 0; i < chunks; ++i) {
                a[i] |= b[i];
            }
            return *this;
        }

        vector &operator^=(const vector &other) noexcept {
            auto *a = reinterpret_cast<uint64_t *>(m_data);
            const auto *b = reinterpret_cast<const uint64_t *>(other.m_data);
            constexpr size_t chunks = num_lanes * sizeof(scalar_type) / sizeof(uint64_t);
            for (size_t i = 0; i < chunks; ++i) {
                a[i] ^= b[i];
            }
            return *this;
        }

        // Comparison operators (return mask with all bits set for true)
        [[nodiscard]] vector cmp_eq(const vector &other) const noexcept {
            vector result;
            auto *r = reinterpret_cast<uint64_t *>(result.m_data);
            constexpr size_t chunks_per_lane = sizeof(scalar_type) / sizeof(uint64_t);
            for (size_t i = 0; i < num_lanes; ++i) {
                const bool cmp = m_data[i] == other.m_data[i];
                for (size_t j = 0; j < chunks_per_lane; ++j) {
                    constexpr uint64_t true_mask = 0xFFFFFFFFFFFFFFFFULL;
                    r[i * chunks_per_lane + j] = cmp ? true_mask : 0;
                }
            }
            return result;
        }

        [[nodiscard]] vector cmp_ne(const vector &other) const noexcept {
            vector result;
            auto *r = reinterpret_cast<uint64_t *>(result.m_data);
            constexpr size_t chunks_per_lane = sizeof(scalar_type) / sizeof(uint64_t);
            for (size_t i = 0; i < num_lanes; ++i) {
                const bool cmp = m_data[i] != other.m_data[i];
                for (size_t j = 0; j < chunks_per_lane; ++j) {
                    constexpr uint64_t true_mask = 0xFFFFFFFFFFFFFFFFULL;
                    r[i * chunks_per_lane + j] = cmp ? true_mask : 0;
                }
            }
            return result;
        }

        [[nodiscard]] vector cmp_lt(const vector &other) const noexcept {
            vector result;
            auto *r = reinterpret_cast<uint64_t *>(result.m_data);
            constexpr size_t chunks_per_lane = sizeof(scalar_type) / sizeof(uint64_t);
            for (size_t i = 0; i < num_lanes; ++i) {
                const bool cmp = m_data[i] < other.m_data[i];
                for (size_t j = 0; j < chunks_per_lane; ++j) {
                    constexpr uint64_t true_mask = 0xFFFFFFFFFFFFFFFFULL;
                    r[i * chunks_per_lane + j] = cmp ? true_mask : 0;
                }
            }
            return result;
        }

        [[nodiscard]] vector cmp_le(const vector &other) const noexcept {
            vector result;
            auto *r = reinterpret_cast<uint64_t *>(result.m_data);
            constexpr size_t chunks_per_lane = sizeof(scalar_type) / sizeof(uint64_t);
            for (size_t i = 0; i < num_lanes; ++i) {
                const bool cmp = m_data[i] <= other.m_data[i];
                for (size_t j = 0; j < chunks_per_lane; ++j) {
                    constexpr uint64_t true_mask = 0xFFFFFFFFFFFFFFFFULL;
                    r[i * chunks_per_lane + j] = cmp ? true_mask : 0;
                }
            }
            return result;
        }

        [[nodiscard]] vector cmp_gt(const vector &other) const noexcept {
            vector result;
            auto *r = reinterpret_cast<uint64_t *>(result.m_data);
            constexpr size_t chunks_per_lane = sizeof(scalar_type) / sizeof(uint64_t);
            for (size_t i = 0; i < num_lanes; ++i) {
                const bool cmp = m_data[i] > other.m_data[i];
                for (size_t j = 0; j < chunks_per_lane; ++j) {
                    constexpr uint64_t true_mask = 0xFFFFFFFFFFFFFFFFULL;
                    r[i * chunks_per_lane + j] = cmp ? true_mask : 0;
                }
            }
            return result;
        }

        [[nodiscard]] vector cmp_ge(const vector &other) const noexcept {
            vector result;
            auto *r = reinterpret_cast<uint64_t *>(result.m_data);
            constexpr size_t chunks_per_lane = sizeof(scalar_type) / sizeof(uint64_t);
            for (size_t i = 0; i < num_lanes; ++i) {
                const bool cmp = m_data[i] >= other.m_data[i];
                for (size_t j = 0; j < chunks_per_lane; ++j) {
                    constexpr uint64_t true_mask = 0xFFFFFFFFFFFFFFFFULL;
                    r[i * chunks_per_lane + j] = cmp ? true_mask : 0;
                }
            }
            return result;
        }

        vector operator==(const vector &other) const noexcept { return cmp_eq(other); }
        vector operator!=(const vector &other) const noexcept { return cmp_ne(other); }
        vector operator<(const vector &other) const noexcept { return cmp_lt(other); }
        vector operator<=(const vector &other) const noexcept { return cmp_le(other); }
        vector operator>(const vector &other) const noexcept { return cmp_gt(other); }
        vector operator>=(const vector &other) const noexcept { return cmp_ge(other); }

        // Static functions
        static vector blend(const vector &a, const vector &b, const vector &mask) noexcept {
            vector result;
            const auto *m = reinterpret_cast<const uint64_t *>(mask.m_data);
            constexpr size_t chunks_per_lane = sizeof(scalar_type) / sizeof(uint64_t);
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = m[i * chunks_per_lane] ? a.m_data[i] : b.m_data[i];
            }
            return result;
        }

        static vector select(const vector &mask, const vector &a, const vector &b) noexcept {
            return blend(a, b, mask);
        }

        static vector min(const vector &a, const vector &b) noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = a.m_data[i] < b.m_data[i] ? a.m_data[i] : b.m_data[i];
            }
            return result;
        }

        static vector max(const vector &a, const vector &b) noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = a.m_data[i] > b.m_data[i] ? a.m_data[i] : b.m_data[i];
            }
            return result;
        }

        static vector sqrt(const vector &v) noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = std::sqrt(v.m_data[i]);
            }
            return result;
        }

        static vector abs(const vector &v) noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = std::abs(v.m_data[i]);
            }
            return result;
        }

        static vector fma(const vector &a, const vector &b, const vector &c) noexcept {
            vector result;
            for (size_t i = 0; i < num_lanes; ++i) {
                result.m_data[i] = a.m_data[i] * b.m_data[i] + c.m_data[i];
            }
            return result;
        }

        static vector zero() noexcept {
            return vector(0.0L);
        }

        static vector set1(const scalar_type val) noexcept {
            return vector(val);
        }

        static constexpr size_t size() noexcept { return num_lanes; }

        // Horizontal operations
        [[nodiscard]] scalar_type horizontal_sum() const noexcept {
            scalar_type sum = 0.0L;
            for (const long double i: m_data) {
                sum += i;
            }
            return sum;
        }

        [[nodiscard]] scalar_type horizontal_min() const noexcept {
            scalar_type result = m_data[0];
            for (size_t i = 1; i < num_lanes; ++i) {
                if (m_data[i] < result) result = m_data[i];
            }
            return result;
        }

        [[nodiscard]] scalar_type horizontal_max() const noexcept {
            scalar_type result = m_data[0];
            for (size_t i = 1; i < num_lanes; ++i) {
                if (m_data[i] > result) result = m_data[i];
            }
            return result;
        }
    };

    // ============================================================================
    // SSE Integer Vector (__m128i with int32_t)
    // ============================================================================

    template<typename Tag>
    class vector<__m128i, Tag> {
        static_assert(std::is_same_v<Tag, signed> || std::is_same_v<Tag, unsigned>,
                      "Tag must be signed or unsigned");

    public:
        using scalar_type = vector_traits<__m128i, Tag>::scalar_type;
        using mask_type = vector_traits<__m128i, Tag>::mask_type;

    private:
        mask_type m_value{};

    public:
        // Constructors
        vector() noexcept = default;

        explicit vector(const mask_type val) noexcept : m_value(val) {
        }

        explicit vector(const scalar_type *data) noexcept { load(data); }

        // Access
        mask_type &get() noexcept { return m_value; }
        [[nodiscard]] const mask_type &get() const noexcept { return m_value; }

        // Load/Store
        void load(const scalar_type *data) noexcept {
            m_value = _mm_loadu_si128(reinterpret_cast<const mask_type *>(data));
        }

        void store(scalar_type *data) const noexcept {
            _mm_storeu_si128(reinterpret_cast<mask_type *>(data), m_value);
        }

        void load_aligned(const scalar_type *data) noexcept {
            m_value = _mm_load_si128(reinterpret_cast<const mask_type *>(data));
        }

        void store_aligned(scalar_type *data) const noexcept {
            _mm_store_si128(reinterpret_cast<mask_type *>(data), m_value);
        }

        // Arithmetic operators
        vector operator+(const vector &other) const noexcept {
            return vector(_mm_add_epi32(m_value, other.m_value));
        }

        vector operator-(const vector &other) const noexcept {
            return vector(_mm_sub_epi32(m_value, other.m_value));
        }

        vector operator*(const vector &other) const noexcept {
            return vector(_mm_mullo_epi32(m_value, other.m_value));
        }

        vector &operator+=(const vector &other) noexcept {
            m_value = _mm_add_epi32(m_value, other.m_value);
            return *this;
        }

        vector &operator-=(const vector &other) noexcept {
            m_value = _mm_sub_epi32(m_value, other.m_value);
            return *this;
        }

        vector &operator*=(const vector &other) noexcept {
            m_value = _mm_mullo_epi32(m_value, other.m_value);
            return *this;
        }

        // Unary operators
        vector operator-() const noexcept requires (std::is_same_v<Tag, signed>) {
            return vector(_mm_sub_epi32(_mm_setzero_si128(), m_value));
        }

        // Bitwise operators
        vector operator&(const vector &other) const noexcept {
            return vector(_mm_and_si128(m_value, other.m_value));
        }

        vector operator|(const vector &other) const noexcept {
            return vector(_mm_or_si128(m_value, other.m_value));
        }

        vector operator^(const vector &other) const noexcept {
            return vector(_mm_xor_si128(m_value, other.m_value));
        }

        vector operator~() const noexcept {
            return vector(_mm_xor_si128(m_value, _mm_set1_epi32(-1)));
        }

        vector &operator&=(const vector &other) noexcept {
            m_value = _mm_and_si128(m_value, other.m_value);
            return *this;
        }

        vector &operator|=(const vector &other) noexcept {
            m_value = _mm_or_si128(m_value, other.m_value);
            return *this;
        }

        vector &operator^=(const vector &other) noexcept {
            m_value = _mm_xor_si128(m_value, other.m_value);
            return *this;
        }

        vector operator<<(const int32_t count) const noexcept {
            return vector(_mm_slli_epi32(m_value, count));
        }

        vector operator>>(const int32_t count) const noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                return vector(_mm_srai_epi32(m_value, count)); // Arithmetic shift
            } else {
                return vector(_mm_srli_epi32(m_value, count)); // Logical shift
            }
        }

        vector &operator<<=(const int32_t count) noexcept {
            m_value = _mm_slli_epi32(m_value, count);
            return *this;
        }

        vector &operator>>=(const int32_t count) noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                m_value = _mm_srai_epi32(m_value, count); // Arithmetic shift
            } else {
                m_value = _mm_srli_epi32(m_value, count); // Logical shift
            }
            return *this;
        }

        // Comparison operators
        [[nodiscard]] vector cmp_eq(const vector &other) const noexcept {
            return vector(_mm_cmpeq_epi32(m_value, other.m_value));
        }

        [[nodiscard]] vector cmp_ne(const vector &other) const noexcept {
            return ~cmp_eq(other);
        }

        [[nodiscard]] vector cmp_lt(const vector &other) const noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                return vector(_mm_cmplt_epi32(m_value, other.m_value));
            } else {
                // Unsigned comparison: flip sign bit
                const mask_type sign_flip = _mm_set1_epi32(0x80000000);
                const mask_type a = _mm_xor_si128(m_value, sign_flip);
                const mask_type b = _mm_xor_si128(other.m_value, sign_flip);
                return vector(_mm_cmplt_epi32(a, b));
            }
        }

        [[nodiscard]] vector cmp_le(const vector &other) const noexcept {
            return ~cmp_gt(other);
        }

        [[nodiscard]] vector cmp_gt(const vector &other) const noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                return vector(_mm_cmpgt_epi32(m_value, other.m_value));
            } else {
                // Unsigned comparison: flip sign bit
                const mask_type sign_flip = _mm_set1_epi32(0x80000000);
                const mask_type a = _mm_xor_si128(m_value, sign_flip);
                const mask_type b = _mm_xor_si128(other.m_value, sign_flip);
                return vector(_mm_cmpgt_epi32(a, b));
            }
        }

        [[nodiscard]] vector cmp_ge(const vector &other) const noexcept {
            return ~cmp_lt(other);
        }

        vector operator==(const vector &other) const noexcept { return cmp_eq(other); }
        vector operator!=(const vector &other) const noexcept { return cmp_ne(other); }
        vector operator<(const vector &other) const noexcept { return cmp_lt(other); }
        vector operator<=(const vector &other) const noexcept { return cmp_le(other); }
        vector operator>(const vector &other) const noexcept { return cmp_gt(other); }
        vector operator>=(const vector &other) const noexcept { return cmp_ge(other); }

        // Static functions
        static vector blend(const vector &a, const vector &b, const vector &mask) noexcept {
            return vector(_mm_blendv_epi8(b.m_value, a.m_value, mask.m_value));
        }

        static vector select(const vector &mask, const vector &a, const vector &b) noexcept {
            return blend(a, b, mask);
        }

        static vector min(const vector &a, const vector &b) noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                return vector(_mm_min_epi32(a.m_value, b.m_value));
            } else {
                return vector(_mm_min_epu32(a.m_value, b.m_value));
            }
        }

        static vector max(const vector &a, const vector &b) noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                return vector(_mm_max_epi32(a.m_value, b.m_value));
            } else {
                return vector(_mm_max_epu32(a.m_value, b.m_value));
            }
        }

        static vector abs(const vector &v) noexcept requires (std::is_same_v<Tag, signed>) {
            return vector(_mm_abs_epi32(v.m_value));
        }

        static vector zero() noexcept {
            return vector(_mm_setzero_si128());
        }

        static vector set1(const scalar_type val) noexcept {
            return vector(_mm_set1_epi32(static_cast<int32_t>(val)));
        }

        static constexpr size_t size() noexcept { return vector_traits<mask_type>::lanes; }

        // Horizontal operations
        [[nodiscard]] scalar_type horizontal_sum() const noexcept {
            mask_type sum = _mm_hadd_epi32(m_value, m_value);
            sum = _mm_hadd_epi32(sum, sum);
            return static_cast<scalar_type>(_mm_cvtsi128_si32(sum));
        }

        [[nodiscard]] scalar_type horizontal_min() const noexcept {
            alignas(16) scalar_type temp[4];
            store_aligned(temp);
            scalar_type result = temp[0];
            for (int i = 1; i < 4; ++i) {
                if (temp[i] < result) result = temp[i];
            }
            return result;
        }

        [[nodiscard]] scalar_type horizontal_max() const noexcept {
            alignas(16) scalar_type temp[4];
            store_aligned(temp);
            scalar_type result = temp[0];
            for (int i = 1; i < 4; ++i) {
                if (temp[i] > result) result = temp[i];
            }
            return result;
        }
    };


    // ============================================================================
    // SSE Float Vector (__m128)
    // ============================================================================

    template<>
    class vector<__m128> {
    public:
        using scalar_type = vector_traits<__m128>::scalar_type;
        using mask_type = vector_traits<__m128>::mask_type;

    private:
        mask_type m_value{};

    public:
        // Constructors
        vector() noexcept = default;

        explicit vector(const mask_type val) noexcept : m_value(val) {
        }

        explicit vector(const scalar_type *data) noexcept { load(data); }

        // Access
        mask_type &get() noexcept { return m_value; }
        [[nodiscard]] const mask_type &get() const noexcept { return m_value; }

        // Load/Store
        void load(const scalar_type *data) noexcept {
            m_value = _mm_loadu_ps(data);
        }

        void store(scalar_type *data) const noexcept {
            _mm_storeu_ps(data, m_value);
        }

        void load_aligned(const scalar_type *data) noexcept {
            m_value = _mm_load_ps(data);
        }

        void store_aligned(scalar_type *data) const noexcept {
            _mm_store_ps(data, m_value);
        }

        // Arithmetic operators
        vector operator+(const vector &other) const noexcept {
            return vector(_mm_add_ps(m_value, other.m_value));
        }

        vector operator-(const vector &other) const noexcept {
            return vector(_mm_sub_ps(m_value, other.m_value));
        }

        vector operator*(const vector &other) const noexcept {
            return vector(_mm_mul_ps(m_value, other.m_value));
        }

        vector operator/(const vector &other) const noexcept {
            return vector(_mm_div_ps(m_value, other.m_value));
        }

        vector &operator+=(const vector &other) noexcept {
            m_value = _mm_add_ps(m_value, other.m_value);
            return *this;
        }

        vector &operator-=(const vector &other) noexcept {
            m_value = _mm_sub_ps(m_value, other.m_value);
            return *this;
        }

        vector &operator*=(const vector &other) noexcept {
            m_value = _mm_mul_ps(m_value, other.m_value);
            return *this;
        }

        vector &operator/=(const vector &other) noexcept {
            m_value = _mm_div_ps(m_value, other.m_value);
            return *this;
        }

        // Unary operators
        vector operator-() const noexcept {
            return vector(_mm_xor_ps(m_value, _mm_set1_ps(-0.0f)));
        }

        // Bitwise operators (reinterpreted as integer operations)
        vector operator&(const vector &other) const noexcept {
            return vector(_mm_and_ps(m_value, other.m_value));
        }

        vector operator|(const vector &other) const noexcept {
            return vector(_mm_or_ps(m_value, other.m_value));
        }

        vector operator^(const vector &other) const noexcept {
            return vector(_mm_xor_ps(m_value, other.m_value));
        }

        vector operator~() const noexcept {
            return vector(_mm_xor_ps(m_value, _mm_castsi128_ps(_mm_set1_epi32(-1))));
        }

        vector &operator&=(const vector &other) noexcept {
            m_value = _mm_and_ps(m_value, other.m_value);
            return *this;
        }

        vector &operator|=(const vector &other) noexcept {
            m_value = _mm_or_ps(m_value, other.m_value);
            return *this;
        }

        vector &operator^=(const vector &other) noexcept {
            m_value = _mm_xor_ps(m_value, other.m_value);
            return *this;
        }

        // Comparison operators
        [[nodiscard]] vector cmp_eq(const vector &other) const noexcept {
            return vector(_mm_cmpeq_ps(m_value, other.m_value));
        }

        [[nodiscard]] vector cmp_ne(const vector &other) const noexcept {
            return vector(_mm_cmpneq_ps(m_value, other.m_value));
        }

        [[nodiscard]] vector cmp_lt(const vector &other) const noexcept {
            return vector(_mm_cmplt_ps(m_value, other.m_value));
        }

        [[nodiscard]] vector cmp_le(const vector &other) const noexcept {
            return vector(_mm_cmple_ps(m_value, other.m_value));
        }

        [[nodiscard]] vector cmp_gt(const vector &other) const noexcept {
            return vector(_mm_cmpgt_ps(m_value, other.m_value));
        }

        [[nodiscard]] vector cmp_ge(const vector &other) const noexcept {
            return vector(_mm_cmpge_ps(m_value, other.m_value));
        }

        vector operator==(const vector &other) const noexcept { return cmp_eq(other); }
        vector operator!=(const vector &other) const noexcept { return cmp_ne(other); }
        vector operator<(const vector &other) const noexcept { return cmp_lt(other); }
        vector operator<=(const vector &other) const noexcept { return cmp_le(other); }
        vector operator>(const vector &other) const noexcept { return cmp_gt(other); }
        vector operator>=(const vector &other) const noexcept { return cmp_ge(other); }

        // Static functions
        static vector blend(const vector &a, const vector &b, const vector &mask) noexcept {
            return vector(_mm_blendv_ps(b.m_value, a.m_value, mask.m_value));
        }

        static vector select(const vector &mask, const vector &a, const vector &b) noexcept {
            return blend(a, b, mask);
        }

        static vector min(const vector &a, const vector &b) noexcept {
            return vector(_mm_min_ps(a.m_value, b.m_value));
        }

        static vector max(const vector &a, const vector &b) noexcept {
            return vector(_mm_max_ps(a.m_value, b.m_value));
        }

        static vector sqrt(const vector &v) noexcept {
            return vector(_mm_sqrt_ps(v.m_value));
        }

        static vector abs(const vector &v) noexcept {
            const mask_type sign_mask = _mm_set1_ps(-0.0f);
            return vector(_mm_andnot_ps(sign_mask, v.m_value));
        }

        static vector fma(const vector &a, const vector &b, const vector &c) noexcept {
#ifdef _SIMD_FMA
            return vector(_mm_fmadd_ps(a.m_value, b.m_value, c.m_value));
#else
            return vector(_mm_add_ps(_mm_mul_ps(a.m_value, b.m_value), c.m_value));
#endif
        }

        static vector zero() noexcept {
            return vector(_mm_setzero_ps());
        }

        static vector set1(const scalar_type val) noexcept {
            return vector(_mm_set1_ps(val));
        }

        static constexpr size_t size() noexcept { return vector_traits<mask_type>::lanes; }

        // Horizontal operations
        [[nodiscard]] scalar_type horizontal_sum() const noexcept {
            mask_type shuf = _mm_movehdup_ps(m_value);
            mask_type sums = _mm_add_ps(m_value, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
        }

        [[nodiscard]] scalar_type horizontal_min() const noexcept {
            alignas(16) scalar_type temp[4];
            store_aligned(temp);
            scalar_type result = temp[0];
            for (int i = 1; i < 4; ++i) {
                if (temp[i] < result) result = temp[i];
            }
            return result;
        }

        [[nodiscard]] scalar_type horizontal_max() const noexcept {
            alignas(16) scalar_type temp[4];
            store_aligned(temp);
            scalar_type result = temp[0];
            for (int i = 1; i < 4; ++i) {
                if (temp[i] > result) result = temp[i];
            }
            return result;
        }
    };

    // ============================================================================
    // SSE Double Vector (__m128d)
    // ============================================================================

    template<>
    class vector<__m128d> {
    public:
        using scalar_type = vector_traits<__m128d>::scalar_type;
        using mask_type = vector_traits<__m128d>::mask_type;

    private:
        mask_type m_value{};

    public:
        // Constructors
        vector() noexcept = default;

        explicit vector(const mask_type val) noexcept : m_value(val) {
        }

        explicit vector(const scalar_type *data) noexcept { load(data); }

        // Access
        mask_type &get() noexcept { return m_value; }
        [[nodiscard]] const mask_type &get() const noexcept { return m_value; }

        // Load/Store
        void load(const scalar_type *data) noexcept {
            m_value = _mm_loadu_pd(data);
        }

        void store(scalar_type *data) const noexcept {
            _mm_storeu_pd(data, m_value);
        }

        void load_aligned(const scalar_type *data) noexcept {
            m_value = _mm_load_pd(data);
        }

        void store_aligned(scalar_type *data) const noexcept {
            _mm_store_pd(data, m_value);
        }

        // Arithmetic operators
        vector operator+(const vector &other) const noexcept {
            return vector(_mm_add_pd(m_value, other.m_value));
        }

        vector operator-(const vector &other) const noexcept {
            return vector(_mm_sub_pd(m_value, other.m_value));
        }

        vector operator*(const vector &other) const noexcept {
            return vector(_mm_mul_pd(m_value, other.m_value));
        }

        vector operator/(const vector &other) const noexcept {
            return vector(_mm_div_pd(m_value, other.m_value));
        }

        vector &operator+=(const vector &other) noexcept {
            m_value = _mm_add_pd(m_value, other.m_value);
            return *this;
        }

        vector &operator-=(const vector &other) noexcept {
            m_value = _mm_sub_pd(m_value, other.m_value);
            return *this;
        }

        vector &operator*=(const vector &other) noexcept {
            m_value = _mm_mul_pd(m_value, other.m_value);
            return *this;
        }

        vector &operator/=(const vector &other) noexcept {
            m_value = _mm_div_pd(m_value, other.m_value);
            return *this;
        }

        // Unary operators
        vector operator-() const noexcept {
            return vector(_mm_xor_pd(m_value, _mm_set1_pd(-0.0)));
        }

        // Bitwise operators
        vector operator&(const vector &other) const noexcept {
            return vector(_mm_and_pd(m_value, other.m_value));
        }

        vector operator|(const vector &other) const noexcept {
            return vector(_mm_or_pd(m_value, other.m_value));
        }

        vector operator^(const vector &other) const noexcept {
            return vector(_mm_xor_pd(m_value, other.m_value));
        }

        vector operator~() const noexcept {
            return vector(_mm_xor_pd(m_value, _mm_castsi128_pd(_mm_set1_epi32(-1))));
        }

        vector &operator&=(const vector &other) noexcept {
            m_value = _mm_and_pd(m_value, other.m_value);
            return *this;
        }

        vector &operator|=(const vector &other) noexcept {
            m_value = _mm_or_pd(m_value, other.m_value);
            return *this;
        }

        vector &operator^=(const vector &other) noexcept {
            m_value = _mm_xor_pd(m_value, other.m_value);
            return *this;
        }

        // Comparison operators
        [[nodiscard]] vector cmp_eq(const vector &other) const noexcept {
            return vector(_mm_cmpeq_pd(m_value, other.m_value));
        }

        [[nodiscard]] vector cmp_ne(const vector &other) const noexcept {
            return vector(_mm_cmpneq_pd(m_value, other.m_value));
        }

        [[nodiscard]] vector cmp_lt(const vector &other) const noexcept {
            return vector(_mm_cmplt_pd(m_value, other.m_value));
        }

        [[nodiscard]] vector cmp_le(const vector &other) const noexcept {
            return vector(_mm_cmple_pd(m_value, other.m_value));
        }

        [[nodiscard]] vector cmp_gt(const vector &other) const noexcept {
            return vector(_mm_cmpgt_pd(m_value, other.m_value));
        }

        [[nodiscard]] vector cmp_ge(const vector &other) const noexcept {
            return vector(_mm_cmpge_pd(m_value, other.m_value));
        }

        vector operator==(const vector &other) const noexcept { return cmp_eq(other); }
        vector operator!=(const vector &other) const noexcept { return cmp_ne(other); }
        vector operator<(const vector &other) const noexcept { return cmp_lt(other); }
        vector operator<=(const vector &other) const noexcept { return cmp_le(other); }
        vector operator>(const vector &other) const noexcept { return cmp_gt(other); }
        vector operator>=(const vector &other) const noexcept { return cmp_ge(other); }

        // Static functions
        static vector blend(const vector &a, const vector &b, const vector &mask) noexcept {
            return vector(_mm_blendv_pd(b.m_value, a.m_value, mask.m_value));
        }

        static vector select(const vector &mask, const vector &a, const vector &b) noexcept {
            return blend(a, b, mask);
        }

        static vector min(const vector &a, const vector &b) noexcept {
            return vector(_mm_min_pd(a.m_value, b.m_value));
        }

        static vector max(const vector &a, const vector &b) noexcept {
            return vector(_mm_max_pd(a.m_value, b.m_value));
        }

        static vector sqrt(const vector &v) noexcept {
            return vector(_mm_sqrt_pd(v.m_value));
        }

        static vector abs(const vector &v) noexcept {
            const mask_type sign_mask = _mm_set1_pd(-0.0);
            return vector(_mm_andnot_pd(sign_mask, v.m_value));
        }

        static vector fma(const vector &a, const vector &b, const vector &c) noexcept {
#ifdef _SIMD_FMA
            return vector(_mm_fmadd_pd(a.m_value, b.m_value, c.m_value));
#else
            return vector(_mm_add_pd(_mm_mul_pd(a.m_value, b.m_value), c.m_value));
#endif
        }

        static vector zero() noexcept {
            return vector(_mm_setzero_pd());
        }

        static vector set1(const scalar_type val) noexcept {
            return vector(_mm_set1_pd(val));
        }

        static constexpr size_t size() noexcept { return vector_traits<mask_type>::lanes; }

        // Horizontal operations
        [[nodiscard]] scalar_type horizontal_sum() const noexcept {
            const mask_type shuf = _mm_shuffle_pd(m_value, m_value, 1);
            const mask_type sums = _mm_add_sd(m_value, shuf);
            return _mm_cvtsd_f64(sums);
        }

        [[nodiscard]] scalar_type horizontal_min() const noexcept {
            alignas(16) scalar_type temp[2];
            store_aligned(temp);
            return temp[0] < temp[1] ? temp[0] : temp[1];
        }

        [[nodiscard]] scalar_type horizontal_max() const noexcept {
            alignas(16) scalar_type temp[2];
            store_aligned(temp);
            return temp[0] > temp[1] ? temp[0] : temp[1];
        }
    };

    // ============================================================================
    // AVX Integer Vector (__m256i)
    // ============================================================================

    template<typename Tag>
    class vector<__m256i, Tag> {
        static_assert(std::is_same_v<Tag, signed> || std::is_same_v<Tag, unsigned>,
                      "Tag must be signed or unsigned");

    public:
        using scalar_type = vector_traits<__m256i, Tag>::scalar_type;
        using mask_type = vector_traits<__m256i, Tag>::mask_type;

    private:
        mask_type m_value{};

    public:
        // Constructors
        vector() noexcept = default;

        explicit vector(const mask_type val) noexcept : m_value(val) {
        }

        explicit vector(const scalar_type *data) noexcept { load(data); }

        // Access
        mask_type &get() noexcept { return m_value; }
        [[nodiscard]] const mask_type &get() const noexcept { return m_value; }

        // Load/Store
        void load(const scalar_type *data) noexcept {
            m_value = _mm256_loadu_si256(reinterpret_cast<const mask_type *>(data));
        }

        void store(scalar_type *data) const noexcept {
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(data), m_value);
        }

        void load_aligned(const scalar_type *data) noexcept {
            m_value = _mm256_load_si256(reinterpret_cast<const __m256i *>(data));
        }

        void store_aligned(scalar_type *data) const noexcept {
            _mm256_store_si256(reinterpret_cast<__m256i *>(data), m_value);
        }

        // Arithmetic operators
        vector operator+(const vector &other) const noexcept {
            return vector(_mm256_add_epi32(m_value, other.m_value));
        }

        vector operator-(const vector &other) const noexcept {
            return vector(_mm256_sub_epi32(m_value, other.m_value));
        }

        vector operator*(const vector &other) const noexcept {
            return vector(_mm256_mullo_epi32(m_value, other.m_value));
        }

        vector &operator+=(const vector &other) noexcept {
            m_value = _mm256_add_epi32(m_value, other.m_value);
            return *this;
        }

        vector &operator-=(const vector &other) noexcept {
            m_value = _mm256_sub_epi32(m_value, other.m_value);
            return *this;
        }

        vector &operator*=(const vector &other) noexcept {
            m_value = _mm256_mullo_epi32(m_value, other.m_value);
            return *this;
        }

        // Unary operators
        vector operator-() const noexcept requires (std::is_same_v<Tag, signed>) {
            return vector(_mm256_sub_epi32(_mm256_setzero_si256(), m_value));
        }

        // Bitwise operators
        vector operator&(const vector &other) const noexcept {
            return vector(_mm256_and_si256(m_value, other.m_value));
        }

        vector operator|(const vector &other) const noexcept {
            return vector(_mm256_or_si256(m_value, other.m_value));
        }

        vector operator^(const vector &other) const noexcept {
            return vector(_mm256_xor_si256(m_value, other.m_value));
        }

        vector operator~() const noexcept {
            return vector(_mm256_xor_si256(m_value, _mm256_set1_epi32(-1)));
        }

        vector &operator&=(const vector &other) noexcept {
            m_value = _mm256_and_si256(m_value, other.m_value);
            return *this;
        }

        vector &operator|=(const vector &other) noexcept {
            m_value = _mm256_or_si256(m_value, other.m_value);
            return *this;
        }

        vector &operator^=(const vector &other) noexcept {
            m_value = _mm256_xor_si256(m_value, other.m_value);
            return *this;
        }

        vector operator<<(const int32_t count) const noexcept {
            return vector(_mm256_slli_epi32(m_value, count));
        }

        vector operator>>(const int32_t count) const noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                return vector(_mm256_srai_epi32(m_value, count)); // Arithmetic shift
            } else {
                return vector(_mm256_srli_epi32(m_value, count)); // Logical shift
            }
        }

        vector &operator<<=(const int32_t count) noexcept {
            m_value = _mm256_slli_epi32(m_value, count);
            return *this;
        }

        vector &operator>>=(const int32_t count) noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                m_value = _mm256_srai_epi32(m_value, count); // Arithmetic shift
            } else {
                m_value = _mm256_srli_epi32(m_value, count); // Logical shift
            }
            return *this;
        }

        // Comparison operators
        [[nodiscard]] vector cmp_eq(const vector &other) const noexcept {
            return vector(_mm256_cmpeq_epi32(m_value, other.m_value));
        }

        [[nodiscard]] vector cmp_ne(const vector &other) const noexcept {
            return ~cmp_eq(other);
        }

        [[nodiscard]] vector cmp_lt(const vector &other) const noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                return vector(_mm256_cmpgt_epi32(other.m_value, m_value));
            } else {
                // Unsigned comparison: flip sign bit
                const mask_type sign_flip = _mm256_set1_epi32(0x80000000);
                const mask_type a = _mm256_xor_si256(m_value, sign_flip);
                const mask_type b = _mm256_xor_si256(other.m_value, sign_flip);
                return vector(_mm256_cmpgt_epi32(b, a));
            }
        }

        [[nodiscard]] vector cmp_le(const vector &other) const noexcept {
            return ~cmp_gt(other);
        }

        [[nodiscard]] vector cmp_gt(const vector &other) const noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                return vector(_mm256_cmpgt_epi32(m_value, other.m_value));
            } else {
                // Unsigned comparison: flip sign bit
                const mask_type sign_flip = _mm256_set1_epi32(0x80000000);
                const mask_type a = _mm256_xor_si256(m_value, sign_flip);
                const mask_type b = _mm256_xor_si256(other.m_value, sign_flip);
                return vector(_mm256_cmpgt_epi32(a, b));
            }
        }

        [[nodiscard]] vector cmp_ge(const vector &other) const noexcept {
            return ~cmp_lt(other);
        }

        vector operator==(const vector &other) const noexcept { return cmp_eq(other); }
        vector operator!=(const vector &other) const noexcept { return cmp_ne(other); }
        vector operator<(const vector &other) const noexcept { return cmp_lt(other); }
        vector operator<=(const vector &other) const noexcept { return cmp_le(other); }
        vector operator>(const vector &other) const noexcept { return cmp_gt(other); }
        vector operator>=(const vector &other) const noexcept { return cmp_ge(other); }

        // Static functions
        static vector blend(const vector &a, const vector &b, const vector &mask) noexcept {
            return vector(_mm256_blendv_epi8(b.m_value, a.m_value, mask.m_value));
        }

        static vector select(const vector &mask, const vector &a, const vector &b) noexcept {
            return blend(a, b, mask);
        }

        static vector min(const vector &a, const vector &b) noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                return vector(_mm256_min_epi32(a.m_value, b.m_value));
            } else {
                return vector(_mm256_min_epu32(a.m_value, b.m_value));
            }
        }

        static vector max(const vector &a, const vector &b) noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                return vector(_mm256_max_epi32(a.m_value, b.m_value));
            } else {
                return vector(_mm256_max_epu32(a.m_value, b.m_value));
            }
        }

        static vector abs(const vector &v) noexcept requires (std::is_same_v<Tag, signed>) {
            return vector(_mm256_abs_epi32(v.m_value));
        }

        static vector zero() noexcept {
            return vector(_mm256_setzero_si256());
        }

        static vector set1(const scalar_type val) noexcept {
            return vector(_mm256_set1_epi32(static_cast<int32_t>(val)));
        }

        static constexpr size_t size() noexcept { return vector_traits<mask_type>::lanes; }

        // Horizontal operations
        [[nodiscard]] scalar_type horizontal_sum() const noexcept {
            const mask_type low = _mm256_castsi256_si128(m_value);
            const mask_type high = _mm256_extracti128_si256(m_value, 1);
            mask_type sum = _mm_add_epi32(low, high);
            sum = _mm_hadd_epi32(sum, sum);
            sum = _mm_hadd_epi32(sum, sum);
            return static_cast<scalar_type>(_mm_cvtsi128_si32(sum));
        }

        [[nodiscard]] scalar_type horizontal_min() const noexcept {
            alignas(32) scalar_type temp[8];
            store_aligned(temp);
            scalar_type result = temp[0];
            for (int i = 1; i < 8; ++i) {
                if (temp[i] < result) result = temp[i];
            }
            return result;
        }

        [[nodiscard]] scalar_type horizontal_max() const noexcept {
            alignas(32) scalar_type temp[8];
            store_aligned(temp);
            scalar_type result = temp[0];
            for (int i = 1; i < 8; ++i) {
                if (temp[i] > result) result = temp[i];
            }
            return result;
        }
    };

    // ============================================================================
    // AVX Float Vector (__m256)
    // ============================================================================

    template<>
    class vector<__m256> {
    public:
        using scalar_type = vector_traits<__m256>::scalar_type;
        using mask_type = vector_traits<__m256>::mask_type;

    private:
        mask_type m_value{};

    public:
        // Constructors
        vector() noexcept = default;

        explicit vector(const mask_type val) noexcept : m_value(val) {
        }

        explicit vector(const scalar_type *data) noexcept { load(data); }

        // Access
        mask_type &get() noexcept { return m_value; }
        [[nodiscard]] const mask_type &get() const noexcept { return m_value; }

        // Load/Store
        void load(const scalar_type *data) noexcept {
            m_value = _mm256_loadu_ps(data);
        }

        void store(scalar_type *data) const noexcept {
            _mm256_storeu_ps(data, m_value);
        }

        void load_aligned(const scalar_type *data) noexcept {
            m_value = _mm256_load_ps(data);
        }

        void store_aligned(scalar_type *data) const noexcept {
            _mm256_store_ps(data, m_value);
        }

        // Arithmetic operators
        vector operator+(const vector &other) const noexcept {
            return vector(_mm256_add_ps(m_value, other.m_value));
        }

        vector operator-(const vector &other) const noexcept {
            return vector(_mm256_sub_ps(m_value, other.m_value));
        }

        vector operator*(const vector &other) const noexcept {
            return vector(_mm256_mul_ps(m_value, other.m_value));
        }

        vector operator/(const vector &other) const noexcept {
            return vector(_mm256_div_ps(m_value, other.m_value));
        }

        vector &operator+=(const vector &other) noexcept {
            m_value = _mm256_add_ps(m_value, other.m_value);
            return *this;
        }

        vector &operator-=(const vector &other) noexcept {
            m_value = _mm256_sub_ps(m_value, other.m_value);
            return *this;
        }

        vector &operator*=(const vector &other) noexcept {
            m_value = _mm256_mul_ps(m_value, other.m_value);
            return *this;
        }

        vector &operator/=(const vector &other) noexcept {
            m_value = _mm256_div_ps(m_value, other.m_value);
            return *this;
        }

        // Unary operators
        vector operator-() const noexcept {
            return vector(_mm256_xor_ps(m_value, _mm256_set1_ps(-0.0f)));
        }

        // Bitwise operators
        vector operator&(const vector &other) const noexcept {
            return vector(_mm256_and_ps(m_value, other.m_value));
        }

        vector operator|(const vector &other) const noexcept {
            return vector(_mm256_or_ps(m_value, other.m_value));
        }

        vector operator^(const vector &other) const noexcept {
            return vector(_mm256_xor_ps(m_value, other.m_value));
        }

        vector operator~() const noexcept {
            return vector(_mm256_xor_ps(m_value, _mm256_castsi256_ps(_mm256_set1_epi32(-1))));
        }

        vector &operator&=(const vector &other) noexcept {
            m_value = _mm256_and_ps(m_value, other.m_value);
            return *this;
        }

        vector &operator|=(const vector &other) noexcept {
            m_value = _mm256_or_ps(m_value, other.m_value);
            return *this;
        }

        vector &operator^=(const vector &other) noexcept {
            m_value = _mm256_xor_ps(m_value, other.m_value);
            return *this;
        }

        // Comparison operators
        [[nodiscard]] vector cmp_eq(const vector &other) const noexcept {
            return vector(_mm256_cmp_ps(m_value, other.m_value, _CMP_EQ_OQ));
        }

        [[nodiscard]] vector cmp_ne(const vector &other) const noexcept {
            return vector(_mm256_cmp_ps(m_value, other.m_value, _CMP_NEQ_OQ));
        }

        [[nodiscard]] vector cmp_lt(const vector &other) const noexcept {
            return vector(_mm256_cmp_ps(m_value, other.m_value, _CMP_LT_OQ));
        }

        [[nodiscard]] vector cmp_le(const vector &other) const noexcept {
            return vector(_mm256_cmp_ps(m_value, other.m_value, _CMP_LE_OQ));
        }

        [[nodiscard]] vector cmp_gt(const vector &other) const noexcept {
            return vector(_mm256_cmp_ps(m_value, other.m_value, _CMP_GT_OQ));
        }

        [[nodiscard]] vector cmp_ge(const vector &other) const noexcept {
            return vector(_mm256_cmp_ps(m_value, other.m_value, _CMP_GE_OQ));
        }

        vector operator==(const vector &other) const noexcept { return cmp_eq(other); }
        vector operator!=(const vector &other) const noexcept { return cmp_ne(other); }
        vector operator<(const vector &other) const noexcept { return cmp_lt(other); }
        vector operator<=(const vector &other) const noexcept { return cmp_le(other); }
        vector operator>(const vector &other) const noexcept { return cmp_gt(other); }
        vector operator>=(const vector &other) const noexcept { return cmp_ge(other); }

        // Static functions
        static vector blend(const vector &a, const vector &b, const vector &mask) noexcept {
            return vector(_mm256_blendv_ps(b.m_value, a.m_value, mask.m_value));
        }

        static vector select(const vector &mask, const vector &a, const vector &b) noexcept {
            return blend(a, b, mask);
        }

        static vector min(const vector &a, const vector &b) noexcept {
            return vector(_mm256_min_ps(a.m_value, b.m_value));
        }

        static vector max(const vector &a, const vector &b) noexcept {
            return vector(_mm256_max_ps(a.m_value, b.m_value));
        }

        static vector sqrt(const vector &v) noexcept {
            return vector(_mm256_sqrt_ps(v.m_value));
        }

        static vector abs(const vector &v) noexcept {
            const mask_type sign_mask = _mm256_set1_ps(-0.0f);
            return vector(_mm256_andnot_ps(sign_mask, v.m_value));
        }

        static vector fma(const vector &a, const vector &b, const vector &c) noexcept {
#ifdef _SIMD_FMA
            return vector(_mm256_fmadd_ps(a.m_value, b.m_value, c.m_value));
#else
            return vector(_mm256_add_ps(_mm256_mul_ps(a.m_value, b.m_value), c.m_value));
#endif
        }

        static vector zero() noexcept {
            return vector(_mm256_setzero_ps());
        }

        static vector set1(const scalar_type val) noexcept {
            return vector(_mm256_set1_ps(val));
        }

        static constexpr size_t size() noexcept { return vector_traits<mask_type>::lanes; }

        // Horizontal operations
        [[nodiscard]] scalar_type horizontal_sum() const noexcept {
            const __m128 low = _mm256_castps256_ps128(m_value);
            const __m128 high = _mm256_extractf128_ps(m_value, 1);
            const __m128 sum = _mm_add_ps(low, high);
            __m128 shuf = _mm_movehdup_ps(sum);
            __m128 sums = _mm_add_ps(sum, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
        }

        [[nodiscard]] scalar_type horizontal_min() const noexcept {
            alignas(32) scalar_type temp[8];
            store_aligned(temp);
            scalar_type result = temp[0];
            for (int i = 1; i < 8; ++i) {
                if (temp[i] < result) result = temp[i];
            }
            return result;
        }

        [[nodiscard]] scalar_type horizontal_max() const noexcept {
            alignas(32) scalar_type temp[8];
            store_aligned(temp);
            scalar_type result = temp[0];
            for (int i = 1; i < 8; ++i) {
                if (temp[i] > result) result = temp[i];
            }
            return result;
        }
    };

    // ============================================================================
    // AVX Double Vector (__m256d)
    // ============================================================================

    template<>
    class vector<__m256d> {
    public:
        using scalar_type = vector_traits<__m256d>::scalar_type;
        using mask_type = vector_traits<__m256d>::mask_type;

    private:
        mask_type m_value{};

    public:
        // Constructors
        vector() noexcept = default;

        explicit vector(const mask_type val) noexcept : m_value(val) {
        }

        explicit vector(const scalar_type *data) noexcept { load(data); }

        // Access
        mask_type &get() noexcept { return m_value; }
        [[nodiscard]] const mask_type &get() const noexcept { return m_value; }

        // Load/Store
        void load(const scalar_type *data) noexcept {
            m_value = _mm256_loadu_pd(data);
        }

        void store(scalar_type *data) const noexcept {
            _mm256_storeu_pd(data, m_value);
        }

        void load_aligned(const scalar_type *data) noexcept {
            m_value = _mm256_load_pd(data);
        }

        void store_aligned(scalar_type *data) const noexcept {
            _mm256_store_pd(data, m_value);
        }

        // Arithmetic operators
        vector operator+(const vector &other) const noexcept {
            return vector(_mm256_add_pd(m_value, other.m_value));
        }

        vector operator-(const vector &other) const noexcept {
            return vector(_mm256_sub_pd(m_value, other.m_value));
        }

        vector operator*(const vector &other) const noexcept {
            return vector(_mm256_mul_pd(m_value, other.m_value));
        }

        vector operator/(const vector &other) const noexcept {
            return vector(_mm256_div_pd(m_value, other.m_value));
        }

        vector &operator+=(const vector &other) noexcept {
            m_value = _mm256_add_pd(m_value, other.m_value);
            return *this;
        }

        vector &operator-=(const vector &other) noexcept {
            m_value = _mm256_sub_pd(m_value, other.m_value);
            return *this;
        }

        vector &operator*=(const vector &other) noexcept {
            m_value = _mm256_mul_pd(m_value, other.m_value);
            return *this;
        }

        vector &operator/=(const vector &other) noexcept {
            m_value = _mm256_div_pd(m_value, other.m_value);
            return *this;
        }

        // Unary operators
        vector operator-() const noexcept {
            return vector(_mm256_xor_pd(m_value, _mm256_set1_pd(-0.0)));
        }

        // Bitwise operators
        vector operator&(const vector &other) const noexcept {
            return vector(_mm256_and_pd(m_value, other.m_value));
        }

        vector operator|(const vector &other) const noexcept {
            return vector(_mm256_or_pd(m_value, other.m_value));
        }

        vector operator^(const vector &other) const noexcept {
            return vector(_mm256_xor_pd(m_value, other.m_value));
        }

        vector operator~() const noexcept {
            return vector(_mm256_xor_pd(m_value, _mm256_castsi256_pd(_mm256_set1_epi32(-1))));
        }

        vector &operator&=(const vector &other) noexcept {
            m_value = _mm256_and_pd(m_value, other.m_value);
            return *this;
        }

        vector &operator|=(const vector &other) noexcept {
            m_value = _mm256_or_pd(m_value, other.m_value);
            return *this;
        }

        vector &operator^=(const vector &other) noexcept {
            m_value = _mm256_xor_pd(m_value, other.m_value);
            return *this;
        }

        // Comparison operators
        [[nodiscard]] vector cmp_eq(const vector &other) const noexcept {
            return vector(_mm256_cmp_pd(m_value, other.m_value, _CMP_EQ_OQ));
        }

        [[nodiscard]] vector cmp_ne(const vector &other) const noexcept {
            return vector(_mm256_cmp_pd(m_value, other.m_value, _CMP_NEQ_OQ));
        }

        [[nodiscard]] vector cmp_lt(const vector &other) const noexcept {
            return vector(_mm256_cmp_pd(m_value, other.m_value, _CMP_LT_OQ));
        }

        [[nodiscard]] vector cmp_le(const vector &other) const noexcept {
            return vector(_mm256_cmp_pd(m_value, other.m_value, _CMP_LE_OQ));
        }

        [[nodiscard]] vector cmp_gt(const vector &other) const noexcept {
            return vector(_mm256_cmp_pd(m_value, other.m_value, _CMP_GT_OQ));
        }

        [[nodiscard]] vector cmp_ge(const vector &other) const noexcept {
            return vector(_mm256_cmp_pd(m_value, other.m_value, _CMP_GE_OQ));
        }

        vector operator==(const vector &other) const noexcept { return cmp_eq(other); }
        vector operator!=(const vector &other) const noexcept { return cmp_ne(other); }
        vector operator<(const vector &other) const noexcept { return cmp_lt(other); }
        vector operator<=(const vector &other) const noexcept { return cmp_le(other); }
        vector operator>(const vector &other) const noexcept { return cmp_gt(other); }
        vector operator>=(const vector &other) const noexcept { return cmp_ge(other); }

        // Static functions
        static vector blend(const vector &a, const vector &b, const vector &mask) noexcept {
            return vector(_mm256_blendv_pd(b.m_value, a.m_value, mask.m_value));
        }

        static vector select(const vector &mask, const vector &a, const vector &b) noexcept {
            return blend(a, b, mask);
        }

        static vector min(const vector &a, const vector &b) noexcept {
            return vector(_mm256_min_pd(a.m_value, b.m_value));
        }

        static vector max(const vector &a, const vector &b) noexcept {
            return vector(_mm256_max_pd(a.m_value, b.m_value));
        }

        static vector sqrt(const vector &v) noexcept {
            return vector(_mm256_sqrt_pd(v.m_value));
        }

        static vector abs(const vector &v) noexcept {
            const mask_type sign_mask = _mm256_set1_pd(-0.0);
            return vector(_mm256_andnot_pd(sign_mask, v.m_value));
        }

        static vector fma(const vector &a, const vector &b, const vector &c) noexcept {
#ifdef _SIMD_FMA
            return vector(_mm256_fmadd_pd(a.m_value, b.m_value, c.m_value));
#else
            return vector(_mm256_add_pd(_mm256_mul_pd(a.m_value, b.m_value), c.m_value));
#endif
        }

        static vector zero() noexcept {
            return vector(_mm256_setzero_pd());
        }

        static vector set1(const scalar_type val) noexcept {
            return vector(_mm256_set1_pd(val));
        }

        static constexpr size_t size() noexcept { return vector_traits<mask_type>::lanes; }

        // Horizontal operations
        [[nodiscard]] scalar_type horizontal_sum() const noexcept {
            const __m128d low = _mm256_castpd256_pd128(m_value);
            const __m128d high = _mm256_extractf128_pd(m_value, 1);
            __m128d sum = _mm_add_pd(low, high);
            const __m128d shuf = _mm_shuffle_pd(sum, sum, 1);
            sum = _mm_add_sd(sum, shuf);
            return _mm_cvtsd_f64(sum);
        }

        [[nodiscard]] scalar_type horizontal_min() const noexcept {
            alignas(32) scalar_type temp[4];
            store_aligned(temp);
            scalar_type result = temp[0];
            for (int i = 1; i < 4; ++i) {
                if (temp[i] < result) result = temp[i];
            }
            return result;
        }

        [[nodiscard]] scalar_type horizontal_max() const noexcept {
            alignas(32) scalar_type temp[4];
            store_aligned(temp);
            scalar_type result = temp[0];
            for (int i = 1; i < 4; ++i) {
                if (temp[i] > result) result = temp[i];
            }
            return result;
        }
    };

    // ============================================================================
    // AVX-512 Integer Vector (__m512i)
    // ============================================================================

    template<typename Tag>
    class vector<__m512i, Tag> {
        static_assert(std::is_same_v<Tag, signed> || std::is_same_v<Tag, unsigned>,
                      "Tag must be signed or unsigned");

    public:
        using scalar_type = vector_traits<__m512i, Tag>::scalar_type;
        using mask_type = vector_traits<__m512i, Tag>::mask_type;

    private:
        mask_type m_value{};

    public:
        // Constructors
        vector() noexcept = default;

        explicit vector(const mask_type val) noexcept : m_value(val) {
        }

        explicit vector(const scalar_type *data) noexcept { load(data); }

        // Access
        mask_type &get() noexcept { return m_value; }
        [[nodiscard]] const mask_type &get() const noexcept { return m_value; }

        // Load/Store
        void load(const scalar_type *data) noexcept {
            m_value = _mm512_loadu_si512(data);
        }

        void store(scalar_type *data) const noexcept {
            _mm512_storeu_si512(data, m_value);
        }

        void load_aligned(const scalar_type *data) noexcept {
            m_value = _mm512_load_si512(data);
        }

        void store_aligned(scalar_type *data) const noexcept {
            _mm512_store_si512(data, m_value);
        }

        // Arithmetic operators
        vector operator+(const vector &other) const noexcept {
            return vector(_mm512_add_epi32(m_value, other.m_value));
        }

        vector operator-(const vector &other) const noexcept {
            return vector(_mm512_sub_epi32(m_value, other.m_value));
        }

        vector operator*(const vector &other) const noexcept {
            return vector(_mm512_mullo_epi32(m_value, other.m_value));
        }

        vector &operator+=(const vector &other) noexcept {
            m_value = _mm512_add_epi32(m_value, other.m_value);
            return *this;
        }

        vector &operator-=(const vector &other) noexcept {
            m_value = _mm512_sub_epi32(m_value, other.m_value);
            return *this;
        }

        vector &operator*=(const vector &other) noexcept {
            m_value = _mm512_mullo_epi32(m_value, other.m_value);
            return *this;
        }

        // Unary operators
        vector operator-() const noexcept requires (std::is_same_v<Tag, signed>) {
            return vector(_mm512_sub_epi32(_mm512_setzero_si512(), m_value));
        }

        // Bitwise operators
        vector operator&(const vector &other) const noexcept {
            return vector(_mm512_and_si512(m_value, other.m_value));
        }

        vector operator|(const vector &other) const noexcept {
            return vector(_mm512_or_si512(m_value, other.m_value));
        }

        vector operator^(const vector &other) const noexcept {
            return vector(_mm512_xor_si512(m_value, other.m_value));
        }

        vector operator~() const noexcept {
            return vector(_mm512_xor_si512(m_value, _mm512_set1_epi32(-1)));
        }

        vector &operator&=(const vector &other) noexcept {
            m_value = _mm512_and_si512(m_value, other.m_value);
            return *this;
        }

        vector &operator|=(const vector &other) noexcept {
            m_value = _mm512_or_si512(m_value, other.m_value);
            return *this;
        }

        vector &operator^=(const vector &other) noexcept {
            m_value = _mm512_xor_si512(m_value, other.m_value);
            return *this;
        }

        vector operator<<(const int32_t count) const noexcept {
            return vector(_mm512_slli_epi32(m_value, count));
        }

        vector operator>>(const int32_t count) const noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                return vector(_mm512_srai_epi32(m_value, count)); // Arithmetic shift
            } else {
                return vector(_mm512_srli_epi32(m_value, count)); // Logical shift
            }
        }

        vector &operator<<=(const int32_t count) noexcept {
            m_value = _mm512_slli_epi32(m_value, count);
            return *this;
        }

        vector &operator>>=(const int32_t count) noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                m_value = _mm512_srai_epi32(m_value, count); // Arithmetic shift
            } else {
                m_value = _mm512_srli_epi32(m_value, count); // Logical shift
            }
            return *this;
        }

        // Comparison operators (AVX-512 uses mask registers)
        [[nodiscard]] vector cmp_eq(const vector &other) const noexcept {
            const __mmask16 mask = _mm512_cmpeq_epi32_mask(m_value, other.m_value);
            return vector(_mm512_movm_epi32(mask));
        }

        [[nodiscard]] vector cmp_ne(const vector &other) const noexcept {
            const __mmask16 mask = _mm512_cmpneq_epi32_mask(m_value, other.m_value);
            return vector(_mm512_movm_epi32(mask));
        }

        [[nodiscard]] vector cmp_lt(const vector &other) const noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                const __mmask16 mask = _mm512_cmplt_epi32_mask(m_value, other.m_value);
                return vector(_mm512_movm_epi32(mask));
            } else {
                const __mmask16 mask = _mm512_cmplt_epu32_mask(m_value, other.m_value);
                return vector(_mm512_movm_epi32(mask));
            }
        }

        [[nodiscard]] vector cmp_le(const vector &other) const noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                const __mmask16 mask = _mm512_cmple_epi32_mask(m_value, other.m_value);
                return vector(_mm512_movm_epi32(mask));
            } else {
                const __mmask16 mask = _mm512_cmple_epu32_mask(m_value, other.m_value);
                return vector(_mm512_movm_epi32(mask));
            }
        }

        [[nodiscard]] vector cmp_gt(const vector &other) const noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                const __mmask16 mask = _mm512_cmpgt_epi32_mask(m_value, other.m_value);
                return vector(_mm512_movm_epi32(mask));
            } else {
                const __mmask16 mask = _mm512_cmpgt_epu32_mask(m_value, other.m_value);
                return vector(_mm512_movm_epi32(mask));
            }
        }

        [[nodiscard]] vector cmp_ge(const vector &other) const noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                const __mmask16 mask = _mm512_cmpge_epi32_mask(m_value, other.m_value);
                return vector(_mm512_movm_epi32(mask));
            } else {
                const __mmask16 mask = _mm512_cmpge_epu32_mask(m_value, other.m_value);
                return vector(_mm512_movm_epi32(mask));
            }
        }

        vector operator==(const vector &other) const noexcept { return cmp_eq(other); }
        vector operator!=(const vector &other) const noexcept { return cmp_ne(other); }
        vector operator<(const vector &other) const noexcept { return cmp_lt(other); }
        vector operator<=(const vector &other) const noexcept { return cmp_le(other); }
        vector operator>(const vector &other) const noexcept { return cmp_gt(other); }
        vector operator>=(const vector &other) const noexcept { return cmp_ge(other); }

        // Static functions
        static vector blend(const vector &a, const vector &b, const vector &mask) noexcept {
            const __mmask16 k = _mm512_movepi32_mask(mask.m_value);
            return vector(_mm512_mask_blend_epi32(k, b.m_value, a.m_value));
        }

        static vector select(const vector &mask, const vector &a, const vector &b) noexcept {
            return blend(a, b, mask);
        }

        static vector min(const vector &a, const vector &b) noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                return vector(_mm512_min_epi32(a.m_value, b.m_value));
            } else {
                return vector(_mm512_min_epu32(a.m_value, b.m_value));
            }
        }

        static vector max(const vector &a, const vector &b) noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                return vector(_mm512_max_epi32(a.m_value, b.m_value));
            } else {
                return vector(_mm512_max_epu32(a.m_value, b.m_value));
            }
        }

        static vector abs(const vector &v) noexcept requires (std::is_same_v<Tag, signed>) {
            return vector(_mm512_abs_epi32(v.m_value));
        }

        static vector zero() noexcept {
            return vector(_mm512_setzero_si512());
        }

        static vector set1(const scalar_type val) noexcept {
            return vector(_mm512_set1_epi32(static_cast<int32_t>(val)));
        }

        static constexpr size_t size() noexcept { return vector_traits<mask_type>::lanes; }

        // Horizontal operations
        [[nodiscard]] scalar_type horizontal_sum() const noexcept {
            return static_cast<scalar_type>(_mm512_reduce_add_epi32(m_value));
        }

        [[nodiscard]] scalar_type horizontal_min() const noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                return static_cast<scalar_type>(_mm512_reduce_min_epi32(m_value));
            } else {
                return static_cast<scalar_type>(_mm512_reduce_min_epu32(m_value));
            }
        }

        [[nodiscard]] scalar_type horizontal_max() const noexcept {
            if constexpr (std::is_same_v<Tag, signed>) {
                return static_cast<scalar_type>(_mm512_reduce_max_epi32(m_value));
            } else {
                return static_cast<scalar_type>(_mm512_reduce_max_epu32(m_value));
            }
        }
    };

    // ============================================================================
    // AVX-512 Float Vector (__m512)
    // ============================================================================

    template<>
    class vector<__m512> {
    public:
        using scalar_type = vector_traits<__m512>::scalar_type;
        using mask_type = vector_traits<__m512>::mask_type;

    private:
        __m512 m_value{};

    public:
        // Constructors
        vector() noexcept = default;

        explicit vector(const mask_type val) noexcept : m_value(val) {
        }

        explicit vector(const scalar_type *data) noexcept { load(data); }

        // Access
        mask_type &get() noexcept { return m_value; }
        [[nodiscard]] const mask_type &get() const noexcept { return m_value; }

        // Load/Store
        void load(const scalar_type *data) noexcept {
            m_value = _mm512_loadu_ps(data);
        }

        void store(scalar_type *data) const noexcept {
            _mm512_storeu_ps(data, m_value);
        }

        void load_aligned(const scalar_type *data) noexcept {
            m_value = _mm512_load_ps(data);
        }

        void store_aligned(scalar_type *data) const noexcept {
            _mm512_store_ps(data, m_value);
        }

        // Arithmetic operators
        vector operator+(const vector &other) const noexcept {
            return vector(_mm512_add_ps(m_value, other.m_value));
        }

        vector operator-(const vector &other) const noexcept {
            return vector(_mm512_sub_ps(m_value, other.m_value));
        }

        vector operator*(const vector &other) const noexcept {
            return vector(_mm512_mul_ps(m_value, other.m_value));
        }

        vector operator/(const vector &other) const noexcept {
            return vector(_mm512_div_ps(m_value, other.m_value));
        }

        vector &operator+=(const vector &other) noexcept {
            m_value = _mm512_add_ps(m_value, other.m_value);
            return *this;
        }

        vector &operator-=(const vector &other) noexcept {
            m_value = _mm512_sub_ps(m_value, other.m_value);
            return *this;
        }

        vector &operator*=(const vector &other) noexcept {
            m_value = _mm512_mul_ps(m_value, other.m_value);
            return *this;
        }

        vector &operator/=(const vector &other) noexcept {
            m_value = _mm512_div_ps(m_value, other.m_value);
            return *this;
        }

        // Unary operators
        vector operator-() const noexcept {
            return vector(_mm512_xor_ps(m_value, _mm512_set1_ps(-0.0f)));
        }

        // Bitwise operators
        vector operator&(const vector &other) const noexcept {
            return vector(_mm512_and_ps(m_value, other.m_value));
        }

        vector operator|(const vector &other) const noexcept {
            return vector(_mm512_or_ps(m_value, other.m_value));
        }

        vector operator^(const vector &other) const noexcept {
            return vector(_mm512_xor_ps(m_value, other.m_value));
        }

        vector operator~() const noexcept {
            return vector(_mm512_xor_ps(m_value, _mm512_castsi512_ps(_mm512_set1_epi32(-1))));
        }

        vector &operator&=(const vector &other) noexcept {
            m_value = _mm512_and_ps(m_value, other.m_value);
            return *this;
        }

        vector &operator|=(const vector &other) noexcept {
            m_value = _mm512_or_ps(m_value, other.m_value);
            return *this;
        }

        vector &operator^=(const vector &other) noexcept {
            m_value = _mm512_xor_ps(m_value, other.m_value);
            return *this;
        }

        // Comparison operators
        [[nodiscard]] vector cmp_eq(const vector &other) const noexcept {
            const __mmask16 mask = _mm512_cmp_ps_mask(m_value, other.m_value, _CMP_EQ_OQ);
            return vector(_mm512_castsi512_ps(_mm512_movm_epi32(mask)));
        }

        [[nodiscard]] vector cmp_ne(const vector &other) const noexcept {
            const __mmask16 mask = _mm512_cmp_ps_mask(m_value, other.m_value, _CMP_NEQ_OQ);
            return vector(_mm512_castsi512_ps(_mm512_movm_epi32(mask)));
        }

        [[nodiscard]] vector cmp_lt(const vector &other) const noexcept {
            const __mmask16 mask = _mm512_cmp_ps_mask(m_value, other.m_value, _CMP_LT_OQ);
            return vector(_mm512_castsi512_ps(_mm512_movm_epi32(mask)));
        }

        [[nodiscard]] vector cmp_le(const vector &other) const noexcept {
            const __mmask16 mask = _mm512_cmp_ps_mask(m_value, other.m_value, _CMP_LE_OQ);
            return vector(_mm512_castsi512_ps(_mm512_movm_epi32(mask)));
        }

        [[nodiscard]] vector cmp_gt(const vector &other) const noexcept {
            const __mmask16 mask = _mm512_cmp_ps_mask(m_value, other.m_value, _CMP_GT_OQ);
            return vector(_mm512_castsi512_ps(_mm512_movm_epi32(mask)));
        }

        [[nodiscard]] vector cmp_ge(const vector &other) const noexcept {
            const __mmask16 mask = _mm512_cmp_ps_mask(m_value, other.m_value, _CMP_GE_OQ);
            return vector(_mm512_castsi512_ps(_mm512_movm_epi32(mask)));
        }

        vector operator==(const vector &other) const noexcept { return cmp_eq(other); }
        vector operator!=(const vector &other) const noexcept { return cmp_ne(other); }
        vector operator<(const vector &other) const noexcept { return cmp_lt(other); }
        vector operator<=(const vector &other) const noexcept { return cmp_le(other); }
        vector operator>(const vector &other) const noexcept { return cmp_gt(other); }
        vector operator>=(const vector &other) const noexcept { return cmp_ge(other); }

        // Static functions
        static vector blend(const vector &a, const vector &b, const vector &mask) noexcept {
            const __mmask16 k = _mm512_movepi32_mask(_mm512_castps_si512(mask.m_value));
            return vector(_mm512_mask_blend_ps(k, b.m_value, a.m_value));
        }

        static vector select(const vector &mask, const vector &a, const vector &b) noexcept {
            return blend(a, b, mask);
        }

        static vector min(const vector &a, const vector &b) noexcept {
            return vector(_mm512_min_ps(a.m_value, b.m_value));
        }

        static vector max(const vector &a, const vector &b) noexcept {
            return vector(_mm512_max_ps(a.m_value, b.m_value));
        }

        static vector sqrt(const vector &v) noexcept {
            return vector(_mm512_sqrt_ps(v.m_value));
        }

        static vector abs(const vector &v) noexcept {
            return vector(_mm512_abs_ps(v.m_value));
        }

        static vector fma(const vector &a, const vector &b, const vector &c) noexcept {
            return vector(_mm512_fmadd_ps(a.m_value, b.m_value, c.m_value));
        }

        static vector zero() noexcept {
            return vector(_mm512_setzero_ps());
        }

        static vector set1(const scalar_type val) noexcept {
            return vector(_mm512_set1_ps(val));
        }

        static constexpr size_t size() noexcept { return vector_traits<mask_type>::lanes; }

        // Horizontal operations
        [[nodiscard]] scalar_type horizontal_sum() const noexcept {
            return _mm512_reduce_add_ps(m_value);
        }

        [[nodiscard]] scalar_type horizontal_min() const noexcept {
            return _mm512_reduce_min_ps(m_value);
        }

        [[nodiscard]] scalar_type horizontal_max() const noexcept {
            return _mm512_reduce_max_ps(m_value);
        }
    };

    // ============================================================================
    // AVX-512 Double Vector (__m512d)
    // ============================================================================

    template<>
    class vector<__m512d> {
    public:
        using scalar_type = vector_traits<__m512d>::scalar_type;
        using mask_type = vector_traits<__m512d>::mask_type;

    private:
        mask_type m_value{};

    public:
        // Constructors
        vector() noexcept = default;

        explicit vector(const mask_type val) noexcept : m_value(val) {
        }

        explicit vector(const scalar_type *data) noexcept { load(data); }

        // Access
        mask_type &get() noexcept { return m_value; }
        [[nodiscard]] const mask_type &get() const noexcept { return m_value; }

        // Load/Store
        void load(const scalar_type *data) noexcept {
            m_value = _mm512_loadu_pd(data);
        }

        void store(scalar_type *data) const noexcept {
            _mm512_storeu_pd(data, m_value);
        }

        void load_aligned(const scalar_type *data) noexcept {
            m_value = _mm512_load_pd(data);
        }

        void store_aligned(scalar_type *data) const noexcept {
            _mm512_store_pd(data, m_value);
        }

        // Arithmetic operators
        vector operator+(const vector &other) const noexcept {
            return vector(_mm512_add_pd(m_value, other.m_value));
        }

        vector operator-(const vector &other) const noexcept {
            return vector(_mm512_sub_pd(m_value, other.m_value));
        }

        vector operator*(const vector &other) const noexcept {
            return vector(_mm512_mul_pd(m_value, other.m_value));
        }

        vector operator/(const vector &other) const noexcept {
            return vector(_mm512_div_pd(m_value, other.m_value));
        }

        vector &operator+=(const vector &other) noexcept {
            m_value = _mm512_add_pd(m_value, other.m_value);
            return *this;
        }

        vector &operator-=(const vector &other) noexcept {
            m_value = _mm512_sub_pd(m_value, other.m_value);
            return *this;
        }

        vector &operator*=(const vector &other) noexcept {
            m_value = _mm512_mul_pd(m_value, other.m_value);
            return *this;
        }

        vector &operator/=(const vector &other) noexcept {
            m_value = _mm512_div_pd(m_value, other.m_value);
            return *this;
        }

        // Unary operators
        vector operator-() const noexcept {
            return vector(_mm512_xor_pd(m_value, _mm512_set1_pd(-0.0)));
        }

        // Bitwise operators
        vector operator&(const vector &other) const noexcept {
            return vector(_mm512_and_pd(m_value, other.m_value));
        }

        vector operator|(const vector &other) const noexcept {
            return vector(_mm512_or_pd(m_value, other.m_value));
        }

        vector operator^(const vector &other) const noexcept {
            return vector(_mm512_xor_pd(m_value, other.m_value));
        }

        vector operator~() const noexcept {
            return vector(_mm512_xor_pd(m_value, _mm512_castsi512_pd(_mm512_set1_epi32(-1))));
        }

        vector &operator&=(const vector &other) noexcept {
            m_value = _mm512_and_pd(m_value, other.m_value);
            return *this;
        }

        vector &operator|=(const vector &other) noexcept {
            m_value = _mm512_or_pd(m_value, other.m_value);
            return *this;
        }

        vector &operator^=(const vector &other) noexcept {
            m_value = _mm512_xor_pd(m_value, other.m_value);
            return *this;
        }

        // Comparison operators
        [[nodiscard]] vector cmp_eq(const vector &other) const noexcept {
            const __mmask8 mask = _mm512_cmp_pd_mask(m_value, other.m_value, _CMP_EQ_OQ);
            return vector(_mm512_castsi512_pd(_mm512_movm_epi64(mask)));
        }

        [[nodiscard]] vector cmp_ne(const vector &other) const noexcept {
            const __mmask8 mask = _mm512_cmp_pd_mask(m_value, other.m_value, _CMP_NEQ_OQ);
            return vector(_mm512_castsi512_pd(_mm512_movm_epi64(mask)));
        }

        [[nodiscard]] vector cmp_lt(const vector &other) const noexcept {
            const __mmask8 mask = _mm512_cmp_pd_mask(m_value, other.m_value, _CMP_LT_OQ);
            return vector(_mm512_castsi512_pd(_mm512_movm_epi64(mask)));
        }

        [[nodiscard]] vector cmp_le(const vector &other) const noexcept {
            const __mmask8 mask = _mm512_cmp_pd_mask(m_value, other.m_value, _CMP_LE_OQ);
            return vector(_mm512_castsi512_pd(_mm512_movm_epi64(mask)));
        }

        [[nodiscard]] vector cmp_gt(const vector &other) const noexcept {
            const __mmask8 mask = _mm512_cmp_pd_mask(m_value, other.m_value, _CMP_GT_OQ);
            return vector(_mm512_castsi512_pd(_mm512_movm_epi64(mask)));
        }

        [[nodiscard]] vector cmp_ge(const vector &other) const noexcept {
            const __mmask8 mask = _mm512_cmp_pd_mask(m_value, other.m_value, _CMP_GE_OQ);
            return vector(_mm512_castsi512_pd(_mm512_movm_epi64(mask)));
        }

        vector operator==(const vector &other) const noexcept { return cmp_eq(other); }
        vector operator!=(const vector &other) const noexcept { return cmp_ne(other); }
        vector operator<(const vector &other) const noexcept { return cmp_lt(other); }
        vector operator<=(const vector &other) const noexcept { return cmp_le(other); }
        vector operator>(const vector &other) const noexcept { return cmp_gt(other); }
        vector operator>=(const vector &other) const noexcept { return cmp_ge(other); }

        // Static functions
        static vector blend(const vector &a, const vector &b, const vector &mask) noexcept {
            const __mmask8 k = _mm512_movepi64_mask(_mm512_castpd_si512(mask.m_value));
            return vector(_mm512_mask_blend_pd(k, b.m_value, a.m_value));
        }

        static vector select(const vector &mask, const vector &a, const vector &b) noexcept {
            return blend(a, b, mask);
        }

        static vector min(const vector &a, const vector &b) noexcept {
            return vector(_mm512_min_pd(a.m_value, b.m_value));
        }

        static vector max(const vector &a, const vector &b) noexcept {
            return vector(_mm512_max_pd(a.m_value, b.m_value));
        }

        static vector sqrt(const vector &v) noexcept {
            return vector(_mm512_sqrt_pd(v.m_value));
        }

        static vector abs(const vector &v) noexcept {
            return vector(_mm512_abs_pd(v.m_value));
        }

        static vector fma(const vector &a, const vector &b, const vector &c) noexcept {
            return vector(_mm512_fmadd_pd(a.m_value, b.m_value, c.m_value));
        }

        static vector zero() noexcept {
            return vector(_mm512_setzero_pd());
        }

        static vector set1(const scalar_type val) noexcept {
            return vector(_mm512_set1_pd(val));
        }

        static constexpr size_t size() noexcept { return vector_traits<mask_type>::lanes; }

        // Horizontal operations
        [[nodiscard]] scalar_type horizontal_sum() const noexcept {
            return _mm512_reduce_add_pd(m_value);
        }

        [[nodiscard]] scalar_type horizontal_min() const noexcept {
            return _mm512_reduce_min_pd(m_value);
        }

        [[nodiscard]] scalar_type horizontal_max() const noexcept {
            return _mm512_reduce_max_pd(m_value);
        }
    };

    // ============================================================================
    // Type Definitions
    // ============================================================================

    // Scalar
    using vecsu = vector<int64_t, unsigned>;
    using vecsi = vector<int64_t, signed>;
    using vecsd = vector<long double>;
    using vecsf = vector<float>;

    // SSE
    using vec128u = vector<__m128i, unsigned>;
    using vec128i = vector<__m128i, signed>;
    using vec128d = vector<__m128d>;
    using vec128f = vector<__m128>;

    // AVX
    using vec256u = vector<__m256i, signed>;
    using vec256i = vector<__m256i, unsigned>;
    using vec256d = vector<__m256d>;
    using vec256f = vector<__m256>;

    // AVX2
    using vec512u = vector<__m512i, unsigned>;
    using vec512i = vector<__m512i, signed>;
    using vec512d = vector<__m512d>;
    using vec512f = vector<__m512>;

#if defined(_SIMD_AVX512F)
    using avecu = vec512u;
    using aveci = vec512i;
    using avecf = vec512f;
    using avecd = vec512d;
#elif defined(_SIMD_AVX)
    using avecu = vec256u;
    using aveci = vec256i;
    using avecf = vec256f;
    using avecd = vec256d;
#elif defined(_SIMD_SSE)
    using avecu = vec128u;
    using aveci = vec128i;
    using avecf = vec128f;
    using avecd = vec128d;
#else
    using avecu = vecsu;
    using aveci = vecsi;
    using avecf = vecsf;
    using avecd = vecsd;
#endif
}
