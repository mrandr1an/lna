#pragma once
#include <array>
#include <cstddef>
#include <cmath>
#include <type_traits>

// Minimal fixed-size matrix for embedded use.
// - No dynamic allocation
// - No exceptions
// - Compile-time sizes
// - Header-only

namespace emath {

template <typename T, std::size_t R, std::size_t C>
class Matrix {
  static_assert(R > 0 && C > 0, "Matrix dimensions must be > 0");
  static_assert(std::is_trivially_copyable_v<T>,
                "Prefer trivially copyable numeric types on embedded");

public:
  using value_type = T;
  static constexpr std::size_t rows = R;
  static constexpr std::size_t cols = C;

  // Row-major storage: index = r*C + c
  std::array<T, R * C> a{};

  // Constructors
  constexpr Matrix() = default;

  // Fill constructor
  explicit constexpr Matrix(T v) { fill(v); }

  // Access
  constexpr T& operator()(std::size_t r, std::size_t c) {
    return a[r * C + c];
  }
  constexpr const T& operator()(std::size_t r, std::size_t c) const {
    return a[r * C + c];
  }

  // Bounds-checked access in debug builds (optional)
  constexpr T& at(std::size_t r, std::size_t c) {
    // You can replace with assert() if you enable it.
    return a[r * C + c];
  }
  constexpr const T& at(std::size_t r, std::size_t c) const {
    return a[r * C + c];
  }

  constexpr void fill(T v) {
    for (auto& e : a) e = v;
  }

  static constexpr Matrix zeros() { return Matrix(T{0}); }

  template <std::size_t N = R>
  static constexpr std::enable_if_t<(R == C), Matrix> identity(T diag = T{1}) {
    Matrix m(T{0});
    for (std::size_t i = 0; i < R; ++i) m(i, i) = diag;
    return m;
  }

  // Elementwise operators
  friend constexpr Matrix operator+(const Matrix& x, const Matrix& y) {
    Matrix r;
    for (std::size_t i = 0; i < R * C; ++i) r.a[i] = x.a[i] + y.a[i];
    return r;
  }

  friend constexpr Matrix operator-(const Matrix& x, const Matrix& y) {
    Matrix r;
    for (std::size_t i = 0; i < R * C; ++i) r.a[i] = x.a[i] - y.a[i];
    return r;
  }

  constexpr Matrix& operator+=(const Matrix& y) {
    for (std::size_t i = 0; i < R * C; ++i) a[i] += y.a[i];
    return *this;
  }

  constexpr Matrix& operator-=(const Matrix& y) {
    for (std::size_t i = 0; i < R * C; ++i) a[i] -= y.a[i];
    return *this;
  }

  // Scalar ops
  friend constexpr Matrix operator*(const Matrix& x, T s) {
    Matrix r;
    for (std::size_t i = 0; i < R * C; ++i) r.a[i] = x.a[i] * s;
    return r;
  }
  friend constexpr Matrix operator*(T s, const Matrix& x) { return x * s; }

  friend constexpr Matrix operator/(const Matrix& x, T s) {
    Matrix r;
    for (std::size_t i = 0; i < R * C; ++i) r.a[i] = x.a[i] / s;
    return r;
  }

  constexpr Matrix& operator*=(T s) {
    for (auto& e : a) e *= s;
    return *this;
  }

  constexpr Matrix& operator/=(T s) {
    for (auto& e : a) e /= s;
    return *this;
  }

  // Hadamard (elementwise product)
  friend constexpr Matrix hadamard(const Matrix& x, const Matrix& y) {
    Matrix r;
    for (std::size_t i = 0; i < R * C; ++i) r.a[i] = x.a[i] * y.a[i];
    return r;
  }

  // Transpose
  constexpr Matrix<T, C, R> transpose() const {
    Matrix<T, C, R> t;
    for (std::size_t r = 0; r < R; ++r)
      for (std::size_t c = 0; c < C; ++c)
        t(c, r) = (*this)(r, c);
    return t;
  }

  // Map function over elements (useful for sigmoid, etc.)
  template <typename F>
  constexpr auto map(F&& f) const -> Matrix<std::invoke_result_t<F, T>, R, C> {
    using U = std::invoke_result_t<F, T>;
    Matrix<U, R, C> r;
    for (std::size_t i = 0; i < R * C; ++i) r.a[i] = static_cast<U>(f(a[i]));
    return r;
  }

  // Reduce helpers
  constexpr T sum() const {
    T s{};
    for (const auto& e : a) s += e;
    return s;
  }

  constexpr T l2_sq() const { // sum of squares
    T s{};
    for (const auto& e : a) s += e * e;
    return s;
  }
};

// Matrix multiply: (R x K) * (K x C) -> (R x C)
template <typename T, std::size_t R, std::size_t K, std::size_t C>
constexpr Matrix<T, R, C> matmul(const Matrix<T, R, K>& A, const Matrix<T, K, C>& B) {
  Matrix<T, R, C> out(T{0});
  for (std::size_t r = 0; r < R; ++r) {
    for (std::size_t c = 0; c < C; ++c) {
      T acc{};
      for (std::size_t k = 0; k < K; ++k) acc += A(r, k) * B(k, c);
      out(r, c) = acc;
    }
  }
  return out;
}

// Dot product for vectors represented as (N x 1) or (1 x N)
template <typename T, std::size_t N>
constexpr T dot(const Matrix<T, N, 1>& x, const Matrix<T, N, 1>& y) {
  T s{};
  for (std::size_t i = 0; i < N; ++i) s += x(i, 0) * y(i, 0);
  return s;
}

template <typename T, std::size_t N>
constexpr T dot(const Matrix<T, 1, N>& x, const Matrix<T, 1, N>& y) {
  T s{};
  for (std::size_t i = 0; i < N; ++i) s += x(0, i) * y(0, i);
  return s;
}

// Stable-ish sigmoid for float/double.
inline float sigmoid(float z) {
  // prevent overflow in expf on large magnitude
  if (z >= 0.0f) {
    const float ez = std::exp(-z);
    return 1.0f / (1.0f + ez);
  } else {
    const float ez = std::exp(z);
    return ez / (1.0f + ez);
  }
}

} // namespace emath
