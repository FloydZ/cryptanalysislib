#pragma once 

namespace cryptanalysislib::algorithm::bit {

/// Compute (nonadjacent form, NAF) signed binary representation of x:
/// the unique representation of x as
///   x=\sum_{k}{d_k*2^k} where d_j \in {-1,0,+1}
///   and no two adjacent digits d_j, d_{j+1} are both nonzero.
/// np has bits j set where d_j==+1
/// nm has bits j set where d_j==-1
/// We have:  x = np - nm
template<typename T>
constexpr static inline void bin2naf(T &np, 
                                     T &nm, 
                                     const T x) noexcept {
    T xh = x >> 1;  // x/2
    T x3 = x + xh;  // 3*x/2
    T c = xh ^ x3;
    np = x3 & c;
    nm = xh & c;
}

/// Inverse of bin2naf()
/// Works also for signed-binary pairs np, nm
/// that are not nonadjacent forms.
template<typename T>
constexpr static inline T naf2bin(const T np, 
                                  const T nm) noexcept {
    return  np - nm;
}

/// Compute a signed binary representation of x:
/// a representation of x as
///   x=\sum_{k}{d_k*2^k} where d_j \in {-1,0,+1}
/// Inverse function is naf2bin().
template<typename T>
constexpr static inline void bin2sbin(T &np, 
                                      T &nm, 
                                      const T x) noexcept {
    T xh = x << 1;
    T xr = x ^ xh;
    np = xr & xh;
    nm = xr & x;
}
} // end namespace cryptanalysislib::algorithm::bit
