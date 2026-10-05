#pragma once 

#include <vector>

/// \tparam T
template<typename T>
struct BIT {
    const size_t n = 0; 
    std::vector<T> A;
    BIT(const size_t _n) : n(_n), A(_n+1, 0) {}
    BIT(std::vector<T> &v) : n(sz(v)), A(1) {
        for (auto x:v) { 
            A.pb(x); 
        }
        for (size_t i=1, j; j=i&-i, i<=n; i++) {
            if (i+j <= n) {
                A[i+j] += A[i];
            }
        }
    }

    // a[i] += v
    void update(const size_t pos, 
                const T &v) { 
        size_t i = pos;
        while (i <= n) { 
            A[i] += v, i += i&-i;
        }
    }

    // sum_{j<=i} a[j]
    T query(const size_t pos) { 
        size_t i = pos;
        T v = 0;
        while (i) { 
            v += A[i], i -= i&-i;
        }
        return v;
    }
};

/// \tparam T
template<typename T>
struct rangeBIT {
    const size_t n = 0; 
    BIT<T> b1, b2;
    rangeBIT(int _n) : n(_n), b1(_n), b2(_n+1) {}
    rangeBIT(std::vector<T> &v) : n(sz(v)), b1(v), b2(sz(v)+1) {}
    
    void pupdate(const size_t i, T v) { 
        b1.update(i, v); 
    }

    // a[i,..,j]+=v
    void rupdate(const size_t i,
                 const size_t j,
                 const T &v) { 
        b2.update(i, v);
        b2.update(j+1, -v);
        b1.update(j+1, v*j);
        b1.update(i, (1-i)*v);
    }
   
    /// \param i[in];
    T query(const size_t i){
        return b1.query(i) + 
               b2.query(i)*i;
    }
};
