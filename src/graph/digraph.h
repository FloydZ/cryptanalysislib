#pragma once 

#include <cassert>
#include <iomanip>
#include <iostream>

#include "alloc/alloc.h"
#include "algorithm/bits/popcount.h"
#include "algorithm/reverse.h"
#include "math/math.h"
#include "memory/memory.h"
#include "random.h"
#include "sort/sort.h"

/// NOTE: ported from fxt. Before, this header did not compile: the helper
/// 	functions of fxt were missing, `digraph_paths` was declared twice (once
/// 	nested in `digraph` with its definitions in the fxt `.cc` files), and
/// 	`T` and `uint64_t` were mixed.
/// NOTE: the permutations of the `make_perm_*` graphs are numbered with
/// 	a factorial number system (see `num2perm_ffact`, `num2perm_rfact`).
/// 	The graphs are the same as in fxt, the node numbering may differ.
namespace cryptanalysislib::internal::digraph {
	/// \return whether exactly one bit of `x` is set
	[[nodiscard]] constexpr inline bool one_bit_q(const uint64_t x) noexcept {
		return (x != 0) && ((x & (x - 1u)) == 0);
	}

	/// \return the Fibonacci (Zeckendorf) representation of `k`:
	/// 	bit `i` set <=> F(i+2) is used, with F(2) = 1, F(3) = 2, F(4) = 3, ...
	[[nodiscard]] constexpr inline uint64_t bin2fibrep(uint64_t k) noexcept {
		uint64_t f[92];
		uint32_t m = 0;
		f[0] = 1; f[1] = 2;
		for (m = 2; m < 92; m++) {
			f[m] = f[m - 1] + f[m - 2];
			if (f[m] > k) { break; }
		}

		uint64_t ret = 0;
		for (uint32_t i = m; i-- > 0;) {
			if (f[i] <= k) {
				k -= f[i];
				ret |= 1ull << i;
			}
		}
		return ret;
	}

	/// \return `x` rotated left by `r` within the lowest `n` bits
	[[nodiscard]] constexpr inline uint64_t bit_rotate_left(const uint64_t x,
	                                                        const uint32_t r,
	                                                        const uint32_t n) noexcept {
		assert((n > 0) && (n <= 64) && (r < n));
		const uint64_t mask = (n == 64) ? ~0ull : ((1ull << n) - 1u);
		if (r == 0) { return x & mask; }
		return ((x << r) | ((x & mask) >> (n - r))) & mask;
	}

	/// \return the first combination (in colex order) of `k` bits
	[[nodiscard]] constexpr inline uint64_t first_comb(const uint32_t k) noexcept {
		assert(k < 64);
		return (1ull << k) - 1u;
	}

	/// \return the next combination in colex order with the same number of bits
	[[nodiscard]] constexpr inline uint64_t next_colex_comb(const uint64_t x) noexcept {
		assert(x != 0);
		const uint64_t u = x & (~x + 1u);
		const uint64_t v = u + x;
		return v + (((v ^ x) / u) >> 2u);
	}

	/// \return whether the lowest `len` bits of `x` are a balanced paren
	/// 	word, read from bit 0 upwards, with 1 = '(' and 0 = ')'.
	[[nodiscard]] constexpr inline bool is_parenword(uint64_t x,
	                                                 const uint32_t len) noexcept {
		int64_t s = 0;
		for (uint32_t i = 0; i < len; i++, x >>= 1u) {
			s += (x & 1u) ? 1 : -1;
			if (s < 0) { return false; }
		}
		return s == 0;
	}

	/// \return n!
	[[nodiscard]] constexpr inline uint64_t factorial(const uint64_t n) noexcept {
		assert(n <= 20);
		uint64_t ret = 1;
		for (uint64_t i = 2; i <= n; i++) { ret *= i; }
		return ret;
	}

	/// permutation `x` of [0, n) with the Lehmer code (falling factorial
	/// base) of `k`: digit `i` = #{j > i: x[j] < x[i]} with weight (n-1-i)!
	template<typename T>
	constexpr inline void num2perm_ffact(uint64_t k,
	                                     T *x,
	                                     const uint32_t n) noexcept {
		assert(n <= 20);
		T avail[20];
		for (uint32_t i = 0; i < n; i++) { avail[i] = T(i); }
		for (uint32_t i = 0; i < n; i++) {
			const uint64_t f = factorial(n - 1 - i);
			uint32_t d = k / f;
			k %= f;
			x[i] = avail[d];
			for (uint32_t j = d; j + 1 < n - i; j++) { avail[j] = avail[j + 1]; }
		}
	}

	/// inverse of `num2perm_ffact`
	template<typename T>
	[[nodiscard]] constexpr inline uint64_t perm2num_ffact(const T *x,
	                                                       const uint32_t n) noexcept {
		uint64_t k = 0;
		for (uint32_t i = 0; i < n; i++) {
			uint64_t d = 0;
			for (uint32_t j = i + 1; j < n; j++) { d += x[j] < x[i]; }
			k += d * factorial(n - 1 - i);
		}
		return k;
	}

	/// permutation `x` of [0, n) with the rising factorial base digits of
	/// `k`: digit `i` = #{j < i: x[j] > x[i]} with weight i!
	template<typename T>
	constexpr inline void num2perm_rfact(uint64_t k,
	                                     T *x,
	                                     const uint32_t n) noexcept {
		assert(n <= 20);
		uint64_t d[20];
		for (uint32_t i = 0; i < n; i++) {
			d[i] = k % (i + 1u);
			k /= (i + 1u);
		}

		// the values of x[0..i] are the smallest i+1 values not used by x[i+1..]
		T avail[20];
		for (uint32_t i = 0; i < n; i++) { avail[i] = T(i); }
		for (uint32_t i = n; i-- > 0;) {
			const uint32_t pos = i - d[i];
			x[i] = avail[pos];
			for (uint32_t j = pos; j < i; j++) { avail[j] = avail[j + 1]; }
		}
	}

	/// inverse of `num2perm_rfact`
	template<typename T>
	[[nodiscard]] constexpr inline uint64_t perm2num_rfact(const T *x,
	                                                       const uint32_t n) noexcept {
		uint64_t k = 0;
		for (uint32_t i = n; i-- > 0;) {
			uint64_t d = 0;
			for (uint32_t j = 0; j < i; j++) { d += x[j] > x[i]; }
			k = k * (i + 1u) + d;
		}
		return k;
	}

	/// \return index of `v` in the sorted array `a` of length `n`, or `n`
	template<typename T>
	[[nodiscard]] constexpr inline uint64_t bsearch(const T *a,
	                                                const uint64_t n,
	                                                const T v) noexcept {
		uint64_t l = 0, h = n;
		while (l < h) {
			const uint64_t m = l + (h - l) / 2u;
			if (a[m] < v) { l = m + 1u; } else { h = m; }
		}
		return ((l < n) && (a[l] == v)) ? l : n;
	}
} // end namespace cryptanalysislib::internal::digraph

template <typename T, class Allocator>
class digraph_paths;

// Directed graph with ng nodes.
// Initialization just allocates memory,
//  filling in the edges is left to the user.
template <typename T=uint32_t, 
          class Allocator = cryptanalysislib::allocator<T>>
class digraph {
private:
    friend class digraph_paths<T, Allocator>;

    Allocator allocator;

    // number of Nodes of Graph
    T ng_;

    // e[ep[k]], ..., e[ep[k+1]-1]: outgoing connections of node k
    T *ep_;

    // outgoing connections (Edges)
    T *e_;

    // optional: sorted values for nodes
    T *vn_;
    // if vn is used, then node k must correspond to vn[k]

    // number of edges, i.e. the size of `e_`
    T ne_;

    digraph(const digraph&) = delete;
    digraph & operator = (const digraph&) = delete;

    /// \return the value used to sort the edges: `vn[x]` if set, else `x`
    [[nodiscard]] constexpr T sort_key(const T x) const noexcept {
        return vn_ ? vn_[x] : x;
    }

public:
    /// \param ng[in]: number of nodes
    /// \param ne[in]: number of edges
    /// \param ep[out]: set to the edge pointers (`ng+1` elements)
    /// \param e[out]: set to the edges (`ne` elements)
    /// \param vnq[in]: if true, an array for the node values is allocated
    explicit digraph(const T ng,
                     const T ne,
                     T *&ep,
                     T *&e,
                     bool vnq=false) noexcept
    : ng_(0), ep_(nullptr), e_(nullptr), vn_(nullptr), ne_(ne) {
        ng_ = ng;
        ep_ = allocator.allocate(ng_ + 1u);
        e_ = allocator.allocate(ne);
        if ( vnq ) { vn_ = allocator.allocate(ng_); }
        // NOTE: after the allocation. Before, the caller got `nullptr`.
        ep = ep_;
        e = e_;
    }

    ~digraph() noexcept {
        allocator.deallocate(ep_, ng_+1u);
        // NOTE: was a count of 1 (sized deallocation with the wrong size)
        allocator.deallocate(e_, ne_);
        if (vn_) { allocator.deallocate(vn_, ng_); }
    }


    [[nodiscard]] constexpr T num_nodes() const noexcept { return ng_; }
    [[nodiscard]] constexpr T num_edges() const noexcept { return ep_[num_nodes()]; }

    /// \return the node values (`nullptr` if not allocated)
    [[nodiscard]] constexpr const T *node_values() const noexcept { return vn_; }

    // Return how many outgoing edges are at node p.
    [[nodiscard]] constexpr T num_edges(T p) const noexcept {
        return  ep_[p+1] - ep_[p];
    }

    // Setup fe and en so that the nodes reachable from p are
    //   e[fe], e[fe+1], ..., e[en-1].
    // Must have:  0<=p<ng
    constexpr void get_edge_idx(const T p,
                                T &fe,
                                T &en) const noexcept {
        fe = ep_[p];   // (index of) First Edge
        en = ep_[p+1];  // (index of) first Edge of Next node
    }

    // Return the index of the edge that goes from p to pn.
    // Return value t:
    //   0<=t<num_edges(p)  if an edge from p to pn exists
    //   T(-1)  else
    [[nodiscard]] constexpr T edge_idx(const T p,
                                       const T pn) const noexcept {
        const T fe = ep_[p];   // (index of) First Edge
        const T nt = num_edges(p);
        const T *e = e_ + fe;
        for (T t=0; t<nt; ++t) {
            if (pn==e[t]) { 
                return t; 
            }
        }
        return  T(-1);  // pn cannot be reached from p
    }

    // Return whether edge from p to pn exists
    [[nodiscard]] constexpr bool has_edge(const T p,
                                          const T pn) const noexcept  {
        return (edge_idx(p, pn) < num_edges(p)); 
    }

    // Return max number (among all nodes) of outgoing edges.
    [[nodiscard]] constexpr T max_edges() const noexcept {
        T ma = 0;  // max number of outgoing edges
        for (T k=0; k<ng_; ++k) {
            T n = ep_[k+1] - ep_[k];
            if (n > ma) {
                ma = n;
            }
        }
        return  ma;
    }

    /// sorts the outgoing edges of each node by their node value (`vn`, if
    /// set) or index.
    /// \param rq[in]: 1: ascending, 0: descending
    void sort_edges(const int rq=1) noexcept {
        for (T k=0; k<ng_; ++k) {
            const T x = ep_[k];
            const T n = ep_[k+1] - x;
            cryptanalysislib::sort(e_+x, e_+x+n, [this, rq](const T a, const T b) {
                return rq ? (sort_key(a) < sort_key(b)) : (sort_key(a) > sort_key(b));
            });
        }
    }

    // Test for each node whether sets of outgoing edges are sorted.
    // If the test fails for a node, return its index,
    //  else return ng.
    /// \param rq[in]: 1: ascending, 0: descending
    [[nodiscard]] constexpr T test_edge_sorted(const int rq=1) const noexcept  {
        for (T k=0; k<ng_; ++k) {
            for (T j=ep_[k]; j+1<ep_[k+1]; ++j) {
                const T a = sort_key(e_[j]), b = sort_key(e_[j+1]);
                if (rq ? (b < a) : (a < b)) { return k; }
            }
        }
        return ng_;
    }

    [[nodiscard]] constexpr bool is_edge_sorted(const int rq=1) const noexcept {
        return ( ng_==test_edge_sorted(rq));
    }

    // Reverse order of edges at positions p0,...,p1.
    // If p1==0 then action is performed just for position p0.
    constexpr void reverse_edge_order(const T p0,
                                      const T p1=0) noexcept {
        T p = p0;
        do {
            T n = num_edges(p);
            if (n > 1) { reverse(e_+ep_[p], e_+ep_[p]+n); }
        } while ( ++p<=p1 );  // note: inclusive p1
    }

    constexpr void reverse_edge_order() noexcept {
        reverse_edge_order(0, ng_-1); 
    }

    // Random permute order of edges at positions p0,...,p1.
    // If p1==0 then action is performed just for position p0.
    void randomize_edge_order(const T p0, const T p1=0) noexcept {
        T p = p0;
        do {
            T n = num_edges(p);
            T *e = e_ + ep_[p];
            // Fisher-Yates
            for (T i = n; i > 1; --i) {
                const T j = T(cryptanalysislib::rng() % i);
                std::swap(e[i-1], e[j]);
            }
        } while ( ++p<=p1 );  // note: inclusive p1
    }

    void randomize_edge_order() noexcept { 
        randomize_edge_order(0, ng_-1); 
    }


    void print(const char *bla=nullptr)  const noexcept  {
        if (bla) { 
            std::cout << bla << std::endl; 
        }

        std::cout << "Node: Edge0 Edge1 ..." << std::endl;
        for (T k=0; k<ng_; ++k) {
            std::cout << std::setw(3) << k << ":  ";
            for (T j=ep_[k]; j<ep_[k+1]; ++j) {
                std::cout << std::setw(3) << e_[j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "  #nodes=" << num_nodes();
        std::cout << "  #edges=" << num_edges();
        std::cout << std::endl;
    }

    void print_horiz(const char *bla=nullptr)  const noexcept {
        if (bla) {
            std::cout << bla << std::endl;
        }

        std::cout << std::setw(7) << "Node:";
        for (T k=0; k<ng_; ++k) {
            std::cout << " " << std::setw(2) << k;
        }

        std::cout << std::endl;
        const T ma = max_edges();
        for (T j=0; j<ma; ++j) {
            std::cout << std::setw(1) << "Edge" << std::setw(2) << j << ":";
            for (T k=0; k<ng_; ++k) {
                if (num_edges(k) > j) {
                    std::cout << " " << std::setw(2) << e_[ep_[k]+j];
                }
                else  std::cout << "   ";
            }
            std::cout << std::endl;
        }
    }

    /// \return 0 if the graph is consistent, else an error code
    [[nodiscard]] constexpr T test() const noexcept {
        T ng = ng_;
        for (T k=0; k<ng; ++k)  if ( ep_[k] > ep_[k+1] )  return 1;
    
        const T ne = num_edges();
        // NOTE: `>` instead of `>=`: a node without edges at the end has
        // 	`ep[k] == ne`.
        for (T k=0; k<ng; ++k)  if ( ep_[k] > ne  )  return 2;
    
        if (ne > ne_) {
            return 3;
        }
    
        for (T k=0; k<ne; ++k) {
            if (e_[k] >= ng_) { 
                return 10; 
            }
        }
    
        return 0;
    }

    [[nodiscard]] constexpr bool OK() const noexcept {
        return (0==test()); 
    }

    /// Initialization for the complete graph.
    /// \param n[in] 
    static digraph *make_complete_digraph(const T n) noexcept {
        T ng = n, ne = n*(n-1);
    
        T *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e);
    
        T j = 0;
        // for all nodes
        for (T k=0; k<ng; ++k) {
            ep[k] = j;
            // connect to all nodes
            for (T i=0; i<n; ++i) {
                if ( k==i )  continue;  // skip loops
                e[j++] = i;
            }
        }
        ep[ng] = j;
    
        return  dgp;
    }

    /// De Bruijn graph with 2*n nodes
    /// \param n[in] 
    static digraph *make_debruijn_digraph(const T n) noexcept {
        T ng = 2*n, ne = 2*ng;
        T *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e);
    
        T j = 0;
        for (T k=0; k<ng; ++k)  // for all nodes
        {
            ep[k] = j;
            T r = (2*k) % ng;
            e[j++] = r;  // connect node k to node (2*k) mod ng
            r = (2*k+1) % ng;
            e[j++] = r;  // connect node k to node (2*k+1) mod ng
        }
        ep[ng] = j;
    
        return  dgp;
    }
    
    /// \param n[in] 
    static digraph * make_complement_shift_digraph(const T n) noexcept {
        T ng = 2*n, ne = 2*ng;
        T *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e);
    
        T i = 0;
        for (T k=0; k<ng; ++k) {
            ep[k] = i;
            T r = (2*k) % ng;
            e[i++] = r;  // connect node k to node (2*k) mod ng
            r = (2*k+1) % ng;
            e[i++] = r;  // connect node k to node (2*k+1) mod ng
        }
        ep[ng] = i;
        // Here we have a De Bruijn graph.
    
        for (T k=0, j=ng-1;  k<j;  ++k, --j) std::swap(e[ep[k]], e[ep[j]]);  // end with ones
        for (T k=0, j=ng-1;  k<j;  ++k, --j) std::swap(e[ep[k]+1], e[ep[j]+1]);
    
        return  dgp;
    }
    
    /// m-ary version
    /// \param n[in] 
    static digraph* make_debruijn_digraph(const T n,
                                          const T m) noexcept {
        T ng = m*n, ne = m*ng;
        T *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e);
    
        T j = 0;
        for (T k=0; k<ng; ++k)  {
            ep[k] = j;
            for (T i=0; i<m; ++i) {
                T r = (m*k+i) % ng;
                e[j++] = r;  // connect node k to node (m*k+j) mod ng
            }
        }
        ep[ng] = j;
    
        return  dgp;
    }

    /// nodes 0..n-1, connected if their Fibonacci representations differ in one bit
    static digraph *make_fibrepgray_digraph(const T n) noexcept {
        using namespace cryptanalysislib::internal::digraph;
        T *f = new T[n];
        for (T k=0; k<n; ++k) { f[k] = T(bin2fibrep(k)); }

        T nc = 0;
        for (T k=0; k<n; ++k) {
            const T fk = f[k];
            for (T j=0; j<n; ++j) {
                if ( j==k )  continue;
                const T fj = f[j];
                if ( one_bit_q( fj^fk ) )  ++nc;
            }
        }
    
        T *ep, *e;
        digraph * dgp = new digraph(n, nc, ep, e, true);
        digraph &dg = *dgp;
        cryptanalysislib::memcpy<T>(dg.vn_, f, n);
    
        T tnc = 0;
        for (T k=0; k<n; ++k)
        {
            ep[k] = tnc;
            const T fk = f[k];
            for (T j=0; j<n; ++j)
            {
                if ( j==k )  continue;
                const T fj = f[j];
                if ( one_bit_q( fj^fk ) )  e[tnc++] = j;
            }
        }
        assert( nc == tnc );
        ep[n] = tnc;
    
        delete [] f;
        return  dgp;
    }
    
    /// Initialization for directed graph:
    /// Gray code graph for n-bit words.
    /// \param rq[in]: force path to start as 0 1 3
    static digraph *make_gray_digraph(const T n, const bool rq=false) noexcept {
        const T ng = T(1u) << n;
    
        // number of edges
        // NOTE: with `rq` the nodes 0 and 1 have only one edge
        const T ne = ng * n - (rq ? 2*(n-1) : 0);
        T *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e);
    
        T p = 0;
        T k = 0;
        if ( rq )  // force path to start as 0 1 3:
        {
            ep[k] = p;  e[p++] = 1;  ++k;  // 0 --> 1
            ep[k] = p;  e[p++] = 3;  ++k;  // 1 --> 3
        }
    
        for (  ; k<ng; ++k)  // for all nodes
        {
            ep[k] = p;
            for (T c=0, b=1;  c<n;  ++c, b<<=1)
            {
                const T vc = k ^ b;  // change one bit
                e[p++] = vc;
            }
        }
        ep[ng] = p;
        assert(p == ne);
    
        return  dgp;
    }
    
    /// Initialization for the "middle two levels" graph
    /// \param rq[in]: force path to start "canonically"
    static digraph *make_mtl_digraph(const T k, const bool rq=false) noexcept {
        using namespace cryptanalysislib::internal::digraph;
        const T k2 = 2*k-1;
        const T ng = T(2*bc(k2, k));
        T ne = ng * k;  // number of edges
        if ( rq )  ne -= (k-1);
    
        T *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e, true);
        digraph &dg = *dgp;
    
        T *vn = dg.vn_;
        const uint64_t mask = first_comb(k2);
        uint64_t comb = first_comb(k);
        T nct = 0;  // Node counter
        do
        {
            vn[nct++] = T(comb);
            assert( nct < ng );
            vn[nct++] = T(mask & ~comb);
            comb = next_colex_comb(comb);
        }
        while ( comb < mask );
        assert( nct == ng );
    
        cryptanalysislib::sort(vn, vn + ng);
    
        T p = 0;
        T j = 0;
        if ( rq )  // force path to start "canonically":
        {
            const T x = k;
            ep[j] = p;  e[p++] = x;  ++j;  // 0000111 --> 0001111
        }
    
        for (  ; j<ng; ++j)  // for all nodes
        {
            ep[j] = p;
            const T v = vn[j];  // value of node
            for (uint64_t b=1;  0!=(b & mask);  b<<=1)
            {
                const T vc = T(v ^ b);  // change one bit
                const uint64_t x = cryptanalysislib::internal::digraph::bsearch(vn, ng, vc);
                if ( ng != x )
                {
                    assert( p<ne );
                    e[p++] = T(x);
                }
            }
        }
        ep[ng] = p;
        assert( p==ne );
    
        return  dgp;
    }
    
    
    constexpr static uint64_t Catalan[]=
    {
        0UL, 1UL, 2UL, 5UL, 14UL, 42UL, 132UL, 429UL, 1430UL, 4862UL, 16796UL,
        58786UL, 208012UL, 742900UL, 2674440UL, 9694845UL, 35357670UL
    };
    
    /// \param pcd[in]: 0: Gray, 1: changes '11' and '101' only, 2: changes '11' only
    [[nodiscard]] constexpr static bool parengray_is_neighbor(const uint64_t fk,
                                                              const uint64_t fj,
                                                              const uint64_t pcd) noexcept {
        const uint64_t xr = fj^fk;
        bool q = false;
        if ( 2==cryptanalysislib::popcount::popcount( xr ) )
        {
            switch ( pcd )
            {
            case 0:  // Gray:
                q = true;  break;
            case 1:  // changes '11' and '101' only (paths ex. for all n):
                if ( (xr&(xr>>1)) || (xr&(xr>>2)) )  q = true;
                break;
            case 2:  // changes '11' only (path exists for n=6):
                if ( xr & (xr>>1) )  q = true;
                break;
            default:  assert(0);  // criterion does not exist;
            }
        }
    
        return q;
    }
    
    /// graph on the balanced paren words with `nb` pairs, sorted ascending
    /// \param nb[in]: number of paren pairs, 1 <= nb <= 16
    /// \param pcd[in]: see `parengray_is_neighbor`
    static digraph *make_parengray_digraph(const T nb, const T pcd) noexcept {
        using namespace cryptanalysislib::internal::digraph;
        assert((nb >= 1) && (nb <= 16));
        const T n = T(Catalan[nb]);
        T *f = new T[n];
        {
            // NOTE: ascending colex order. Was the descending order (via
            // 	`prev_colex_comb`) followed by a reversal.
            T k = 0;
            const uint64_t end = 1ull << (2*nb);
            for (uint64_t c = first_comb(nb); c < end; c = next_colex_comb(c)) {
                if ( is_parenword(c, 2*nb) ) {
                    assert( k<n );
                    f[k++] = T(c);
                }
            }
            assert( k==n );
        }
    
        T nc = 0;
        for (T k=0; k<n; ++k)  // count number of edges
        {
            for (T j=0; j<n; ++j)
            {
                if ( j==k )  continue;
                if ( parengray_is_neighbor(f[k], f[j], pcd) )  ++nc;
            }
        }
    
        T *ep, *e;
        digraph *dgp = new digraph(n, nc, ep, e, true);
        digraph &dg = *dgp;
        cryptanalysislib::memcpy<T>(dg.vn_, f, n);

        nc = 0;
        for (T k=0; k<n; ++k)  // fill in edges
        {
            ep[k] = nc;
            for (T j=0; j<n; ++j)
            {
                if ( j==k )  continue;
                if ( parengray_is_neighbor(f[k], f[j], pcd) )  e[nc++] = j;
            }
        }
        ep[n] = nc;
    
        delete [] f;
        return  dgp;
    }
    
    // star transpositions:
    static inline void star_swap(T *x, const T c) noexcept {
        std::swap( x[0], x[c] );
    }
    
    // adjacent transpositions:
    static inline void adj_swap(T *x, const T c) noexcept {
        std::swap(x[c-1], x[c]);
    }
    
    /// Initialization for directed graph:
    /// Gray code permutations of n elements
    /// with star transpositions if stq==true,
    /// otherwise with adjacent changes.
    static digraph *make_perm_gray_digraph(const T n, const bool stq) noexcept {
        using namespace cryptanalysislib::internal::digraph;
        assert(n <= 20);
        const T ng = T(factorial(n));
        const T ne = ng * (n-1);  // number of edges
        T *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e);
    
        T xx[32];  // permutations
        T p = 0;
        for (T k=0; k<ng; ++k)  // for all nodes
        {
            ep[k] = p;
            num2perm_rfact(k, xx, n);
    
            for (T j=1;  j<n;  ++j)
            {
                if ( stq ) star_swap(xx, j);
                else       adj_swap(xx, j);
    
                e[p++] = T(perm2num_rfact(xx, n));
    
                // unswap:
                if ( stq ) star_swap(xx, j);
                else       adj_swap(xx, j);
            }
        }
        ep[ng] = p;
    
        return  dgp;
    }
    
    /// Initialization for directed graph:
    /// permutations are connected by prefix reversals
    static digraph *make_perm_pref_rev_digraph(const T n) noexcept {
        using namespace cryptanalysislib::internal::digraph;
        assert(n <= 20);
        const T ng = T(factorial(n));
        const T ne = ng * (n-1);  // number of edges
        T *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e);
    
        T xx[32];  // aux: permutations
        T yy[32];  // aux: prefix-reversed permutations
        T p = 0;
        for (T k=0; k<ng; ++k)  // for all nodes
        {
            ep[k] = p;
    
            num2perm_ffact(k, xx, n);
            for (T j=2;  j<=n;  ++j)
            {
                for (T i=0; i<n; ++i)  yy[i] = xx[i];
                reverse(yy, yy + j);
                e[p++] = T(perm2num_ffact(yy, n));
            }
        }
        ep[ng] = p;
    
        return  dgp;
    }
    
    /// Initialization for directed graph:
    /// permutations are connected by prefix rotations,
    /// rq = 1 ==> right rotations, otherwise left rotations.
    static digraph *make_perm_pref_rot_digraph(const T n, const bool rq=false) noexcept {
        using namespace cryptanalysislib::internal::digraph;
        assert(n <= 20);
        const T ng = T(factorial(n));
        const T ne = ng * (n-1);  // number of edges
        T *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e);
    
        T xx[32];  // aux: permutations
        T yy[32];  // aux: prefix-rotated permutations
        T p = 0;
        for (T k=0; k<ng; ++k)  // for all nodes
        {
            ep[k] = p;
    
            num2perm_ffact(k, xx, n);
            for (T j=2;  j<=n;  ++j)
            {
                for (T i=0; i<n; ++i)  yy[i] = xx[i];
                if ( rq ) {
                    // rotate right by one: yy[0] = yy[j-1]
                    const T t = yy[j-1];
                    for (T i=j-1; i>0; --i)  yy[i] = yy[i-1];
                    yy[0] = t;
                } else {
                    // rotate left by one: yy[j-1] = yy[0]
                    const T t = yy[0];
                    for (T i=0; i+1<j; ++i)  yy[i] = yy[i+1];
                    yy[j-1] = t;
                }
    
                e[p++] = T(perm2num_ffact(yy, n));
            }
        }
        ep[ng] = p;
    
        return  dgp;
    }
};


// Find all full paths in a directed graph.
template <typename T=uint32_t, 
          class Allocator = cryptanalysislib::allocator<T>>
class digraph_paths {
private:
    using G = digraph<T, Allocator>;
    Allocator allocator;

    // the graph
    G &g_;

    // Record of Visits: rv[k] == node visited at step k
    T *rv_;

    // qq[k] == whether node k has been visited yet
    T *qq_;

    // count Paths
    size_t pct_ = 0;

    // count Cycles
    size_t cct_ = 0;
    
    // count Paths where pfunc() returns 1
    size_t pfct_ = 0;

    // whether current path is a cycle
    bool cq_ = 0; 

    // == g_.ng_
    T ng_;

    // number of bits in ng_, used for printing
    T ngbits_ = 0;

    // function to call with each path found with all_paths():
    uint64_t (*pfunc_)(const digraph_paths &) = nullptr;

    // if set (by pfunc()) then search is stopped
    bool pfdone_ = 0;  

    // stop after maxnp times that pfunc returned one (0==forever)
    size_t maxnp_ = 0;

    // function to impose condition with all_cond_paths():
    bool (*cfunc_)(digraph_paths &, uint64_t ns) = nullptr;  // can set pfdone_

    digraph_paths(const digraph_paths&) = delete;
    digraph_paths & operator = (const digraph_paths&) = delete;

public:
    explicit digraph_paths(G &g)  noexcept :
        g_(g), ng_(g.ng_) {
        rv_ = allocator.allocate(ng_);
        qq_ = allocator.allocate(ng_);
        ngbits_ = T(ceil_log2(ng_));
        init();
    }

    ~digraph_paths() noexcept {
        allocator.deallocate(rv_, ng_);
        allocator.deallocate(qq_, ng_);
    }

    /// clears the visit marks and the recorded path
    constexpr void init() noexcept {
        cryptanalysislib::memset<T>(rv_, 0, ng_);
        cryptanalysislib::memset<T>(qq_, 0, ng_);
    }

    [[nodiscard]] constexpr const G & graph() const noexcept { return g_; }

    /// \return the recorded path: rv[k] == node visited at step k
    [[nodiscard]] constexpr const T *path() const noexcept { return rv_; }

    /// \return number of paths found by the last search
    [[nodiscard]] constexpr size_t num_paths() const noexcept { return pct_; }

    /// \return number of cycles found by the last search
    [[nodiscard]] constexpr size_t num_cycles() const noexcept { return cct_; }

    /// \return whether the current path is a cycle (valid within pfunc)
    [[nodiscard]] constexpr bool is_cycle() const noexcept { return cq_; }

    /// \param pfdone[in]: if set (by pfunc()) then the search is stopped
    constexpr void set_done(const bool pfdone=true) noexcept { pfdone_ = pfdone; }

    // Return whether the path is a cycle.
    [[nodiscard]] constexpr bool path_is_cycle()  const noexcept {
        // first node visited
        const T p0 = rv_[0];
        
        // last node visited
        const T p = rv_[ng_-1];
        return graph().has_edge(p, p0);
    }

    void print_turns(bool shortq=true) const noexcept {
        std::cout << "Path:";
        if ( shortq )  std::cout << " (short print) ";
        std::cout << std::endl;
        T nffct = 0;  // count non-first-free turns
        for (T k=0; k<ng_-1; ++k)
        {
            const T pk = rv_[k];
            const T ft = qq_[pk] - 1;
            nffct += (0!=ft);
            if ( !shortq || ft )
            {
                const T nt = g_.num_edges(pk);
                const T pn = rv_[k+1];
                const T tt = g_.edge_idx(pk, pn);
                std::cout << std::setw(4) << k << ":";
                std::cout << " " << std::setw(4) << pk << " ->" << std::setw(4) << pn;
                std::cout << "  [" << std::setw(2) << ft;
                std::cout << " " << std::setw(2) << tt;
                std::cout << " / " << std::setw(2) << nt << "]";
                std::cout << std::endl;
            }
        }
        std::cout << "Path: #non-first-free turns = " << nffct;
        if ( 0==nffct )  std::cout << "  (lucky path)";
        std::cout << std::endl;
    }

    // Return 0 if path is a lucky path,
    // else return 1+k where k is the index where
    //  the edge used was not the first free edge.
    [[nodiscard]] T test_lucky_path()  const noexcept  {
        for (T k=0; k<ng_-1; ++k) {
            if ( qq_[rv_[k]] - 1 ) { return  k+1; }
        }
        return  0;
    }

    /// appends node `p` to the path of length `ns`
    /// \return false if `p` is not a node, the path is full or there is no
    /// 	edge from the last node to `p`.
    bool mark(const T p, T &ns) noexcept {
        if ( p>=ng_ )  return false;
        if ( ns>=ng_ )  return false;
        if ( 0!=ns )
        {
            bool ha = graph().has_edge(rv_[ns-1], p);
            if ( false==ha )  return false;
        }
        rv_[ns] = p;
        qq_[p] = 1;
        ++ns;
        return true;
    }

    /// Let path start as (a canonical monotonic Gray path), the graph
    /// must be `make_gray_digraph(n)`.
    /// \return number of positions marked.
    ///
    /// Example for 5 bits: (return==10)
    /// 0:  ..... 0  0
    /// 1:  ....1 1  1
    /// 2:  ...11 2  3
    /// 3:  ...1. 1  2
    /// 4:  ..11. 2  6
    /// 5:  ..1.. 1  4
    /// 6:  .11.. 2  12
    /// 7:  .1... 1  8
    /// 8:  11... 2  24
    /// 9:  1.... 1  16
    T start_monotonic_gray_path(const T n) noexcept {
        init();
        T ns = 0;
        bool ok = mark(0, ns);
        ok &= mark(1, ns);
        if ( n>=2 )
        {
            ok &= mark(3, ns);
            for (T k=3;  k<2*n; ++k)
            {
                T p = rv_[k-2];
                p = T(cryptanalysislib::internal::digraph::bit_rotate_left(p, 1, n));
                ok &= mark(p, ns);
            }
        }
        assert(ok);
        (void)ok;
        return  ns;
    }

    // Print sequence of nodes.
    void print_path() const noexcept {
        for (T k=0; k<ng_; ++k) {
            std::cout << std::setw(4) << k << ":  " << std::setw(4) << rv_[k] << std::endl;
        }
    }

    // Print sequence of nodes both binary and decimal.
    void print_bin_path() const noexcept {
        for (T k=0; k<ng_; ++k) {
            std::cout << std::setw(4) << k << ":  ";
            for (T b = ngbits_; b-- > 0;) {
                std::cout << (((rv_[k] >> b) & 1u) ? '1' : '.');
            }
            std::cout << "  " << std::setw(4) << rv_[k] << std::endl;
        }
    }

    // Horizontally print sequence of nodes in binary.
    void print_bin_horiz_path()  const noexcept {
        for (T b = ngbits_; b-- > 0;) {
            for (T k=0; k<ng_; ++k) {
                std::cout << (((rv_[k] >> b) & 1u) ? '1' : '.');
            }
            std::cout << std::endl;
        }
    }

    /// calls `pfunc` with each full path starting with the `ns` nodes
    /// already in the path, then node `p`.
    /// \return number of paths where pfunc() returned true
    uint64_t all_paths(uint64_t (*pfunc)(const digraph_paths &),
                       const T ns=0,
                       const T p=0,
                       const uint64_t maxnp=0) noexcept {
        pct_ = 0;
        cct_ = 0;
        pfct_ = 0;
        pfunc_ = pfunc;
        pfdone_ = 0;
        maxnp_ = maxnp;
        next_path(ns, p);
        return pfct_;  // Number of paths where pfunc() returned true
    }

private:
    // called by all_paths()
    // ns+1 == how many nodes seen
    // p == position (node we are on)
    void next_path(T ns, const T p) noexcept {
        if ( pfdone_ )  return;
    
        rv_[ns] = p;  // record position
        ++ns;
    
        // all nodes seen ?
        if ( ns==ng_ ) {
            ++pct_;
            cq_ = path_is_cycle();
            if ( cq_ )  ++cct_;
            const uint64_t pq = pfunc_(*this);
            if ( pq )
            {
                ++pfct_;
                if ( maxnp_ && ( pfct_>=maxnp_ ) )  pfdone_ = true;
            }
        } else {
            qq_[p] = 1;  // mark position as seen (else loops lead to errors)
            T fe, en;
            g_.get_edge_idx(p, fe, en);
            T fct = 0;  // count free reachable nodes
            for (T ep=fe; ep<en; ++ep)
            {
                const T t = g_.e_[ep];  // next node
                if ( 0==qq_[t] )  // node free?
                {
                    ++fct;
                    qq_[p] = fct;  // mark position as seen: record turns
                    next_path(ns, t);
                }
            }
            // if ( 0==fct )  { "dead end: this is a U-turn"; }
    
            qq_[p] = 0;  // unmark position
        }
    }

public:
    /// same as `all_paths`, but node `rv[ns]` is only taken if
    /// `cfunc(*this, ns)` returns true.
    uint64_t all_cond_paths(uint64_t (*pfunc)(const digraph_paths &),
                            bool (*cfunc)(digraph_paths &, uint64_t),
                            const T ns=0, const T p=0, const uint64_t maxnp=0) noexcept {
        pct_ = 0;
        cct_ = 0;
        pfct_ = 0;
        pfunc_ = pfunc;
        cfunc_ = cfunc;
        pfdone_ = 0;
        maxnp_ = maxnp;
        next_cond_path(ns, p);
        return pfct_;  // Number of paths where pfunc() returned true
    }

private:
    // called by all_cond_paths()
    // ns+1 == how many nodes seen
    // p == position (node we are on)
    void next_cond_path(T ns, const T p) noexcept {
        if ( pfdone_ )  return;
    
        rv_[ns] = p;  // record position
        ++ns;
    
        // all nodes seen ?
        if ( ns==ng_ ) {
            ++pct_;
            cq_ = path_is_cycle();
            if ( cq_ )  ++cct_;
            const uint64_t pq = pfunc_(*this);
            if ( pq )
            {
                ++pfct_;
                if ( maxnp_ && ( pfct_>=maxnp_ ) )  pfdone_ = true;
            }
        } else {
            qq_[p] = 1;  // mark position as seen (else loops lead to errors)
            T fe, en;
            g_.get_edge_idx(p, fe, en);
            T fct = 0;  // count free reachable nodes
            for (T ep=fe; ep<en; ++ep)
            {
                const T t = g_.e_[ep];  // next node
                if ( 0==qq_[t] )  // node free?
                {
                    rv_[ns] = t;  // for cfunc()
                    if ( cfunc_(*this, ns) )
                    {
                        ++fct;
                        qq_[p] = fct;  // mark position as seen: record turns
                        next_cond_path(ns, t);
                    }
                }
            }
            // if ( 0==fct )  { "dead end: this is a U-turn"; }
    
            qq_[p] = 0;  // unmark position
        }
    }

public:
    /// follows the first free edge at each node.
    /// NOTE: the visit marks are kept, call `init()` before the next search.
    /// \return 1 if a full path was found, else 0
    uint64_t try_lucky_path(T ns=0, T p=0) noexcept {
        pct_ = 0;
        cct_ = 0;
    
        while (true) {
            rv_[ns] = p;  // record position
            ++ns;
            // ns == how many nodes seen
            // p == position (node we are on)
        
            // all nodes seen ?
            if ( ns==ng_ ) {
                cq_ = path_is_cycle();
                if ( cq_ )  ++cct_;
                ++pct_;
                return  pct_;  // ==1
            }

            T fe, en;
            g_.get_edge_idx(p, fe, en);
            bool found = false;
            for (T ep=fe; ep<en; ++ep)
            {
                const T t = g_.e_[ep];  // next node
                if ( 0==qq_[t] )  // first free node is taken as next
                {
                    qq_[p] = 1;
                    p = t;
                    found = true;
                    break;
                }
            }

            if (!found) { return 0; }
        }
    }
};
