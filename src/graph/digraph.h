#pragma once 

#include <iostream>
#include "alloc/alloc.h"

// Directed graph with ng nodes.
// Initialization just allocates memory,
//  filling in the edges is left to the user.
template <typename T=uint32_t, 
          class Allocator = cryptanalysislib::allocator<T>>
class digraph {
private:
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

    digraph(const digraph&) = delete;
    digraph & operator = (const digraph&) = delete;

public:
    /// \param ng
    explicit digraph(const T ng,
                     const T ne,
                     const T *&ep,
                     const T *&e,
                     bool vnq=false) noexcept
    : ng_(0), ep_(nullptr), e_(nullptr), vn_(nullptr) {
        ng_ = ng;
        ep = ep_;
        e = e_;
        ep_ = allocator.allocate(ng_ + 1u);
        e_ = allocator.allocate(ne);
        if ( vnq ) { vn_ = allocator.allocate(ng_); }
        // ep_ = new uint64_t[ng_+1];
        // e_ = new uint64_t[ne];
        // if ( vnq )  vn_ = new uint64_t[ng_];
    }

    ~digraph() noexcept {
        allocator.deallocate(ep_, ng_+1u);
        allocator.deallocate(e_, 1);
        if (vn_) { allocator.deallocate(vn_, ng_); }
        //delete [] ep_;
        //delete [] e_;
        //if ( vn_ )  delete [] vn_;
    }


    [[nodiscard]] constexpr T num_nodes() const noexcept { return ng_; }
    [[nodiscard]] constexpr T num_edges() const noexcept { return ep_[num_nodes()]; }

    // Return how many outgoing edges are at node p.
    constexpr T num_edges(T p) const noexcept {
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
    //   ~0UL  else
    constexpr T edge_idx(const T p,
                         const T pn) const noexcept {
        T fe = ep_[p];   // (index of) First Edge
        T nt = num_edges(p);
        const uint64_t *e = e_ + fe;
        for (uint64_t t=0; t<nt; ++t) { 
            if (pn==e[t]) { 
                return t; 
            }
        }
        return  ~0UL;  // pn cannot be reached from p
    }

    // Return whether edge from p to pn exists
    [[nodiscard]] constexpr bool has_edge(const T p,
                                          const T pn) const noexcept  {
        return (edge_idx(p, pn) < num_edges(p)); 
    }

    // Return max number (among all nodes) of outgoing edges.
    constexpr T max_edges() const noexcept {
        T ma = 0;  // max number of outgoing edges
        for (T k=0; k<ng_; ++k) {
            T n = ep_[k+1] - ep_[k];
            if (n > ma) {
                ma = n;
            }
        }
        return  ma;
    }

    /// \param rq[in]:
    void sort_edges(int rq=1) noexcept {
        if (rq) sort_edges(cmp0);
        else    sort_edges(cmp1);
    }
    void  sort_edges(int (*cmp)(const uint64_t &, const uint64_t &)) {
        // value == index (in e[])
        if ( nullptr==vn_ )  {
            for (uint64_t k=0; k<ng_; ++k) {
                uint64_t x = ep_[k];
                uint64_t n = ep_[k+1] - x;
                selection_sort(e_+x, n, cmp);
            }
        } else {
            for (uint64_t k=0; k<ng_; ++k) {
                uint64_t x = ep_[k];
                uint64_t n = ep_[k+1] - x;
                idx_selection_sort(vn_, n, e_+x, cmp);
            }
        }
    }

    // Test for each node whether sets of outgoing edges are sorted.
    // If the test fails for a node, return its index,
    //  else return ng.
    constexpr T test_edge_sorted(int (*cmp)(const T &, const T &)) const noexcept  {
        // value == index (in e[])
        if ( nullptr==vn_ ) {
            for (uint64_t k=0; k<ng_; ++k) {
                uint64_t x = ep_[k];
                uint64_t n = ep_[k+1] - x;
                if ( ! is_sorted(e_+x, n, cmp) )  return k;
            }
        } else {
            for (uint64_t k=0; k<ng_; ++k) {
                uint64_t x = ep_[k];
                uint64_t n = ep_[k+1] - x;
                if ( ! is_idx_sorted(vn_, n, e_+x, cmp) )  return k;
            }
        }
        return ng_;
    }

    constexpr bool is_edge_sorted(int (*cmp)(const uint64_t &, const uint64_t &)) const noexcept {
        return ( ng_==test_edge_sorted(cmp));
    }

    // Reverse order of edges at positions p0,...,p1.
    // If p1==0 then action is performed just for position p0.
    constexpr void reverse_edge_order(const T p0,
                                      const T p1=0) noexcept {
        T p = p0;
        do {
            T n = num_edges(p);
            if (n > 1) { reverse(e_+ep_[p], n); }
        } while ( ++p<=p1 );  // note: inclusive p1
    }

    constexpr void reverse_edge_order() noexcept {
        reverse_edge_order(0, ng_-1); 
    }

    // Random permute order of edges at positions p0,...,p1.
    // If p1==0 then action is performed just for position p0.
    void randomize_edge_order(uint64_t p0, uint64_t p1=0) noexcept {
        T p = p0;
        do {
            T n = num_edges(p);
            if (n > 1) { random_permute(e_+ep_[p], n); }
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
        for (uint64_t k=0; k<ng_; ++k) {
            std::cout << std::setw(3) << k << ":  ";
            for (uint64_t j=ep_[k]; j<ep_[k+1]; ++j) {
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
        uint64_t ma = max_edges();
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

    constexpr T test() const noexcept {
        T ng = ng_;
        for (T k=0; k<ng; ++k)  if ( ep_[k] > ep_[k+1] )  return 1;
    
        const T ne = num_edges();
        for (T k=0; k<ng; ++k)  if ( ep_[k] >= ne  )  return 2;
    
        if (ep_[ng] != ne) {
            return 3;
        }
    
        for (T k=0; k<ne; ++k) {
            if (e_[k] >= ng_) { 
                return 10; 
            }
        }
    
        return 0;
    }

    constexpr bool OK() const noexcept { 
        return (0==test()); 
    }

    /// Initialization for the complete graph.
    /// \param n[in] 
    static digraph *make_complete_digraph(T n) noexcept {
        T ng = n, ne = n*(n-1);
    
        uint64_t *ep, *e;
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

    /// \param n[in] 
    static digraph *make_debruijn_digraph(T n) noexcept {
        uint64_t ng = 2*n, ne = 2*ng;
        uint64_t *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e);
    
        uint64_t j = 0;
        for (uint64_t k=0; k<ng; ++k)  // for all nodes
        {
            ep[k] = j;
            uint64_t r = (2*k) % ng;
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
        for (uint64_t k=0; k<ng; ++k)  {
            ep[k] = j;
            for (T i=0; i<m; ++i) {
                T r = (m*k+i) % ng;
                e[j++] = r;  // connect node k to node (m*k+j) mod ng
            }
        }
        ep[ng] = j;
    
        return  dgp;
    }


    static digraph *make_fibrepgray_digraph(const T n) noexcept {
        // TODO allocator
        T *f = new T[n];
        for (uint64_t k=0; k<n; ++k) { f[k] = bin2fibrep(k); }
    
        uint64_t nc = 0;
        for (uint64_t k=0; k<n; ++k) {
            uint64_t fk = f[k];
            for (uint64_t j=0; j<n; ++j) {
                if ( j==k )  continue;
                uint64_t fj = f[j];
                if ( one_bit_q( fj^fk ) )  ++nc;
            }
        }
    
        uint64_t *ep, *e;
        digraph * dgp = new digraph(n, nc, ep, e, 1);
        digraph &dg = *dgp;
        acopy(f, dg.vn_, n);
    
        uint64_t tnc = 0;
        for (uint64_t k=0; k<n; ++k)
        {
            ep[k] = tnc;
            uint64_t fk = f[k];
            for (uint64_t j=0; j<n; ++j)
            {
                if ( j==k )  continue;
                uint64_t fj = f[j];
                if ( one_bit_q( fj^fk ) )  e[tnc++] = j;
            }
        }
    //    jjassert( nc == tnc );
        ep[n] = tnc;
    
    
        delete [] f;
    
        return  dgp;
    }
    
    digraph *
    make_gray_digraph(uint64_t n, bool rq/*=0*/)
    // Initialization for directed graph:
    // Gray code graph for n-bit words.
    {
        uint64_t ng = 1UL<<n;
    
        uint64_t ne = ng * n;  // number of edges
        uint64_t *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e);
    
        uint64_t p = 0;
        uint64_t k = 0;
        if ( rq )  // force path to start as 0 1 3:
        {
            ep[k] = p;  e[p++] = 1;  ++k;  // 0 --> 1
            ep[k] = p;  e[p++] = 3;  ++k;  // 1 --> 3
        }
    
        for (  ; k<ng; ++k)  // for all nodes
        {
            ep[k] = p;
            for (uint64_t c=0, b=1;  c<n;  ++c, b<<=1)
            {
                uint64_t vc = k ^ b;  // change one bit
                e[p++] = vc;
            }
        }
        ep[ng] = p;
    
        return  dgp;
    }
    // -------------------------
    
    
    uint64_t
    start_monotonic_gray_path(digraph_paths &dp, uint64_t n)
    // Let path start as (a canonical monotonic Gray path):
    //
    // Return number of positions marked.
    //
    // Example for 5 bits: (return==10)
    // 0:  ..... 0  0
    // 1:  ....1 1  1
    // 2:  ...11 2  3
    // 3:  ...1. 1  2
    // 4:  ..11. 2  6
    // 5:  ..1.. 1  4
    // 6:  .11.. 2  12
    // 7:  .1... 1  8
    // 8:  11... 2  24
    // 9:  1.... 1  16
    {
        for (uint64_t k=0; k<dp.ng_; ++k)  dp.qq_[k] = 0;
        uint64_t ns = 0;
        jjassert( dp.mark(0, ns) );
        jjassert( dp.mark(1, ns) );
        if ( n>=2 )
        {
            jjassert( dp.mark(3, ns) );
            uint64_t *rv = dp.rv_;
            for (uint64_t k=3;  k<2*n; ++k)
            {
                uint64_t p = rv[k-2];
                p = bit_rotate_left(p, 1, n);
                jjassert( dp.mark(p, ns) );
            }
        }
        return  ns;
    }
    
    
    static digraph *
    make_mtl_digraph(uint64_t k, bool rq/*=0*/)
    // Initialization for the "middle two levels" graph
    {
        uint64_t k2 = 2*k-1;
        uint64_t ng = 2*binomial(k2, k);
        uint64_t ne = ng * k;  // number of edges
        if ( rq )  ne -= (k-1);
    
        uint64_t *ep, *e;
    //    digraph dg(ng, ne, ep, e, true);
        digraph * dgp = new digraph(ng, ne, ep, e, true);
        digraph &dg = *dgp;
    
        uint64_t *vn = dg.vn_;
        uint64_t mask = first_comb(k2);
        uint64_t comb = first_comb(k);
        uint64_t nct = 0;  // Node counter
        do
        {
            vn[nct++] = comb;
            jjassert( nct < ng );
            vn[nct++] = mask & ~comb;
            comb = next_colex_comb(comb);
        }
        while ( comb < mask );
        jjassert( nct == ng );
    
        quick_sort(vn, ng);
    
        uint64_t p = 0;
        uint64_t j = 0;
        if ( rq )  // force path to start "canonically":
        {
            uint64_t x = k;
            ep[j] = p;  e[p++] = x;  ++j;  // 0000111 --> 0001111
    //        print_bin(" 2nd= ", vn[x], pbn);  cout << endl;
            // #0   == 0000111
            // #1   == 0001011
            // #2   == 0001101
            // #3   == 0001110
            // #k+1 == 0001111
        }
    
        for (  ; j<ng; ++j)  // for all nodes
        {
            ep[j] = p;
            uint64_t v = vn[j];  // value of node
            for (uint64_t b=1;  0!=(b & mask);  b<<=1)
            {
                uint64_t vc = v ^ b;  // change one bit
                uint64_t x = bsearch(vn, ng, vc);
                if ( ng != x )
                {
                    jjassert( p<ne );
                    e[p++] = x;
                }
            }
        }
        ep[ng] = p;
        jjassert( p==ne );
    
        return  dgp;
    }
    
    
    constexpr static uint64_t Catalan[]=
    {
        0UL, 1UL, 2UL, 5UL, 14UL, 42UL, 132UL, 429UL, 1430UL, 4862UL, 16796UL,
        58786UL, 208012UL, 742900UL, 2674440UL, 9694845UL, 35357670UL
    //    129644790UL, 477638700UL, 1767263190UL, 6564120420UL };
    };
    // -------------------------
    
    static bool
    parengray_is_neighbor(uint64_t fk, uint64_t fj, uint64_t pcd, uint64_t /*nb*/)
    {
        uint64_t xr = fj^fk;
        bool q = false;
        if ( 2==bit_count( xr ) )
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
            default:  jjassert(0);  // criterion does not exist;
            }
        }
    
        return q;
    }
    // -------------------------
    
    
    digraph *
    make_parengray_digraph(uint64_t nb, uint64_t pcd)
    {
        uint64_t n = Catalan[nb];
        uint64_t *f = new uint64_t[n];
        {
            uint64_t k = 0;
            uint64_t c = last_comb(nb, 2*nb);
            do
            {
                if ( is_parenword(c) )
                {
                    f[k++] = c;
    //                jjassert( k<=n );
    //                if ( k>=n )  break;
                }
            }
            while ( (c = prev_colex_comb(c)) );
            jjassert( k==n );
            reverse(f, n);
        }
    
        uint64_t nc = 0;
        for (uint64_t k=0; k<n; ++k)  // count number of edges
        {
            uint64_t fk = f[k];
            for (uint64_t j=0; j<n; ++j)
            {
                if ( j==k )  continue;
                uint64_t fj = f[j];
                if ( parengray_is_neighbor(fk, fj, pcd, nb) )  ++nc;
            }
        }
    
        uint64_t *cp = new uint64_t[n+1];
        uint64_t *c = new uint64_t[nc];
        nc = 0;
        for (uint64_t k=0; k<n; ++k)  // fill in edges
        {
            cp[k] = nc;
            uint64_t fk = f[k];
            for (uint64_t j=0; j<n; ++j)
            {
                if ( j==k )  continue;
                uint64_t fj = f[j];
                if ( parengray_is_neighbor(fk, fj, pcd, nb) )  c[nc++] = j;
            }
        }
        cp[n] = nc;
    
    
    
    
    //    digraph(uint64_t ng, uint64_t ne, uint64_t *&ep, uint64_t *&e, bool vnq=false)
        uint64_t *ep, *e;
    //    digraph dg(n, nc, ep, e, 1);
        digraph *dgp = new digraph(n, nc, ep, e, 1);
        digraph &dg = *dgp;
    
        acopy(f, dg.vn_, n);
        acopy(c, dg.e_, nc);
        acopy(cp, dg.ep_, n+1);
    
        delete [] c;
        delete [] cp;
        delete [] f;
    
        return  dgp;
    }
    
    static inline void star_swap(uint64_t *x, uint64_t c)
    {
        // star transpositions:
        swap2( x[0], x[c] );
    }
    // -------------------------
    
    
    static inline void adj_swap(uint64_t *x, uint64_t c)
    {
        // adjacent transpositions:
        swap2(x[c-1], x[c]);
    }
    // -------------------------
    
    digraph *
    make_perm_gray_digraph(uint64_t n, bool stq)
    // Initialization for directed graph:
    // Gray code permutations of n elements
    // with star transpositions if stq==true,
    // otherwise with adjacent changes.
    {
        uint64_t ng = factorial(n);
        uint64_t ne = ng * (n-1);  // number of edges
        uint64_t *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e);
    
        uint64_t xx[32];  // permutations
        uint64_t p = 0;
        for (uint64_t k=0; k<ng; ++k)  // for all nodes
        {
            ep[k] = p;
            num2perm_rfact(k, xx, n);
    
            for (uint64_t j=1;  j<n;  ++j)
    //        for (uint64_t j=n-1;  j!=0;  --j)
            {
                if ( stq ) star_swap(xx, j);
                else       adj_swap(xx, j);
    
                uint64_t vc = perm2num_rfact(xx, n);
                e[p++] = vc;
    
                // unswap:
                if ( stq ) star_swap(xx, j);
                else       adj_swap(xx, j);
            }
        }
        ep[ng] = p;
    
        return  dgp;
    }
    
    digraph *
    make_perm_pref_rev_digraph(uint64_t n)
    // Initialization for directed graph:
    // permutations are connected by prefix reversals
    {
        uint64_t ng = factorial(n);
    
        uint64_t ne = ng * (n-1);  // number of edges
        uint64_t *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e);
    
    
        uint64_t xx[32];  // aux: permutations
        uint64_t yy[32];  // aux: prefix-reversed permutations
        uint64_t p = 0;
        for (uint64_t k=0; k<ng; ++k)  // for all nodes
        {
            ep[k] = p;
    
            num2perm_ffact(k, xx, n);
            for (uint64_t j=2;  j<=n;  ++j)
            {
                for (uint64_t i=0; i<n; ++i)  yy[i] = xx[i];
                reverse(yy, j);
    
                uint64_t vc = perm2num_ffact(yy, n);
                e[p++] = vc;
            }
        }
        ep[ng] = p;
    
        return  dgp;
    }
    
    digraph *
    make_perm_pref_rot_digraph(uint64_t n, bool rq/*=0*/)
    // Initialization for directed graph:
    // permutations are connected by prefix rotations,
    // rq = 1 ==> right rotations, otherwise left rotations.
    {
        uint64_t ng = factorial(n);
    
        uint64_t ne = ng * (n-1);  // number of edges
        uint64_t *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e);
    
    
        uint64_t xx[32];  // aux: permutations
        uint64_t yy[32];  // aux: prefix-reversed permutations
        uint64_t p = 0;
        for (uint64_t k=0; k<ng; ++k)  // for all nodes
        {
            ep[k] = p;
    
            num2perm_ffact(k, xx, n);
            for (uint64_t j=2;  j<=n;  ++j)
            {
                for (uint64_t i=0; i<n; ++i)  yy[i] = xx[i];
                if ( rq ) rotate_right1(yy, j);
                else      rotate_left1(yy, j);
    
                uint64_t vc = perm2num_ffact(yy, n);
                e[p++] = vc;
            }
        }
        ep[ng] = p;
    
        return  dgp;
    }
    
    
    
    
    // Find all full paths in a directed graph.
    class digraph_paths {
    public:
        digraph &g_;  // the graph
        uint64_t *rv_;  // Record of Visits: rv[k] == node visited at step k
        uint64_t *qq_;  // qq[k] == whether node k has been visited yet
    
        uint64_t pct_;  // count Paths
        uint64_t cct_;  // count Cycles
        uint64_t pfct_;  // count Paths where pfunc() returns 1
    
        bool cq_;  // whether current path is a cycle
    
        bool pany_;    // whether to print anything (set automatically)
        uint64_t ng_;  // == g_.ng_
        uint64_t ngbits_;  // number of bits in ng_, used for printing
    
        // function to call with each path found with all_paths():
        uint64_t (*pfunc_)(const digraph_paths &);
    
        bool pfdone_;  // if set (by pfunc()) then search is stopped
        uint64_t maxnp_;  // stop after maxnp times that pfunc returned one (0==forever)
    
        // function to impose condition with all_cond_paths():
        bool (*cfunc_)(digraph_paths &, uint64_t ns);  // can set pfdone_
    
        digraph_paths(const digraph_paths&) = delete;
        digraph_paths & operator = (const digraph_paths&) = delete;
    
    public:
        // graph/digraph.cc:
        explicit digraph_paths(digraph &g);
        ~digraph_paths();
        void init();
    
        const digraph & graph()  const  { return g_; }
    
        bool path_is_cycle()  const;
    
        void print_turns(bool shortq=true) const;
        uint64_t test_lucky_path()  const;
    
        bool mark(uint64_t p, uint64_t &ns);
    
        void print_path() const
        // Print sequence of nodes.
        { ::print_path(rv_, ng_); }
    
        void print_bin_path() const
        // Print sequence of nodes both binary and decimal.
        { ::print_bin_path(rv_, ng_, ngbits_); }
    
        void print_bin_horiz_path()  const
        // Horizontally print sequence of nodes in binary.
        { ::print_bin_horiz_path(rv_, ng_, ngbits_); }
    
    
        // graph/search-digraph.cc:
    public:
        uint64_t all_paths(uint64_t (*pfunc)(const digraph_paths &),
                        uint64_t ns=0, uint64_t p=0, uint64_t maxnp=0);
    private:
        void next_path(uint64_t ns, uint64_t p);  // called by all_paths()
    
        // graph/search-digraph-cond.cc:
    public:
        uint64_t all_cond_paths(uint64_t (*pfunc)(const digraph_paths &),
                             bool (*cfunc)(digraph_paths &, uint64_t),
                             uint64_t ns=0, uint64_t p=0, uint64_t maxnp=0);
    private:
        void next_cond_path(uint64_t ns, uint64_t p);  // called by all_cond_paths()
    
        // graph/search-digraph-trylucky.cc:
    public:
        uint64_t try_lucky_path(uint64_t ns=0, uint64_t p=0);
    private:
        void next_lucky(uint64_t ns, uint64_t p);  // called by try_lucky_path()
    };
private:
    /// \param a[in]:
    /// \param b[in]:
    /// \return 0: a == b 
    ///        -1: a > b 
    ///         1: a < b
    constexpr static inline int cmp1(const T &a,
                                     const T &b) noexcept {
        if ( a==b )  return 0;
        if ( a<b )  return +1;
        else        return -1;
    }
    
    /// \param a[in]:
    /// \param b[in]:
    /// \return 0: a == b 
    ///        -1: a < b 
    ///         1: a > b
    constexpr static inline int cmp0(const T &a,
                                     const T &b) noexcept {
        return -cmp1(a, b);
    }
};


// Find all full paths in a directed graph.
template <typename T=uint32_t, 
          class Allocator = cryptanalysislib::allocator<T>>
class digraph_paths {
private:
    Allocator allocator;

    // the graph
    digraph &g_; 

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

    // whether to print anything (set automatically)
    bool pany_ = 0;
    
    // == g_.ng_
    T ng_;

    // number of bits in ng_, used for printing
    T ngbits_ = 0;

    // function to call with each path found with all_paths():
    uint64_t (*pfunc_)(const digraph_paths &);

    // if set (by pfunc()) then search is stopped
    bool pfdone_ = 0;  

    // stop after maxnp times that pfunc returned one (0==forever)
    size_t maxnp_ = 0;

    // function to impose condition with all_cond_paths():
    bool (*cfunc_)(digraph_paths &, uint64_t ns);  // can set pfdone_

    digraph_paths(const digraph_paths&) = delete;
    digraph_paths & operator = (const digraph_paths&) = delete;

public:
    // graph/digraph.cc:
    explicit digraph_paths(digraph &g)  noexcept :
        g_(g), ng_(g_.ng_) {
        // rv_ = new T[ng_];
        // qq_ = new T[ng_];
        rv_ = allocator.allocate(ng_);
        qq_ = allocator.allocate(ng_);
        ngbits_ = next_exp_of_2(ng_);
        pfunc_ = nullptr;
        cryptanalysislib::memset(qq_, 0, ng_);
    }

    ~digraph_paths() noexcept {
        allocator.deallocate(rv_, ng_);
        allocator.deallocate(qq_, ng_);
    }

    constexpr const digraph & graph() const noexcept { return g_; }

    // Return whether the path is a cycle.
    constexpr bool path_is_cycle()  const noexcept {
        // first node visited
        uint64_t p0 = rv_[0];
        
        // last node visited
        uint64_t p = rv_[ng_-1];  
        return graph().has_edge(p, p0);
    }

    void print_turns(bool shortq=true) const {
        cout << "Path:";
        if ( shortq )  cout << " (short print) ";
        cout << endl;
        uint64_t nffct = 0;  // count non-first-free turns
        for (uint64_t k=0; k<ng_-1; ++k)
        {
            uint64_t pk = rv_[k];
            uint64_t ft = qq_[pk] - 1;
            nffct += (0!=ft);
            if ( !shortq || ft )
            {
                uint64_t nt = g_.num_edges(pk);
                uint64_t pn = rv_[k+1];
                uint64_t tt = g_.edge_idx(pk, pn);
                cout << setw(4) << k << ":";
                cout << " " << setw(4) << pk << " ->" << setw(4) << pn;
                cout << "  [" << setw(2) << ft;
                cout << " " << setw(2) << tt;
                cout << " / " << setw(2) << nt << "]";
                cout << endl;
            }
        }
        cout << "Path: #non-first-free turns = " << nffct;
        if ( 0==nffct )  cout << "  (lucky path)";
        cout << endl;
    }

    // Return 0 if path is a lucky path,
    // else return 1+k where k is the index where
    //  the edge used was not the first free edge.
    T test_lucky_path()  const noexcept  {
        for (T k=0; k<ng_-1; ++k) {
            if ( qq_[rv_[k]] - 1 ) { return  k+1; }
        }
        return  0;
    }

    bool mark(uint64_t p, uint64_t &ns) noexcept {
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

    void print_path() const
    // Print sequence of nodes.
    { ::print_path(rv_, ng_); }

    void print_bin_path() const
    // Print sequence of nodes both binary and decimal.
    { ::print_bin_path(rv_, ng_, ngbits_); }

    void print_bin_horiz_path()  const
    // Horizontally print sequence of nodes in binary.
    { ::print_bin_horiz_path(rv_, ng_, ngbits_); }


    // graph/search-digraph.cc:
public:
    uint64_t all_paths(uint64_t (*pfunc)(const digraph_paths &),
                    uint64_t ns=0,
                    uint64_t p=0,
                    uint64_t maxnp=0) noexcept {
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
    void next_path(uint64_t ns, uint64_t p) noexcept {
        if ( pfdone_ )  return;
    
        rv_[ns] = p;  // record position
        ++ns;
    
        // all nodes seen ?
        if ( ns==ng_ ) {
            ++pct_;
            cq_ = path_is_cycle();
            if ( cq_ )  ++cct_;
            uint64_t pq = pfunc_(*this);
            if ( pq )
            {
                ++pfct_;
                if ( maxnp_ && ( pfct_>=maxnp_ ) )  pfdone_ = true;
            }
        } else {
            qq_[p] = 1;  // mark position as seen (else loops lead to errors)
            uint64_t fe, en;
            g_.get_edge_idx(p, fe, en);
            uint64_t fct = 0;  // count free reachable nodes
            for (uint64_t ep=fe; ep<en; ++ep)
            {
                uint64_t t = g_.e_[ep];  // next node
                if ( 0==qq_[t] )  // node free?
                {
                    ++fct;
                    qq_[p] = fct;  // mark position as seen: record turns
    //                jjassert( fct>=1 );
                    next_path(ns, t);
                }
            }
            // if ( 0==fct )  { "dead end: this is a U-turn"; }
    
            qq_[p] = 0;  // unmark position
        }
    }
    // graph/search-digraph-cond.cc:
public:
    uint64_t all_cond_paths(uint64_t (*pfunc)(const digraph_paths &),
                         bool (*cfunc)(digraph_paths &, uint64_t),
                         uint64_t ns=0, uint64_t p=0, uint64_t maxnp=0) {
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
    void next_cond_path(uint64_t ns, uint64_t p) {
        if ( pfdone_ )  return;
    
        rv_[ns] = p;  // record position
        ++ns;
    
        // all nodes seen ?
        if ( ns==ng_ ) {
            ++pct_;
            cq_ = path_is_cycle();
            if ( cq_ )  ++cct_;
            uint64_t pq = pfunc_(*this);
            if ( pq )
            {
                ++pfct_;
                if ( maxnp_ && ( pfct_>=maxnp_ ) )  pfdone_ = true;
            }
        } else {
            qq_[p] = 1;  // mark position as seen (else loops lead to errors)
            uint64_t fe, en;
            g_.get_edge_idx(p, fe, en);
            uint64_t fct = 0;  // count free reachable nodes
            for (uint64_t ep=fe; ep<en; ++ep)
            {
                uint64_t t = g_.e_[ep];  // next node
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
    uint64_t try_lucky_path(uint64_t ns=0, uint64_t p=0){
        pct_ = 0;
        cct_ = 0;
        // TODO init();
    
     start:
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
        } else {
            uint64_t fe, en;
            g_.get_edge_idx(p, fe, en);
            for (uint64_t ep=fe; ep<en; ++ep)
            {
                uint64_t t = g_.e_[ep];  // next node
                if ( 0==qq_[t] )  // first free node is taken as next
                {
                    qq_[p] = 1;
                    p = t;
                    goto start;
                }
            }
            return 0;
        }
    
    //    return 0;  // never reached
    }
};
