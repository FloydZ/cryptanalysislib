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
        // ep_ = new ulong[ng_+1];
        // e_ = new ulong[ne];
        // if ( vnq )  vn_ = new ulong[ng_];
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
        const ulong *e = e_ + fe;
        for (ulong t=0; t<nt; ++t) { 
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
    void  sort_edges(int (*cmp)(const ulong &, const ulong &)) {
        // value == index (in e[])
        if ( nullptr==vn_ )  {
            for (ulong k=0; k<ng_; ++k) {
                ulong x = ep_[k];
                ulong n = ep_[k+1] - x;
                selection_sort(e_+x, n, cmp);
            }
        } else {
            for (ulong k=0; k<ng_; ++k) {
                ulong x = ep_[k];
                ulong n = ep_[k+1] - x;
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
            for (ulong k=0; k<ng_; ++k) {
                ulong x = ep_[k];
                ulong n = ep_[k+1] - x;
                if ( ! is_sorted(e_+x, n, cmp) )  return k;
            }
        } else {
            for (ulong k=0; k<ng_; ++k) {
                ulong x = ep_[k];
                ulong n = ep_[k+1] - x;
                if ( ! is_idx_sorted(vn_, n, e_+x, cmp) )  return k;
            }
        }
        return ng_;
    }

    constexpr bool is_edge_sorted(int (*cmp)(const ulong &, const ulong &)) const noexcept {
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
    void randomize_edge_order(ulong p0, ulong p1=0) noexcept {
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
        for (ulong k=0; k<ng_; ++k) {
            std::cout << std::setw(3) << k << ":  ";
            for (ulong j=ep_[k]; j<ep_[k+1]; ++j) {
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
        ulong ma = max_edges();
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

    // Initialization for the complete graph.
    static digraph *make_complete_digraph(T n) noexcept {
        T ng = n, ne = n*(n-1);
    
        ulong *ep, *e;
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

    static digraph *make_debruijn_digraph(ulong n) noexcept {
        ulong ng = 2*n, ne = 2*ng;
        ulong *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e);
    
        ulong j = 0;
        for (ulong k=0; k<ng; ++k)  // for all nodes
        {
            ep[k] = j;
            ulong r = (2*k) % ng;
            e[j++] = r;  // connect node k to node (2*k) mod ng
            r = (2*k+1) % ng;
            e[j++] = r;  // connect node k to node (2*k+1) mod ng
        }
        ep[ng] = j;
    
        return  dgp;
    }
    
    
    static digraph * make_complement_shift_digraph(ulong n)
    {
        ulong ng = 2*n, ne = 2*ng;
        ulong *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e);
    
        ulong i = 0;
        for (ulong k=0; k<ng; ++k)  // for all nodes
        {
            ep[k] = i;
            ulong r = (2*k) % ng;
            e[i++] = r;  // connect node k to node (2*k) mod ng
            r = (2*k+1) % ng;
            e[i++] = r;  // connect node k to node (2*k+1) mod ng
        }
        ep[ng] = i;
        // Here we have a De Bruijn graph.
    
        for (ulong k=0, j=ng-1;  k<j;  ++k, --j) swap2(e[ep[k]], e[ep[j]]);  // end with ones
        for (ulong k=0, j=ng-1;  k<j;  ++k, --j) swap2(e[ep[k]+1], e[ep[j]+1]);
    
        return  dgp;
    }
    // -------------------------
    
    
    digraph *
    make_debruijn_digraph(ulong n, ulong m)
    // m-ary version
    {
        ulong ng = m*n, ne = m*ng;
        ulong *ep, *e;
        digraph * dgp = new digraph(ng, ne, ep, e);
    
        ulong j = 0;
        for (ulong k=0; k<ng; ++k)  // for all nodes
        {
            ep[k] = j;
            for (ulong i=0; i<m; ++i)
            {
                ulong r = (m*k+i) % ng;
                e[j++] = r;  // connect node k to node (m*k+j) mod ng
            }
        }
        ep[ng] = j;
    
        return  dgp;
    }


digraph *
make_fibrepgray_digraph(ulong n)
{
    ulong *f = new ulong[n];
    for (ulong k=0; k<n; ++k)  f[k] = bin2fibrep(k);

    ulong nc = 0;
    for (ulong k=0; k<n; ++k)
    {
        ulong fk = f[k];
        for (ulong j=0; j<n; ++j)
        {
            if ( j==k )  continue;
            ulong fj = f[j];
            if ( one_bit_q( fj^fk ) )  ++nc;
        }
    }

    ulong *ep, *e;
    digraph * dgp = new digraph(n, nc, ep, e, 1);
    digraph &dg = *dgp;
    acopy(f, dg.vn_, n);

    ulong tnc = 0;
    for (ulong k=0; k<n; ++k)
    {
        ep[k] = tnc;
        ulong fk = f[k];
        for (ulong j=0; j<n; ++j)
        {
            if ( j==k )  continue;
            ulong fj = f[j];
            if ( one_bit_q( fj^fk ) )  e[tnc++] = j;
        }
    }
//    jjassert( nc == tnc );
    ep[n] = tnc;


    delete [] f;

    return  dgp;
}

digraph *
make_gray_digraph(ulong n, bool rq/*=0*/)
// Initialization for directed graph:
// Gray code graph for n-bit words.
{
    ulong ng = 1UL<<n;

    ulong ne = ng * n;  // number of edges
    ulong *ep, *e;
    digraph * dgp = new digraph(ng, ne, ep, e);

    ulong p = 0;
    ulong k = 0;
    if ( rq )  // force path to start as 0 1 3:
    {
        ep[k] = p;  e[p++] = 1;  ++k;  // 0 --> 1
        ep[k] = p;  e[p++] = 3;  ++k;  // 1 --> 3
    }

    for (  ; k<ng; ++k)  // for all nodes
    {
        ep[k] = p;
        for (ulong c=0, b=1;  c<n;  ++c, b<<=1)
        {
            ulong vc = k ^ b;  // change one bit
            e[p++] = vc;
        }
    }
    ep[ng] = p;

    return  dgp;
}
// -------------------------


ulong
start_monotonic_gray_path(digraph_paths &dp, ulong n)
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
    for (ulong k=0; k<dp.ng_; ++k)  dp.qq_[k] = 0;
    ulong ns = 0;
    jjassert( dp.mark(0, ns) );
    jjassert( dp.mark(1, ns) );
    if ( n>=2 )
    {
        jjassert( dp.mark(3, ns) );
        ulong *rv = dp.rv_;
        for (ulong k=3;  k<2*n; ++k)
        {
            ulong p = rv[k-2];
            p = bit_rotate_left(p, 1, n);
            jjassert( dp.mark(p, ns) );
        }
    }
    return  ns;
}


static digraph *
make_mtl_digraph(ulong k, bool rq/*=0*/)
// Initialization for the "middle two levels" graph
{
    ulong k2 = 2*k-1;
    ulong ng = 2*binomial(k2, k);
    ulong ne = ng * k;  // number of edges
    if ( rq )  ne -= (k-1);

    ulong *ep, *e;
//    digraph dg(ng, ne, ep, e, true);
    digraph * dgp = new digraph(ng, ne, ep, e, true);
    digraph &dg = *dgp;

    ulong *vn = dg.vn_;
    ulong mask = first_comb(k2);
    ulong comb = first_comb(k);
    ulong nct = 0;  // Node counter
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

    ulong p = 0;
    ulong j = 0;
    if ( rq )  // force path to start "canonically":
    {
        ulong x = k;
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
        ulong v = vn[j];  // value of node
        for (ulong b=1;  0!=(b & mask);  b<<=1)
        {
            ulong vc = v ^ b;  // change one bit
            ulong x = bsearch(vn, ng, vc);
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


constexpr static ulong Catalan[]=
{
    0UL, 1UL, 2UL, 5UL, 14UL, 42UL, 132UL, 429UL, 1430UL, 4862UL, 16796UL,
    58786UL, 208012UL, 742900UL, 2674440UL, 9694845UL, 35357670UL
//    129644790UL, 477638700UL, 1767263190UL, 6564120420UL };
};
// -------------------------

static bool
parengray_is_neighbor(ulong fk, ulong fj, ulong pcd, ulong /*nb*/)
{
    ulong xr = fj^fk;
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
make_parengray_digraph(ulong nb, ulong pcd)
{
    ulong n = Catalan[nb];
    ulong *f = new ulong[n];
    {
        ulong k = 0;
        ulong c = last_comb(nb, 2*nb);
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

    ulong nc = 0;
    for (ulong k=0; k<n; ++k)  // count number of edges
    {
        ulong fk = f[k];
        for (ulong j=0; j<n; ++j)
        {
            if ( j==k )  continue;
            ulong fj = f[j];
            if ( parengray_is_neighbor(fk, fj, pcd, nb) )  ++nc;
        }
    }

    ulong *cp = new ulong[n+1];
    ulong *c = new ulong[nc];
    nc = 0;
    for (ulong k=0; k<n; ++k)  // fill in edges
    {
        cp[k] = nc;
        ulong fk = f[k];
        for (ulong j=0; j<n; ++j)
        {
            if ( j==k )  continue;
            ulong fj = f[j];
            if ( parengray_is_neighbor(fk, fj, pcd, nb) )  c[nc++] = j;
        }
    }
    cp[n] = nc;




//    digraph(ulong ng, ulong ne, ulong *&ep, ulong *&e, bool vnq=false)
    ulong *ep, *e;
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

static inline void star_swap(ulong *x, ulong c)
{
    // star transpositions:
    swap2( x[0], x[c] );
}
// -------------------------


static inline void adj_swap(ulong *x, ulong c)
{
    // adjacent transpositions:
    swap2(x[c-1], x[c]);
}
// -------------------------

digraph *
make_perm_gray_digraph(ulong n, bool stq)
// Initialization for directed graph:
// Gray code permutations of n elements
// with star transpositions if stq==true,
// otherwise with adjacent changes.
{
    ulong ng = factorial(n);
    ulong ne = ng * (n-1);  // number of edges
    ulong *ep, *e;
    digraph * dgp = new digraph(ng, ne, ep, e);

    ulong xx[32];  // permutations
    ulong p = 0;
    for (ulong k=0; k<ng; ++k)  // for all nodes
    {
        ep[k] = p;
        num2perm_rfact(k, xx, n);

        for (ulong j=1;  j<n;  ++j)
//        for (ulong j=n-1;  j!=0;  --j)
        {
            if ( stq ) star_swap(xx, j);
            else       adj_swap(xx, j);

            ulong vc = perm2num_rfact(xx, n);
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
make_perm_pref_rev_digraph(ulong n)
// Initialization for directed graph:
// permutations are connected by prefix reversals
{
    ulong ng = factorial(n);

    ulong ne = ng * (n-1);  // number of edges
    ulong *ep, *e;
    digraph * dgp = new digraph(ng, ne, ep, e);


    ulong xx[32];  // aux: permutations
    ulong yy[32];  // aux: prefix-reversed permutations
    ulong p = 0;
    for (ulong k=0; k<ng; ++k)  // for all nodes
    {
        ep[k] = p;

        num2perm_ffact(k, xx, n);
        for (ulong j=2;  j<=n;  ++j)
        {
            for (ulong i=0; i<n; ++i)  yy[i] = xx[i];
            reverse(yy, j);

            ulong vc = perm2num_ffact(yy, n);
            e[p++] = vc;
        }
    }
    ep[ng] = p;

    return  dgp;
}

digraph *
make_perm_pref_rot_digraph(ulong n, bool rq/*=0*/)
// Initialization for directed graph:
// permutations are connected by prefix rotations,
// rq = 1 ==> right rotations, otherwise left rotations.
{
    ulong ng = factorial(n);

    ulong ne = ng * (n-1);  // number of edges
    ulong *ep, *e;
    digraph * dgp = new digraph(ng, ne, ep, e);


    ulong xx[32];  // aux: permutations
    ulong yy[32];  // aux: prefix-reversed permutations
    ulong p = 0;
    for (ulong k=0; k<ng; ++k)  // for all nodes
    {
        ep[k] = p;

        num2perm_ffact(k, xx, n);
        for (ulong j=2;  j<=n;  ++j)
        {
            for (ulong i=0; i<n; ++i)  yy[i] = xx[i];
            if ( rq ) rotate_right1(yy, j);
            else      rotate_left1(yy, j);

            ulong vc = perm2num_ffact(yy, n);
            e[p++] = vc;
        }
    }
    ep[ng] = p;

    return  dgp;
}




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
