#pragma once

#include <cstdlib>
#include "alloc/alloc.h"

// whether to include some O(n) methods for testing purposes
// TODO actual use them for tests
//#define LR_WITH_DUMB_METHODS  // default off


// Maintain index array [0,...,n-1], keep track which index is set or free.
// Allows, in time O(log(n)), to
// - find k-th free index (where 0<=k<=num_free())
// - find k-th set index (where 0<=k<=num_set())
// - determine how many indices are free/set to the left/right of
//   an absolute index i (where 0<=i<n).
template <template<class> class Allocator=cryptanalysislib::allocator>
class left_right_array {
private:
    size_t *fl_;  // Free indices Left (including current element) in bsearch interval
    bool *tg_;   // tags: tg[i]==true if and only if index i is free
    size_t n_;    // total number of indices
    size_t f_;    // number of free indices

    left_right_array(const left_right_array&) = delete;
    left_right_array & operator = (const left_right_array&) = delete;

public:
    explicit left_right_array(const size_t n) {
        n_ = n;
        // TODO allocator
        fl_ = new size_t[n_];
        tg_ = new bool[n_];
        free_all();
    }

    ~left_right_array()
    {
        delete [] fl_;
        delete [] tg_;
    }

    size_t num_free() const  { return f_; }
    size_t num_set() const  { return  n_ - f_; }
    bool is_free(size_t i) const  { return  tg_[i]; }
    bool is_set(size_t i) const  { return  ! tg_[i]; }

private:
    void init_rec(size_t i0, size_t i1)
    // Set elements of fl[0,...,n-2] according to all indices free.
    // The element fl[n-1] needs to be set to 1 afterwards.
    // Work is O(n).
    {
        if ( (i1-i0)!=0 )
        {
            size_t t = (i1+i0)/2;
            init_rec(i0, t);
            init_rec(t+1, i1);
        }
        fl_[i1] = i1-i0+1;
    }

public:
    // Mark all indices as free.
    void free_all() {
        f_ = n_;
        for (size_t j=0; j<n_; ++j)  tg_[j] = true;
        init_rec(0, n_-1);
        fl_[n_-1] = 1;
    }

    // Mark all indices of as set.
    void set_all() {
        f_ = 0;
        for (size_t j=0; j<n_; ++j)  tg_[j] = false;
        for (size_t j=0; j<n_; ++j)  fl_[j] = 0;
    }

    // Return the k-th ( 0 <= k < num_free() ) free index.
    // Return ~0UL if k is out of bounds.
    // Work is O(log(n)).
    size_t get_free_idx(size_t k) const noexcept {
        if ( k >= num_free() )  return ~0UL;

        size_t i0 = 0,  i1 = n_-1;
        while ( 1 ) {
            size_t t = (i1+i0)/2;
            if ( (fl_[t] == k+1) && (tg_[t]) )  return t;

            if ( fl_[t] > k )  // left:
            {
                i1 = t;
            } else  {
                i0 = t+1;  k-=fl_[t];
            }
        }
    }

    // Return the k-th ( 0 <= k < num_free() ) free index.
    // Return ~0UL if k is out of bounds.
    // Change the arrays and fl[] and tg[] reflecting
    //   that index i will be set afterwards.
    // Work is O(log(n)).
    size_t get_free_idx_chg(size_t k)    {
        if ( k >= num_free() )  return ~0UL;

        --f_;

        size_t i0 = 0,  i1 = n_-1;
        while ( 1 )
        {
            size_t t = (i1+i0)/2;

            if ( (fl_[t] == k+1) && (tg_[t]) )
            {
                --fl_[t];
                tg_[t] = false;
                return t;
            }

            if ( fl_[t] > k )  // left:
            {
                --fl_[t];
                i1 = t;
            }
            else    // right:
            {
                i0 = t+1;  k-=fl_[t];
            }
        }
    }


    size_t get_set_idx(size_t k)  const
    // Return the k-th ( 0 <= k < num_set() ) set index.
    // Return ~0UL if k is out of bounds.
    // Work is O(log(n)).
    {
        if ( k >= num_set() )  return ~0UL;

        size_t i0 = 0,  i1 = n_-1;
        while ( 1 )
        {
            size_t t = (i1+i0)/2;
            // how many elements to the left are set:
            size_t slt = t-i0+1 - fl_[t];

            if ( (slt == k+1) && (tg_[t]==false) )  return t;

            if ( slt > k )  // left:
            {
                i1 = t;
            }
            else   // right:
            {
                i0 = t+1;  k-=slt;
            }
        }
    }

    size_t get_set_idx_chg(size_t k)
    // Return the k-th ( 0 <= k < num_set() ) set index.
    // Return ~0UL if k is out of bounds.
    // Change the arrays and fl[] and tg[] reflecting
    //   that index i will be freed afterwards.
    // Work is O(log(n)).
    {
        if ( k >= num_set() )  return ~0UL;

        ++f_;

        size_t i0 = 0,  i1 = n_-1;
        while ( 1 )
        {
            size_t t = (i1+i0)/2;
            // how many elements to the left are set:
            size_t slt = t-i0+1 - fl_[t];

            if ( (slt == k+1) && (tg_[t]==false) )
            {
                ++fl_[t];
                tg_[t] = true;
                return t;
            }

            if ( slt > k )  // left:
            {
                ++fl_[t];
                i1 = t;
            }
            else   // right:
            {
                i0 = t+1;  k-=slt;
            }
        }
    }


    // The methods num_[FS][LR][IE](size_t i) return the number of
    // Free/Set indices Left/Right if (absolute) index i, Including/Excluding i.
    // Return ~0UL if i >= n.

    size_t num_FLE(size_t i)  const
    // Return number of Free indices Left of (absolute) index i (Excluding i).
    // Work is O(log(n)).
    {
        if ( i >= n_ )  { return ~0UL; }  // out of bounds

        size_t i0 = 0,  i1 = n_-1;
        size_t ns = i;  // number of set element left to i (including i)
        while ( 1 )
        {
            if ( i0==i1 )  break;

            size_t t = (i1+i0)/2;
            if ( i<=t )  // left:
            {
                i1 = t;
            }
            else   // right:
            {
                ns -= fl_[t];
                i0 = t+1;
            }
        }

        return  i-ns;
    }

    size_t num_FLI(size_t i)  const
    // Return number of Free indices Left of (absolute) index i (Including i).
    {
        if ( i >= n_ )  { return ~0UL; }
        return num_FLE(i) + tg_[i];
    }


    size_t num_FRE(size_t i)  const
    // Return number of Free indices Right of (absolute) index i (Excluding i).
    {
        if ( i >= n_ )  { return ~0UL; }
        return  num_free() - num_FLI(i);
    }

    size_t num_FRI(size_t i)  const
    // Return number of Free indices Right of (absolute) index i (Including i).
    {
        if ( i >= n_ )  { return ~0UL; }
        return  num_free() - num_FLE(i);
    }


    size_t num_SLE(size_t i)  const
    // Return number of Set indices Left of (absolute) index i (Excluding i).
    {
        if ( i >= n_ )  { return ~0UL; }
        return i - num_FLE(i);
    }

    size_t num_SLI(size_t i)  const
    // Return number of Set indices Left of (absolute) index i (Including i).
    {
        if ( i >= n_ )  { return ~0UL; }
        return i - num_FLE(i) + !tg_[i];
    }


    size_t num_SRE(size_t i)  const
    // Return number of Set indices Right of (absolute) index i (Excluding i).
    {
        if ( i >= n_ )  { return ~0UL; }
        return  num_set() - num_SLI(i);
    }

    size_t num_SRI(size_t i)  const
    // Return number of Set indices Right of (absolute) index i (Including i).
    {
        if ( i >= n_ )  { return ~0UL; }
        return  num_set() - i + num_FLE(i);
    }


#if defined LR_WITH_DUMB_METHODS   // Work with all methods *_dumb() is O(n).
    size_t get_free_idx_dumb(size_t k)  const
    // Return the k-th ( 0 <= k < num_free() ) free index.
    // Return ~0UL if k is out of bounds.
    {
        if ( k >= num_free() )  return ~0UL;

        size_t idx = 0;
        for ( ; idx<n_; ++idx)
        {
            if ( tg_[idx]==true )
            {
                if ( k==0 )  break;
                --k;
            }
        }
        return idx;
    }

    size_t get_set_idx_dumb(size_t k)  const
    // Return the k-th ( 0 <= k < num_set() ) set index.
    // Return ~0UL if k is out of bounds.
    {
        if ( k >= num_set() )  return ~0UL;

        size_t idx = 0;
        for ( ; idx<n_; ++idx)
        {
            if ( tg_[idx]==false )
            {
                if ( k==0 )  break;
                --k;
            }
        }
        return idx;
    }


    size_t num_FLE_dumb(size_t i)  const
    // Return number of Free indices Left of (absolute) index i (Excluding i).
    {
        if ( i >= n_ )  { return ~0UL; }
        size_t nf = 0;
        for (size_t j=0; j<i; ++j)  nf += tg_[j];
        return nf;
    }

    size_t num_FLI_dumb(size_t i)  const
    // Return number of Free indices Left of (absolute) index i (Including i).
    {
        if ( i >= n_ )  { return ~0UL; }
        size_t nf = 0;
        for (size_t j=0; j<=i; ++j)  nf += tg_[j];
        return nf;
    }


    size_t num_FRE_dumb(size_t i)  const
    // Return number of Free indices Right of (absolute) index i (Excluding i).
    {
        if ( i >= n_ )  { return ~0UL; }
        size_t nf = 0;
        for (size_t j=i+1; j<n_; ++j)  nf += tg_[j];
        return nf;
    }

    size_t num_FRI_dumb(size_t i)  const
    // Return number of Free indices Right of (absolute) index i (Including i).
    {
        if ( i >= n_ )  { return ~0UL; }
        size_t nf = 0;
        for (size_t j=i; j<n_; ++j)  nf += tg_[j];
        return nf;
    }


    size_t num_SLE_dumb(size_t i)  const
    // Return number of Set indices Left of (absolute) index i (Excluding i).
    {
        if ( i >= n_ )  { return ~0UL; }
        size_t nf = 0;
        for (size_t j=0; j<i; ++j)  nf += !tg_[j];
        return nf;
    }

    size_t num_SLI_dumb(size_t i)  const
    // Return number of Set indices Left of (absolute) index i (Including i).
    {
        if ( i >= n_ )  { return ~0UL; }
        size_t nf = 0;
        for (size_t j=0; j<=i; ++j)  nf += !tg_[j];
        return nf;
    }


    size_t num_SRE_dumb(size_t i)  const
    // Return number of Set indices Right of (absolute) index i (Excluding i).
    {
        if ( i >= n_ )  { return ~0UL; }
        size_t nf = 0;
        for (size_t j=i+1; j<n_; ++j)  nf += !tg_[j];
        return nf;
    }

    size_t num_SRI_dumb(size_t i)  const
    // Return number of Set indices Right of (absolute) index i (Including i).
    {
        if ( i >= n_ )  { return ~0UL; }
        size_t nf = 0;
        for (size_t j=i; j<n_; ++j)  nf += !tg_[j];
        return nf;
    }

#endif  // LR_WITH_DUMB_METHODS
};
// -------------------------
