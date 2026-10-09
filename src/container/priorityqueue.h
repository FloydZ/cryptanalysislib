#if !defined  HAVE_PRIORITYQUEUE_H__
#define       HAVE_PRIORITYQUEUE_H__

#include <cstdint>
// This file is part of the FXT library.
// Copyright (C) 2010, 2011, 2012, 2014, 2016, 2019, 2023 Joerg Arndt
// License: GNU General Public License version 3 or later,
// see the file COPYING.txt in the main directory.


#include <cstddef>

#include "alloc/alloc.h"


//<<
#if  1
// next() is the one with the smallest key
// i.e.  extract_next()  is  extract_min()
#define _CMP_ <
#define _CMPEQ_ <=
#else
// next() is the one with the biggest key
// i.e.  extract_next()  is  extract_max()
#define _CMP_ >
#define _CMPEQ_ >=
#endif
//>>

template <typename Type1, typename Type2,
          typename Allocator1 = cryptanalysislib::allocator<Type1>,
          typename Allocator2 = cryptanalysislib::allocator<Type2>>
class priority_queue
// Priority queue.
// Can grow dynamically.
{
public:
    Allocator1 alloc1_;
    Allocator2 alloc2_;
    // s+1 slots are allocated; slot 0 is unused so that the heap is one-based
    Type1 *t1_;  // time:   t1[1..s]  one-based array!
    Type2 *e1_;  // events: e1[1..s]  one-based array!
    uint64_t s_;    // allocated size (# of elements)
    uint64_t n_;    // current number of events
    uint64_t gq_;   // grow gq elements if necessary, 0 for "never grow"

    priority_queue(const priority_queue&) = delete;
    priority_queue & operator = (const priority_queue&) = delete;

public:
    explicit priority_queue(uint64_t n, uint64_t growq=0)
    {
        s_ = n;
        t1_ = alloc1_.allocate( s_ + 1 );
        e1_ = alloc2_.allocate( s_ + 1 );

        n_ = 0;
        gq_ = growq;
    }

    ~priority_queue()
    {
        alloc1_.deallocate( t1_, s_ + 1 );
        alloc2_.deallocate( e1_, s_ + 1 );
    }

    uint64_t num()  const  { return n_; }

    bool get_next_t(Type1 &t)  const
    {
        if ( n_ == 0 )  return false;
        t = t1_[1];
        return true;
    }

    bool get_next_e(Type2 &e)  const
    {
        if ( n_ == 0 )  return false;
        e = e1_[1];
        return true;
    }

    bool get_next(Type1 &t, Type2 &e)  const
    {
        if ( n_ == 0 )  return false;
        e = e1_[1];
        t = t1_[1];
        return true;
    }

    bool extract_next(Type1 &t, Type2 &e)
    // Extract next event.
    {
        if ( n_ == 0 )  return false;

        t = t1_[1];
        e = e1_[1];
        t1_[1] = t1_[n_];
        e1_[1] = e1_[n_];
        --n_;
        heapify(1);

        return true;
    }

    bool insert(const Type1 &t, const Type2 &e)
    // Insert event e at time t.
    // Return true if successful,
    //   else false (space exhausted and growth disabled).
    {
        if ( n_ >= s_ )
        {
            if ( 0==gq_ )  return false;  // growing disabled
            grow();
        }

        ++n_;
        uint64_t j = n_;
        while ( j > 1 )
        {
            uint64_t k = (j>>1);  // k==parent(j)
            if ( t1_[k] _CMPEQ_ t )  break;
            t1_[j] = t1_[k];  e1_[j] = e1_[k];
            j = k;
        }
        t1_[j] = t;
        e1_[j] = e;

        return true;
    }

    void reschedule_next(Type1 t)
    {
        t1_[1] = t;
        heapify(1);
    }


private:
    void heapify(uint64_t k)
    {
        uint64_t m = k;

    hstart:
        uint64_t l = (k<<1);  // left(k);
        uint64_t r = l + 1;  // right(k);
        if ( (l <= n_) && (t1_[l] _CMP_ t1_[k]) )  m = l;
        if ( (r <= n_) && (t1_[r] _CMP_ t1_[m]) )  m = r;

        if ( m != k )
        {
            const Type1 t = t1_[k];  t1_[k] = t1_[m];  t1_[m] = t;
            const Type2 e = e1_[k];  e1_[k] = e1_[m];  e1_[m] = e;
//            heapify(m);
            k = m;
            goto hstart;  // tail recursion
        }
    }

    void grow()
    {
        uint64_t ns = s_ + gq_;  // new size
        Type1 *t = alloc1_.allocate( ns + 1 );
        Type2 *e = alloc2_.allocate( ns + 1 );
        for (uint64_t i = 1; i <= n_; i++) {
            t[i] = t1_[i];
            e[i] = e1_[i];
        }
        alloc1_.deallocate( t1_, s_ + 1 );
        alloc2_.deallocate( e1_, s_ + 1 );
        t1_ = t;
        e1_ = e;
        s_ = ns;
    }
};
// -------------------------

#undef _CMP_
#undef _CMPEQ_


#endif  // !defined HAVE_PRIORITYQUEUE_H__
