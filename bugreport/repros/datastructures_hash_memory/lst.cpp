#include <cstdio>
#include "container/fq_vector.h"
#include "matrix/matrix.h"
#include "list/common.h"
constexpr uint32_t k = 20, n = 20, q = 5;
using Matrix = FqMatrix<uint8_t, n, k, q>;
using Value  = FqNonPackedVector<k, q, uint8_t>;
using Label  = FqNonPackedVector<n, q, uint8_t>;
using Element = Element_T<Value, Label, Matrix>;
using List = MetaListT<Element>;
int main(){
  // 1. copy() with threads=1
  {
    List in{10,1}, out{10,1};
    Matrix m; m.random();
    in.random(10, m);  // load=10
    out.zero();
    List::copy(out, in, 0);
    size_t same=0; for (size_t i=0;i<10;i++) same += in[i].is_equal(out[i]);
    printf("[copy] out.load()=%zu (exp %zu), equal elements=%zu/10, sizeof(Element)=%zu sizeof(Value)=%zu\n", out.load(), in.load(), same, sizeof(Element), sizeof(Value));
  }
  // 2. size(tid) with threads=3, size=10
  {
    List L{10,3};
    size_t s=0; for (uint32_t t=0;t<3;t++){ s+=L.size(t); printf("[size(tid)] tid=%u size=%zu start=%zu end=%zu\n",t,L.size(t),L.start_pos(t),L.end_pos(t)); }
    printf("[size(tid)] sum=%zu (exp 10)\n", s);
  }
  // 3. random(m, tid)
  {
    List L{12,2}; Matrix m; m.random(); L.zero(0); L.zero(1);
    L.random(m, 0); L.random(m, 1);
    printf("[random(m,tid)] load(0)=%zu load(1)=%zu (exp 6,6); L[0].is_zero=%d\n", L.load(0), L.load(1), (int)L[0].is_zero());
  }
  // 4. is_sorted(t, ...) ignores last element
  {
    List L{4,1}; L.zero();
    for (size_t i=0;i<4;i++){ L[i].label.zero(); }
    L[0].label.data()[0]=1; L[1].label.data()[0]=2; L[2].label.data()[0]=3; L[3].label.data()[0]=0; // last out of order
    L.set_load(4);
    Label t; t.zero();
    printf("[is_sorted] plain=%d  with t=0 -> %d (both should be 0)\n", (int)L.is_sorted(), (int)L.is_sorted(t));
  }
}
