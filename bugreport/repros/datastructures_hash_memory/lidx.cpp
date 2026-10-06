#include <cstdio>
#include "container/fq_vector.h"
#include "matrix/matrix.h"
#include "list/parallel_index.h"
constexpr uint32_t k = 20, n = 20, q = 5;
using Matrix = FqMatrix<uint8_t, n, k, q>;
using Value  = FqNonPackedVector<k, q, uint8_t>;
using Label  = FqNonPackedVector<n, q, uint8_t>;
using Element = Element_T<Value, Label, Matrix>;
using List = Parallel_List_IndexElement_T<Element, 2>;
int main(){
  List L{100,4};
  for (uint32_t t=0;t<4;t++) printf("[index list] tid=%u start_pos=%zu end_pos=%zu (exp %u..%u)\n", t, L.start_pos(t), L.end_pos(t), t*25, (t+1)*25);
  Label a, b; a.zero(); b.zero(); a.data()[0]=1;
  uint64_t l1=0, l2=0;
  L.add_and_append(a, b, 7, 8, l1, 1);   // thread 1 should write at index 25
  L.add_and_append(b, b, 9, 9, l2, 0);   // thread 0 writes at index 0
  printf("[index list] __data[0].second={%u,%u} (thread0 wrote 9,9) __data[25].second={%u,%u} (thread1 expected 7,8)\n", L.__data[0].second[0], L.__data[0].second[1], L.__data[25].second[0], L.__data[25].second[1]);
}
