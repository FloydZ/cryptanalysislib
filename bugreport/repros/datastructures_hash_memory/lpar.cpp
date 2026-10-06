#include <cstdio>
#include "container/fq_vector.h"
#include "matrix/matrix.h"
#include "list/parallel.h"
constexpr uint32_t k = 20, n = 20, q = 5;
using Matrix = FqMatrix<uint8_t, n, k, q>;
using Value  = FqNonPackedVector<k, q, uint8_t>;
using Label  = FqNonPackedVector<n, q, uint8_t>;
using Element = Element_T<Value, Label, Matrix>;
using List = Parallel_List_T<Element>;
int main(){
  List L{100,4};
  for (uint32_t t=0;t<4;t++) printf("[Parallel_List_T] tid=%u start_pos=%zu end_pos=%zu (exp %u..%u)\n", t, L.start_pos(t), L.end_pos(t), t*25, (t+1)*25);
}
