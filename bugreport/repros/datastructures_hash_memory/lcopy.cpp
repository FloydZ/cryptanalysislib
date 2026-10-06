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
  List in{10,1}, out{10,1};
  Matrix m; m.random(); in.random(10, m); out.zero();
  printf("copying: list::copy will memcpy %zu Elements (=%zu bytes) into a buffer of %zu bytes\n", 10*sizeof(Value), 10*sizeof(Value)*sizeof(Element), 10*sizeof(Element));
  fflush(stdout);
  List::copy(out, in, 0);
  printf("survived; out.load()=%zu (expected 10)\n", out.load());
}
