#include "element.h"
#include "matrix/fq_matrix.h"
#include "container/fq_vector.h"
#include <cstdio>
using Value = FqNonPackedVector<8, 3, uint8_t>;
using Label = FqNonPackedVector<8, 3, uint8_t>;
using Matrix = FqMatrix<uint8_t, 8, 8, 3>;
using E = Element_T<Value, Label, Matrix>;
int main() {
  E e1, e2, e3; e1.zero(); e2.zero();
  e1.get_value().set(2, 0); e2.get_value().set(1, 0);
  e1.get_label().set(2, 0); e2.get_label().set(1, 0);
  E::sub(e3, e1, e2, 0, 8);
  printf("Element::sub runtime: label[0]=%u (exp 1) value[0]=%u (exp 1)\n", (unsigned)e3.get_label().get(0), (unsigned)e3.get_value().get(0));
  E::sub<0,8>(e3, e1, e2);
  printf("Element::sub<0,8>:    label[0]=%u (exp 1) value[0]=%u (exp 1)\n", (unsigned)e3.get_label().get(0), (unsigned)e3.get_value().get(0));
}
