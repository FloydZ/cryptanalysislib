#pragma once

// TODO 
template<class ForwardIt, 
class Generator>
constexpr 
void generate(ForwardIt first, 
              ForwardIt last, 
              Generator g) {
    for (; first != last; ++first)
        *first = g();
}
