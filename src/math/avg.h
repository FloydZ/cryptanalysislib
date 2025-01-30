#pragma once

namespace cryptanalysislib::math {

/// original from: https://www.jjj.de/fxt/fxtpage.html
/// Result is correct even if (x+y) wouldn't fit into a T
/// Use:      x+y == ((x&y)<<1) + (x^y)
/// that is:  sum ==  carries   + sum_without_carries
/// \param x[in]:
/// \param y[in]:
/// \return floor((x+y)/2) 
template<typename T>
constexpr static inline T floor_average(T x,
                                        T y) {
    return  (x & y) + ((x ^ y) >> 1);
    // return  y + ((x-y)>>1);  // works if x>=y
}

/// Result is correct even if (x+y) wouldn't fit into a T
/// Use:      x+y == ((x|y)<<1) - (x^y)
/// ceil_average(x,y) == average(x,y) + ((x^y)&1))
/// \param x[in]:
/// \param y[in]:
/// \return floor((x+y)/2) 
template<typename T>
constexpr static inline T ceil_average(T x,
                                       T y) {
    return  (x | y) - ((x ^ y) >> 1);
}

}; // end namespace cryptanalysislib::math
