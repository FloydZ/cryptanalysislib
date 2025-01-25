#pragma once 



// Fibonacci Gray code with binary words.
template<typename T>
class bit_fibgray {
public:
    ulong x_;  // current Fibonacci word
    ulong k_;  // aux
    ulong fw_, lw_;  // first and last Fibonacci word in Gray code
    ulong mw_;  // max(fw_, lw_)
    ulong n_;   // Number of bits

public:
    explicit bit_fibgray(ulong n) {
        n_ = n;

        fw_ = 0;
        for (ulong m=(1UL<<(n-1)); m!=0; m>>=3)  fw_ |= m;
        lw_ = fw_ >> 1;
        if ( 0==(n&1) )  { ulong t=fw_; fw_=lw_; lw_=t; }  // swap first/last
        mw_ = ( lw_>fw_ ? lw_ : fw_ );
        x_ = fw_;

        k_ = inverse_gray_code(fw_);
        k_ = neg2bin(k_);
    }

    ~bit_fibgray()  { ; }

    ulong data()  const  { return x_; }

    // Return next word in Gray code.
    // Return ~0 if current word is the last one.
    ulong next() {
        if ( x_ == lw_ )  return ~0UL;

        ulong s = n_;  // shift
        while(1) {
            --s;
            ulong c = 1 | (mw_ >> s);  // possible difference for negbin word
            ulong i = k_ - c;
            ulong x = bin2neg(i);
            x ^= (x>>1);

            if ( 0==(x&(x>>1))) {  // is_fibrep(x)
                k_ = i;
                x_ = x;
                return x;
            }
        }
    }
};
