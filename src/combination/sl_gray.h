#pragma once

/// Class for generating binary words in SL-Gray order (subset-lex Gray code)
/// Implements a minimal-change order related to subset-lex order
/// Successive transitions are mostly adjacent (one-close),
/// and otherwise have distance 3 (three-close)
/// Uses a loopless algorithm for efficient generation
///
/// References:
/// - See comb/binary-sl-gray.h for generation in an array
/// - See bits/bitlex.h for subset-lex order
class bit_sl_gray {
public:
    /// Current Gray code word
    ulong x_;
    
    /// Current track (a one-bit word) that controls bit changes
    ulong tr_;
    
    /// Highest allowed track position
    ulong h_;
    
    /// Direction track tries to move in: true means try to move right
    bool dt_;

public:
    /// Constructor initializes the SL-Gray code generator
    /// \param n[in]: number of bits in the words to generate
    explicit bit_sl_gray(ulong n)  { first(n); }
    
    /// Destructor
    ~bit_sl_gray()  { ; }

    /// Sets the generator to the first word in the SL-Gray sequence
    /// \param n[in]: number of bits in the words to generate
    void first(ulong n) {
        tr_ = 1UL << (n-1);
        h_ = tr_;
        dt_ = true;
        x_ = 0;

        if ( n == 0 ) {
            // Special case handling for n==0
            h_ = 0;
            tr_ = 1;
            dt_ = false;
        }
    }

    /// Gets the current SL-Gray code word
    /// \return current Gray code word
    ulong data()  const  { return x_; }

    /// Advances to the next word in the SL-Gray sequence
    /// The algorithm is "loopless" meaning it has constant time per word
    /// \return the next SL-Gray code word, or 0 if at the end of the sequence
    ulong next() {
        if ( dt_ ) {
            // Try to append trailing ones (moving right)
            if ( (x_ & tr_) == 0 ) {
                // Bit not set at current track
                x_ ^= tr_;  // Set bit
                if ( tr_>>1 )  tr_ >>= 1;  // Move right unless already at end
            } else {
                // Change bit one track left
                dt_ = false;
                x_ ^= (tr_<<1);
                if ( tr_ >= h_ )  return  0;  // Current is last (only for n==1)
            }
        } else {
            // Try to remove trailing ones (moving left)
            if ( (x_ & (tr_<<1)) != 0 ) {
                // Left bit is set
                x_ ^= tr_;  // Remove bit on track
                tr_ <<= 1;  // Move left
            } else  {
                // Change bit three tracks left, move right
                if ( h_ <= tr_<<1 )  return 0;  // Current is last
                dt_ = true;
                x_ ^= (tr_ << 2);
                if ( tr_>>1 )  tr_ >>= 1;
            }
        }
        return x_;
    }
};
