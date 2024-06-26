#ifndef CRYPTANALYSISLIB_UIN128T_H
#define CRYPTANALYSISLIB_UIN128T_H

// random implementation of `__uint128_t`
#ifndef __SIZEOF_INT128__
struct __uint128_t
private:

	uint64_t LOWER, UPPER;

	public:
	// Constructors
	constexpr uint128_t() = default;
	constexpr uint128_t(const uint128_t & rhs) = default;
	constexpr uint128_t(uint128_t && rhs) = default;

	// do not use prefixes (0x, 0b, etc.)
	// if the input string is too long, only right most characters are read
	constexpr uint128_t(const std::string & s, uint8_t base);
	constexpr uint128_t(const char *s, std::size_t len, uint8_t base);

	uint128_t(const bool & b);

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t(const T & rhs)
	    : LOWER(rhs), UPPER(0) {
		if (std::is_signed<T>::value) {
			if (rhs < 0) {
				UPPER = -1;
			}
		}
	}

	template <typename S, typename T, typename = typename std::enable_if <std::is_integral<S>::value && std::is_integral<T>::value, void>::type>
	constexpr uint128_t(const S & upper_rhs, const T & lower_rhs)
	    : LOWER(lower_rhs), UPPER(upper_rhs)
	{}

	//  RHS input args only

	// Assignment Operator
	constexpr uint128_t & operator=(const uint128_t & rhs) = default;
	constexpr uint128_t & operator=(uint128_t && rhs) = default;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpt uint128_t & operator=(const T & rhs){
		UPPER = 0;

		if (std::is_signed<T>::value) {
			if (rhs < 0) {
				UPPER = -1;
			}
		}

		LOWER = rhs;
		return *this;
	}

	uint128_t & operator=(const bool & rhs);

	// Typecast Operators
	constexpr operator bool() const;
	constexpr operator uint8_t() const;
	constexpr operator uint16_t() const;
	constexpr operator uint32_t() const;
	constexpr operator uint64_t() const;

	// Bitwise Operators
	constexpr uint128_t operator&(const uint128_t & rhs) const;

	void export_bits(std::vector<uint8_t> & ret) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t operator&(const T & rhs) const{
		return uint128_t(0, LOWER & (uint64_t) rhs);
	}

	uint128_t & operator&=(const uint128_t & rhs);

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t & operator&=(const T & rhs){
		UPPER = 0;
		LOWER &= rhs;
		return *this;
	}

	uint128_t operator|(const uint128_t & rhs) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t operator|(const T & rhs) const{
		return uint128_t(UPPER, LOWER | (uint64_t) rhs);
	}

	uint128_t & operator|=(const uint128_t & rhs);

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t & operator|=(const T & rhs){
		LOWER |= (uint64_t) rhs;
		return *this;
	}

	uint128_t operator^(const uint128_t & rhs) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t operator^(const T & rhs) const{
		return uint128_t(UPPER, LOWER ^ (uint64_t) rhs);
	}

	uint128_t & operator^=(const uint128_t & rhs);

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t & operator^=(const T & rhs){
		LOWER ^= (uint64_t) rhs;
		return *this;
	}

	constexpr uint128_t operator~() const;

	// Bit Shift Operators
	uint128_t operator<<(const uint128_t & rhs) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t operator<<(const T & rhs) const{
		return *this << uint128_t(rhs);
	}

	uint128_t & operator<<=(const uint128_t & rhs);

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t & operator<<=(const T & rhs){
		*this = *this << uint128_t(rhs);
		return *this;
	}

	uint128_t operator>>(const uint128_t & rhs) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t operator>>(const T & rhs) const{
		return *this >> uint128_t(rhs);
	}

	uint128_t & operator>>=(const uint128_t & rhs);

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t & operator>>=(const T & rhs){
		*this = *this >> uint128_t(rhs);
		return *this;
	}

	// Logical Operators
	constexpr bool operator!() const;
	constexpr bool operator&&(const uint128_t & rhs) const;
	constexpr bool operator||(const uint128_t & rhs) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr bool operator&&(const T & rhs) const{
		return static_cast <bool> (*this && rhs);
	}

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr bool operator||(const T & rhs) const{
		return static_cast <bool> (*this || rhs);
	}

	// Comparison Operators
	constexpr bool operator==(const uint128_t & rhs) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr bool operator==(const T & rhs) const{
		return (!UPPER && (LOWER == (uint64_t) rhs));
	}

	bool operator!=(const uint128_t & rhs) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr bool operator!=(const T & rhs) const{
		return (UPPER | (LOWER != (uint64_t) rhs));
	}

	bool operator>(const uint128_t & rhs) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr bool operator>(const T & rhs) const{
		return (UPPER || (LOWER > (uint64_t) rhs));
	}

	bool operator<(const uint128_t & rhs) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	cosntexpr bool operator<(const T & rhs) const{
		return (!UPPER)?(LOWER < (uint64_t) rhs):false;
	}

	bool operator>=(const uint128_t & rhs) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr bool operator>=(const T & rhs) const{
		return ((*this > rhs) | (*this == rhs));
	}

	bool operator<=(const uint128_t & rhs) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr bool operator<=(const T & rhs) const{
		return ((*this < rhs) | (*this == rhs));
	}

	// Arithmetic Operators
	uint128_t operator+(const uint128_t & rhs) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t operator+(const T & rhs) const{
		return uint128_t(UPPER + ((LOWER + (uint64_t) rhs) < LOWER), LOWER + (uint64_t) rhs);
	}

	uint128_t & operator+=(const uint128_t & rhs);

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t & operator+=(const T & rhs){
		return *this += uint128_t(rhs);
	}

	uint128_t operator-(const uint128_t & rhs) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t operator-(const T & rhs) const{
		return uint128_t((uint64_t) (UPPER - ((LOWER - rhs) > LOWER)), (uint64_t) (LOWER - rhs));
	}

	uint128_t & operator-=(const uint128_t & rhs);

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t & operator-=(const T & rhs){
		return *this = *this - uint128_t(rhs);
	}

	uint128_t operator*(const uint128_t & rhs) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t operator*(const T & rhs) const{
		return *this * uint128_t(rhs);
	}

	uint128_t & operator*=(const uint128_t & rhs);

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t & operator*=(const T & rhs){
		return *this = *this * uint128_t(rhs);
	}

	private:
	std::pair <uint128_t, uint128_t> divmod(const uint128_t & lhs, const uint128_t & rhs) const;
	void ConvertToVector(std::vector<uint8_t> & current, const uint64_t & val) const;

    // do not use prefixes (0x, 0b, etc.)
	// if the input string is too long, only right most characters are read
	constexpr void init(const char * s, std::size_t len, uint8_t base);
	constexpr void _init_hex(const char *s, std::size_t len);
	constexpr void _init_dec(const char *s, std::size_t len);
	constexpr void _init_oct(const char *s, std::size_t len);
	constexpr void _init_bin(const char *s, std::size_t len);

	public:
	constexpr uint128_t operator/(const uint128_t & rhs) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	constexpr uint128_t operator/(const T & rhs) const{
		return *this / uint128_t(rhs);
	}

	constexpr uint128_t & operator/=(const uint128_t & rhs);

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	cosntexpr uint128_t & operator/=(const T & rhs){
		return *this = *this / uint128_t(rhs);
	}

	uint128_t operator%(const uint128_t & rhs) const;

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	uint128_t operator%(const T & rhs) const{
		return *this % uint128_t(rhs);
	}

	uint128_t & operator%=(const uint128_t & rhs);

	template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
	uint128_t & operator%=(const T & rhs){
		return *this = *this % uint128_t(rhs);
	}

	// Increment Operator
	uint128_t & operator++();
	uint128_t operator++(int);

	// Decrement Operator
	uint128_t & operator--();
	uint128_t operator--(int);

	// Nothing done since promotion doesn't work here
	uint128_t operator+() const;

	// two's complement
	uint128_t operator-() const;

	// Get private values
	const uint64_t & upper() const;
	const uint64_t & lower() const;

	// Get bitsize of value
	uint8_t bits() const;

	// Get string representation of value
	std::string str(uint8_t base = 10, const unsigned int & len = 0) const;
};


// useful values
UINT128_T_EXTERN extern const uint128_t uint128_0;
UINT128_T_EXTERN extern const uint128_t uint128_1;

// lhs type T as first arguemnt
// If the output is not a bool, casts to type T

// Bitwise Operators
template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
uint128_t operator&(const T & lhs, const uint128_t & rhs){
	    return rhs & lhs;
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
T & operator&=(T & lhs, const uint128_t & rhs){
	    return lhs = static_cast <T> (rhs & lhs);
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
uint128_t operator|(const T & lhs, const uint128_t & rhs){
	    return rhs | lhs;
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
T & operator|=(T & lhs, const uint128_t & rhs){
	    return lhs = static_cast <T> (rhs | lhs);
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
uint128_t operator^(const T & lhs, const uint128_t & rhs){
	    return rhs ^ lhs;
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
T & operator^=(T & lhs, const uint128_t & rhs){
	    return lhs = static_cast <T> (rhs ^ lhs);
}

// Bitshift operators
UINT128_T_EXTERN uint128_t operator<<(const bool     & lhs, const uint128_t & rhs);
UINT128_T_EXTERN uint128_t operator<<(const uint8_t  & lhs, const uint128_t & rhs);
UINT128_T_EXTERN uint128_t operator<<(const uint16_t & lhs, const uint128_t & rhs);
UINT128_T_EXTERN uint128_t operator<<(const uint32_t & lhs, const uint128_t & rhs);
UINT128_T_EXTERN uint128_t operator<<(const uint64_t & lhs, const uint128_t & rhs);
UINT128_T_EXTERN uint128_t operator<<(const int8_t   & lhs, const uint128_t & rhs);
UINT128_T_EXTERN uint128_t operator<<(const int16_t  & lhs, const uint128_t & rhs);
UINT128_T_EXTERN uint128_t operator<<(const int32_t  & lhs, const uint128_t & rhs);
UINT128_T_EXTERN uint128_t operator<<(const int64_t  & lhs, const uint128_t & rhs);

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
T & operator<<=(T & lhs, const uint128_t & rhs){
	    return lhs = static_cast <T> (uint128_t(lhs) << rhs);
}

UINT128_T_EXTERN uint128_t operator>>(const bool     & lhs, const uint128_t & rhs);
UINT128_T_EXTERN uint128_t operator>>(const uint8_t  & lhs, const uint128_t & rhs);
UINT128_T_EXTERN uint128_t operator>>(const uint16_t & lhs, const uint128_t & rhs);
UINT128_T_EXTERN uint128_t operator>>(const uint32_t & lhs, const uint128_t & rhs);
UINT128_T_EXTERN uint128_t operator>>(const uint64_t & lhs, const uint128_t & rhs);
UINT128_T_EXTERN uint128_t operator>>(const int8_t   & lhs, const uint128_t & rhs);
UINT128_T_EXTERN uint128_t operator>>(const int16_t  & lhs, const uint128_t & rhs);
UINT128_T_EXTERN uint128_t operator>>(const int32_t  & lhs, const uint128_t & rhs);
UINT128_T_EXTERN uint128_t operator>>(const int64_t  & lhs, const uint128_t & rhs);

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
constexpr T & operator>>=(T & lhs, const uint128_t & rhs){
	    return lhs = static_cast <T> (uint128_t(lhs) >> rhs);
}

// Comparison Operators
template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
constexpr bool operator==(const T & lhs, const uint128_t & rhs){
	    return (!rhs.upper() && ((uint64_t) lhs == rhs.lower()));
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
constexpr bool operator!=(const T & lhs, const uint128_t & rhs){
	    return (rhs.upper() | ((uint64_t) lhs != rhs.lower()));
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
constexpr bool operator>(const T & lhs, const uint128_t & rhs){
	    return (!rhs.upper()) && ((uint64_t) lhs > rhs.lower());
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
conxtexpr bool operator<(const T & lhs, const uint128_t & rhs){
	    if (rhs.upper()){
		    return true;
	    }
	    return ((uint64_t) lhs < rhs.lower());
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
constexpr bool operator>=(const T & lhs, const uint128_t & rhs){
	    if (rhs.upper()){
		    return false;
	    }
	    return ((uint64_t) lhs >= rhs.lower());
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
constexpr bool operator<=(const T & lhs, const uint128_t & rhs){
	    if (rhs.upper()){
		    return true;
	    }
	    return ((uint64_t) lhs <= rhs.lower());
}

// Arithmetic Operators
template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
constexpr uint128_t operator+(const T & lhs, const uint128_t & rhs){
	    return rhs + lhs;
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
constexpr T & operator+=(T & lhs, const uint128_t & rhs){
	    return lhs = static_cast <T> (rhs + lhs);
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
constexpr uint128_t operator-(const T & lhs, const uint128_t & rhs){
	    return -(rhs - lhs);
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
constexpr T & operator-=(T & lhs, const uint128_t & rhs){
	    return lhs = static_cast <T> (-(rhs - lhs));
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
constexpr uint128_t operator*(const T & lhs, const uint128_t & rhs){
	    return rhs * lhs;
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
constexpr T & operator*=(T & lhs, const uint128_t & rhs){
	    return lhs = static_cast <T> (rhs * lhs);
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
constexpr uint128_t operator/(const T & lhs, const uint128_t & rhs){
	    return uint128_t(lhs) / rhs;
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
constexpr T & operator/=(T & lhs, const uint128_t & rhs){
	    return lhs = static_cast <T> (uint128_t(lhs) / rhs);
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
constexpr uint128_t operator%(const T & lhs, const uint128_t & rhs){
	    return uint128_t(lhs) % rhs;
}

template <typename T, typename = typename std::enable_if<std::is_integral<T>::value, T>::type >
constexpr T & operator%=(T & lhs, const uint128_t & rhs){
	    return lhs = static_cast <T> (uint128_t(lhs) % rhs);
}

// IO Operator
constexpr UINT128_T_EXTERN std::ostream & operator<<(std::ostream & stream, const uint128_t & rhs);
#endif
#endif//CRYPTANALYSISLIB_UIN128T_H
