#ifndef CRYPTANALYSISLIB_CRYPTO_SHA1_H
#define CRYPTANALYSISLIB_CRYPTO_SHA1_H

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <array>
#include <bit>

#include "memory/memory.h"
#include "algorithm/rotate.h"


//TODO all 3 algorithms move somewhere useful
/// Converts an array of integers to an array of bytes with a specified endianness.
///
/// @tparam endianness   The desired byte order.
/// @tparam data_t       The type of integers in the array.
/// @tparam num_elements The number of integerts in the array.
///
/// @param value The array of integers to convert to bytes.
///
/// @returns The converted array.
template <std::endian endianness,
          typename data_t,
          std::size_t num_elements> requires std::unsigned_integral<data_t>
consteval std::array<std::byte, sizeof(data_t) * num_elements> 
to_bytes(const std::array<data_t, num_elements>& value) noexcept {
    constexpr std::size_t bits_per_byte = 8;
    std::array<std::byte, sizeof(data_t) * num_elements> result{};
    for (auto current_byte = result.begin(); const data_t& element : value) {
        for (std::size_t byte_index = 0; byte_index < sizeof(data_t); ++byte_index, ++current_byte) {
            std::size_t shift = endianness == std::endian::little ? byte_index * bits_per_byte
                                                                  : (sizeof(data_t) - byte_index - 1) * bits_per_byte;
            *current_byte = static_cast<std::byte>((element & (data_t{0xff} << shift)) >> shift);
        }
    }

    return result;
}


/// Allows easier, more readable declaration of std::array<std::byte> using a string literal where the string literal is
/// interpreted as hex digits.
///
/// @tparam char_t The type of characters in the string literal. Only char is supported here.
/// @tparam chars  The characters in the string literal.
///
/// @returns A byte array representing the provided hexadecmal string.
///
/// @throws std::invalid_argument if the input string is malformed.
template <typename char_t, char_t... chars> requires (sizeof...(chars) >= 2 && sizeof...(chars) % 2 == 0)
static constexpr std::array<std::byte, sizeof...(chars) / 2> operator "" _hex_bytes() {
    constexpr auto hex2val = [](char c) constexpr {
        if (c >= '0' && c <= '9')
            return c - '0';
        else if (c >= 'a' && c <= 'f')
            return c - 'a' + static_cast<char>(0xa);
        else if (c >= 'A' && c <= 'F')
            return c - 'A' + static_cast<char>(0xa);
        else
            throw std::invalid_argument("Character is not a hex digit.");
    };

    // Convert the characters pairwise into bytes.
    const std::array<char, sizeof...(chars)> char_array{chars...};
    std::array<std::byte, sizeof...(chars) / 2> bytes{};
    for (std::size_t i = 0; i < char_array.size(); i += 2)
        bytes.at(i / 2) = static_cast<std::byte>(hex2val(char_array.at(i)) << 4 | hex2val(char_array.at(i + 1)));
    return bytes;
}

/// Makes a byte array from a string literal.
///
/// @tparam char_t The type of characters in the string literal. Only char is supported here.
/// @tparam chars  The characters in the string literal.
///
/// @returns A byte array representing the provided string.
template <typename char_t, char_t... chars>
static constexpr std::array<std::byte, sizeof...(chars)> operator "" _bytes() {
    return std::array<std::byte, sizeof...(chars)>{std::byte{chars}...};
}


namespace cryptanalysislib {
	namespace internal {
		// output size in bytes
		constexpr size_t SHA1_DIGEST_LEN = 20;

		// same as `SHA1_DIGEST_LEN` only as number of limbs of type `uint32_t`
		constexpr size_t SHA1_DIGEST_UINT32_LEN = SHA1_DIGEST_LEN/4;

		// length of the internal state in bytes
		constexpr size_t SHA1_STATE_LEN = 32;

		// length of the CBLOCK in bytes
		constexpr size_t SHA1_CBLOCK_LEN = 64;

		constexpr size_t SHA1_ROUNDS = 80;


        /// The "Ch(x, y, z)" function defined in FIPS 180-4 sections 4.1.1, 4.1.2, and 4.1.3.
        /// For each bit i in words x, y, and z if x[i] is set then result[i]
        /// is y[i], otherwise result[i] is z[i]. In other words the bit in x 
        /// "chooses" if the result bit comes from y or z.
        /// \tparam word_t The type of number being used.
        template <typename T> 
        constexpr T choose(const T x, const T y, const T z) noexcept {
            return (x & y) ^ (~x & z);
        }
        
        /// The "Maj(x, y z)" function defined in FIPS 180-4 sections 4.1.1, 4.1.2, and 4.1.3.
        /// For each bit i in words x, y, and z if the majority of x[i], y[i], 
        /// and z[i] are set then result[i] is set, otherwise result[i] is not set.
        /// \tparam T The type of number being used.
        template <typename T> 
        constexpr T majority(const T x, const T y, const T z) noexcept {
            return (x & y) ^ (x & z) ^ (y & z);
        }
        
        /// The "Parity(x, y, z)" function defined in FIPS 180-4 section 4.1.1.
        /// For each bit i in words x, y, and z if there are an even number of 
        /// set bits in x[i], y[i], z[i] then result[i] is not set, otherwise 
        /// result[i] is set.
        /// \tparam word_t The type of number being used.
        constexpr std::uint32_t parity(const uint32_t x,
                                       const uint32_t y,
                                       const uint32_t z) noexcept {
            return x ^ y ^ z;
        }

        /// Creates an array where each element is generated using a function 
        /// that takes its position in the array as a template argument.
        /// \tparam num_elements The number of elements in the array.
        /// \tparam func_t       The type of predicate function used to generate the elements.
        /// \param func The function used to generate the elements.
        /// \returns An array of data_t containing num_elements where each element is generated by func.
        template <std::size_t num_elements, typename func_t>
        consteval auto generate_array(func_t func) 
            -> std::array<decltype(func.template operator()<0>()), num_elements> {
            return [&func]<std::size_t... indices>(std::index_sequence<indices...>) consteval {
                return std::array{func.template operator()<indices>()...};
            }(std::make_index_sequence<num_elements>{});
        }

        /// An array of function pointers representing the eighty SHA-1 functions as defined in FIPS 180-4 section 4.1.1.
        constexpr auto sha1_functions = generate_array<80>([]<std::size_t index>() consteval {
            constexpr std::array funcs{
                choose<std::uint32_t>, 
                parity, 
                majority<std::uint32_t>, 
                parity
            };
            return funcs.at(index / 20);
        });
        
        constexpr auto sha1_constants = generate_array<80>([]<std::size_t index>() consteval {
            constexpr std::array<std::uint32_t, 4> constants = {
                0x5A827999,
                0x6ED9EBA1,
                0x8F1BBCDC,
                0xCA62C1D6,
            };
            return constants.at(index / 20);
        });
        
        constexpr static std::array<uint32_t, 5> sha1_initialization_vector = {
    		0x67452301,
    		0xefcdab89,
    		0x98badcfe,
    		0x10325476,
    		0xc3d2e1f0,
        };

    }; // end namespace internal
   
    /// source https://github.com/vexingcodes/ctsha/blob/master/ctsha.hpp
    /// Computes the SHA-1 hash of a given message.
    /// \tparam num_bytes The length of the message.
    /// \param message The message for which the SHA-1 hash is being computed.
    /// \returns An array of bytes representing the SHA-1 hash result.
    template <std::size_t num_bytes>
    consteval std::array<std::byte, 20> sha1(const std::array<std::byte, num_bytes>& message) noexcept {
        auto state = internal::sha1_initialization_vector;
        for (const auto& block : preprocess_message<std::uint32_t>(message)) {
            // Prepare the message schedule.
            std::array<std::uint32_t, 80> w{};
            for (std::size_t t = 0; t < w.size(); ++t)
                w.at(t) = (t < 16) ? big_endian_to_host(block.at(t))
                                   : rotl<1>(w.at(t - 3) ^ w.at(t - 8) ^ w.at(t - 14) ^ w.at(t - 16));
    
            // Initialize the working variables. (a=0, b=1, c=2, d=3, e=4)
            auto v = state;
    
            // Compute new values for the working variables.
            for (std::size_t t = 0; t < w.size(); ++t) {
                std::uint32_t upper_t = rotl<5>(v.at(0)) + internal::sha1_functions.at(t)(v.at(1), v.at(2), v.at(3)) +
                                        v.at(4) + internal::sha1_constants.at(t) + w.at(t);
                v.at(4) = v.at(3);                  // e = d
                v.at(3) = v.at(2);                  // d = c
                v.at(2) = rotl<30>(v.at(1));        // c = ROTL30(b)
                v.at(1) = v.at(0);                  // b = a
                v.at(0) = upper_t;                  // a = T
            }
    
            // Compute the intermediate hash value.
            for (auto si = state.begin(), vi = v.begin(); si != state.end() && vi != v.end(); ++si, ++vi)
                *si = *vi + *si;
        }
    
        return to_bytes<std::endian::big>(state);
    }
}; // end namespace cryptanalysislib

/// Allows the SHA-1 hash of a message to be computed using a string literal.
///
/// @tparam char_t The type of characters in the string literal. Only char is supported here.
/// @tparam chars  The characters in the string literal.
///
/// @returns A byte array representing the SHA-1 hash of the given string literal.
template <typename char_t, char_t... chars>
static constexpr auto operator "" _sha1() {
    return cryptanalysislib::sha1(std::array<std::byte, sizeof...(chars)>{std::byte{chars}...});
}


#endif //CRYPTANALYSISLIB_SHA1_H
