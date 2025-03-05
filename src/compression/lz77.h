#ifndef CRYPTANALYSISLIB_COMPRESSION_LZ77_H
#define CRYPTANALYSISLIB_COMPRESSION_LZ77_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef CRYPTANALYSISLIB_COMPRESSION_H
#error "dont include this file directly. Use `#include <compression/compression.h>`"
#endif

/// org code:https://github.com/andyherbert/lz1

///
/// \param compressed_text
/// \param uncompressed_text
/// \param uncompressed_size
/// \param pointer_length_width
/// \return compresses size
uint32_t lz77_compress(uint8_t *compressed_text,
                       const uint8_t *uncompressed_text,
                       const size_t uncompressed_size,
                       const uint8_t pointer_length_width) {
	uint16_t pointer_pos, temp_pointer_pos, output_pointer, pointer_length, temp_pointer_length;
	uint32_t compressed_pointer, output_size, coding_pos, output_lookahead_ref, look_behind, look_ahead;
	uint16_t pointer_pos_max, pointer_length_max;
	pointer_pos_max = 1u << (16 - pointer_length_width);
	pointer_length_max = 1u<< pointer_length_width;

	*((uint32_t *) compressed_text) = uncompressed_size;
	*(compressed_text + 4) = pointer_length_width;
	compressed_pointer = output_size = 5;

	for (coding_pos = 0; coding_pos < uncompressed_size; ++coding_pos) {
		pointer_pos = 0;
		pointer_length = 0;
		for (temp_pointer_pos = 1; (temp_pointer_pos < pointer_pos_max) &&
		                           (temp_pointer_pos <= coding_pos);
		     ++temp_pointer_pos) {

			look_behind = coding_pos - temp_pointer_pos;
			look_ahead = coding_pos;
			for (temp_pointer_length = 0; uncompressed_text[look_ahead++] == uncompressed_text[look_behind++]; ++temp_pointer_length)
				if (temp_pointer_length == pointer_length_max)
					break;
			if (temp_pointer_length > pointer_length) {
				pointer_pos = temp_pointer_pos;
				pointer_length = temp_pointer_length;
				if (pointer_length == pointer_length_max)
					break;
			}
		}

		coding_pos += pointer_length;
		if ((coding_pos == uncompressed_size) && pointer_length) {
			output_pointer = (pointer_length == 1) ? 0 : ((pointer_pos << pointer_length_width) | (pointer_length - 2));
			output_lookahead_ref = coding_pos - 1;
		} else {
			output_pointer = (pointer_pos << pointer_length_width) | (pointer_length ? (pointer_length - 1) : 0);
			output_lookahead_ref = coding_pos;
		}
		*((uint16_t *) (compressed_text + compressed_pointer)) = output_pointer;
		compressed_pointer += 2;
		*(compressed_text + compressed_pointer++) = *(uncompressed_text + output_lookahead_ref);
		output_size += 3;
	}

	return output_size;
}

uint32_t lz77_decompress(uint8_t *uncompressed_text,
                         const uint8_t *compressed_text) {
	uint8_t pointer_length_width;
	uint16_t input_pointer, pointer_length, pointer_pos, pointer_length_mask;
	uint32_t compressed_pointer, coding_pos, pointer_offset, uncompressed_size;

	uncompressed_size = *((uint32_t *) compressed_text);
	pointer_length_width = *(compressed_text + 4);
	compressed_pointer = 5;

	pointer_length_mask = (1u << pointer_length_width) - 1;

	for (coding_pos = 0; coding_pos < uncompressed_size; ++coding_pos) {
		input_pointer = *((uint16_t *) (compressed_text + compressed_pointer));
		compressed_pointer += 2;
		pointer_pos = input_pointer >> pointer_length_width;
		pointer_length = pointer_pos ? ((input_pointer & pointer_length_mask) + 1) : 0;
		if (pointer_pos)
			for (pointer_offset = coding_pos - pointer_pos; pointer_length > 0; --pointer_length)
				uncompressed_text[coding_pos++] = uncompressed_text[pointer_offset++];
		*(uncompressed_text + coding_pos) = *(compressed_text + compressed_pointer++);
	}

	return coding_pos;
}

#endif
