/// ispc kernels for `algorithm/histogram.h`. Declared in
/// `cryptanalysislib::internal::ispc` (only if `USE_ISPC` is defined).

/// 256-bin histogram over `inlen` elements
/// NOTE: only the lower 8 bits of each element are counted
/// NOTE: `cnt` is overwritten, it does not need to be zeroed
/// NOTE: every program instance counts into its own row of a private
/// 	programCount x 256 table, hence the gather/scatter of one gang never
/// 	hits the same counter twice. The rows are summed into `cnt` at the end.
/// \param cnt[out]: Output histogram array (256 elements)
/// \param data[in]: Input data array (`in` is an ispc keyword)
/// \param inlen[in]: Number of elements in input array
template <typename T>
void histogram(uniform uint32 cnt[],
               const uniform T data[],
               const uniform uint64 inlen) {
	uniform uint32 table[256 * programCount];
	foreach (i = 0 ... 256 * programCount) {
		table[i] = 0;
	}

	// NOTE: `foreach` iterates over 32-bit indices, hence the input is
	// 	processed in chunks of 2**30 elements
	const uniform uint64 chunk = 1ull << 30;
	for (uniform uint64 o = 0; o < inlen; o += chunk) {
		const uniform int32 len = (uniform int32)min(chunk, inlen - o);
		const uniform T *uniform ptr = data + o;
		foreach (i = 0 ... len) {
			const uint32 v = (uint32)(ptr[i] & 0xFF);
			table[programIndex * 256 + v] += 1;
		}
	}

	foreach (v = 0 ... 256) {
		uint32 s = 0;
		for (uniform int32 j = 0; j < programCount; j++) {
			s += table[j * 256 + v];
		}
		cnt[v] = s;
	}
}

export void histogram_u8(uniform uint32 cnt[],
                         const uniform uint8 data[],
                         const uniform uint64 inlen) {
	histogram<uint8>(cnt, data, inlen);
}

export void histogram_u16(uniform uint32 cnt[],
                          const uniform uint16 data[],
                          const uniform uint64 inlen) {
	histogram<uint16>(cnt, data, inlen);
}

export void histogram_u32(uniform uint32 cnt[],
                          const uniform uint32 data[],
                          const uniform uint64 inlen) {
	histogram<uint32>(cnt, data, inlen);
}

export void histogram_u64(uniform uint32 cnt[],
                          const uniform uint64 data[],
                          const uniform uint64 inlen) {
	histogram<uint64>(cnt, data, inlen);
}
