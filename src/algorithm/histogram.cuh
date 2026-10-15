#ifndef CRYPTANALYSISLIB_ALGORITHM_HISTOGRAM_CUH
#define CRYPTANALYSISLIB_ALGORITHM_HISTOGRAM_CUH

/// CUDA histogram over all values of 8/16-bit samples, i.e. 2**(8*sizeof(T))
/// bins (same semantic as `cryptanalysislib::algorithm::histogram`).
/// The primitives follow the CUB layering and conventions:
/// 	- thread-level: `ThreadLoadVectorized`, `ThreadHistogram`
/// 	- warp-level: `WarpHistogram`
/// 	- block-level: `BlockHistogram` (interface of `cub::BlockHistogram`)
/// 	- device-level: `DeviceHistogram` (like the environment based,
/// 		temp-storage free overloads of the CUB device algorithms)
/// NOTE: needs nvcc. `algorithm/histogram.h` includes it, if compiled with
/// 	nvcc, to enable `parallel_cuda_policy`.
/// NOTE: `::cuda` and not `cuda` is used for libcu++, as
/// 	`cryptanalysislib::internal::cuda` hides it.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cuda/cmath>
#include <cuda/std/type_traits>
#include <cuda/stream_ref>

namespace cryptanalysislib::internal::cuda {

/// number of bins of a histogram over all values of `T`
template<typename T>
constexpr static uint32_t histogram_bins = 1u << (8u * sizeof(T));

/// size of the vectorized loads
constexpr static uint32_t histogram_vector_bytes = 16;

/// the shared memory a block may use for its private histogram, the
/// histograms of larger samples are directly computed in global memory
constexpr static size_t histogram_max_smem_bytes = 48u * 1024u;

/// thread-level: loads `ITEMS` consecutive samples with a single vector load
/// (`cub::LoadDirectBlockedVectorized`)
/// NOTE: `ptr` must be aligned to `ITEMS * sizeof(T)`
/// \param ptr[in]: first sample
/// \param items[out]: the samples
template<typename T, const int ITEMS>
__device__ __forceinline__ void ThreadLoadVectorized(const T *ptr,
                                                     T (&items)[ITEMS]) noexcept {
	struct alignas(ITEMS * sizeof(T)) Vector { T v[ITEMS]; };
	const Vector v = *reinterpret_cast<const Vector *>(ptr);
	#pragma unroll
	for (int i = 0; i < ITEMS; i++) {
		items[i] = v.v[i];
	}
}

/// thread-level: adds the samples of a thread to `histogram`
/// NOTE: a run of equal consecutive samples needs a single atomic, as in the
/// 	run-length accumulation of `cub::DeviceHistogram`
/// \tparam Scope: `::cuda::thread_scope_block` if `histogram` is in shared
/// 	memory, `::cuda::thread_scope_device` if it is in global memory
/// \param items[in]: the samples
/// \param histogram[in/out]: one counter per value of `T`
template<const ::cuda::thread_scope Scope, typename T, const int ITEMS, typename CounterT>
__device__ __forceinline__ void ThreadHistogram(const T (&items)[ITEMS],
                                                CounterT *histogram) noexcept {
	CounterT run = 1;
	#pragma unroll
	for (int i = 1; i < ITEMS; i++) {
		if (items[i] == items[i - 1]) {
			run += 1;
		} else {
			::cuda::atomic_ref<CounterT, Scope>(histogram[items[i - 1]]).fetch_add(run, ::cuda::std::memory_order_relaxed);
			run = 1;
		}
	}
	::cuda::atomic_ref<CounterT, Scope>(histogram[items[ITEMS - 1]]).fetch_add(run, ::cuda::std::memory_order_relaxed);
}

/// warp-level: adds the samples of a warp to `histogram`. The lanes holding
/// the same sample are matched (`__match_any_sync`), only the first of them
/// updates the counter. Reduces the contention for skewed samples.
/// NOTE: all 32 lanes of the warp must call it
/// \tparam Scope: see `ThreadHistogram`
template<const ::cuda::thread_scope Scope, typename CounterT>
struct WarpHistogram {
	/// \param items[in]: the samples of the calling lane
	/// \param histogram[in/out]: one counter per value of `T`
	template<typename T, const int ITEMS>
	__device__ __forceinline__ void Composite(const T (&items)[ITEMS],
	                                          CounterT *histogram) const noexcept {
		const uint32_t lane = threadIdx.x % 32u;
		#pragma unroll
		for (int i = 0; i < ITEMS; i++) {
			const uint32_t peers = __match_any_sync(0xFFFFFFFFu, uint32_t(items[i]));
			if (lane == uint32_t(__ffs(peers) - 1)) {
				::cuda::atomic_ref<CounterT, Scope>(histogram[items[i]]).fetch_add(CounterT(__popc(peers)), ::cuda::std::memory_order_relaxed);
			}
		}
	}
};

/// algorithms of `BlockHistogram`
/// NOTE: `cub::BlockHistogramAlgorithm` offers `BLOCK_HISTO_SORT` instead of
/// 	the warp aggregation
enum BlockHistogramAlgorithm {
	/// shared memory atomics, see `ThreadHistogram`
	BLOCK_HISTO_ATOMIC,
	/// warp aggregated shared memory atomics, see `WarpHistogram`
	BLOCK_HISTO_WARP_AGGREGATE,
};

/// block-level: histogram in shared memory, same interface as
/// `cub::BlockHistogram`
/// \tparam T: sample type
/// \tparam BLOCK_THREADS: number of threads per block
/// \tparam ITEMS_PER_THREAD: number of samples per thread
/// \tparam BINS: number of bins
/// \tparam ALGORITHM: see `BlockHistogramAlgorithm`
template<typename T,
         const int BLOCK_THREADS,
         const int ITEMS_PER_THREAD,
         const int BINS,
         const BlockHistogramAlgorithm ALGORITHM = BLOCK_HISTO_ATOMIC>
class BlockHistogram {
	static_assert(BLOCK_THREADS % 32 == 0, "BlockHistogram: BLOCK_THREADS must be a multiple of the warp size");

public:
	/// no temporary storage is needed (as `cub::BlockHistogramAtomic`)
	struct TempStorage {};

	__device__ __forceinline__ BlockHistogram() noexcept = default;
	__device__ __forceinline__ explicit BlockHistogram(TempStorage &) noexcept {}

	/// zeros the shared memory histogram
	/// NOTE: a `__syncthreads()` is needed before the next `Composite`
	template<typename CounterT>
	__device__ __forceinline__ void InitHistogram(CounterT histogram[BINS]) const noexcept {
		for (int i = threadIdx.x; i < BINS; i += BLOCK_THREADS) {
			histogram[i] = 0;
		}
	}

	/// adds the samples of the block to the shared memory histogram
	template<typename CounterT>
	__device__ __forceinline__ void Composite(const T (&items)[ITEMS_PER_THREAD],
	                                          CounterT histogram[BINS]) const noexcept {
		if constexpr (ALGORITHM == BLOCK_HISTO_WARP_AGGREGATE) {
			WarpHistogram<::cuda::thread_scope_block, CounterT>{}.Composite(items, histogram);
		} else {
			ThreadHistogram<::cuda::thread_scope_block>(items, histogram);
		}
	}

	/// zeros the shared memory histogram and adds the samples of the block
	template<typename CounterT>
	__device__ __forceinline__ void Histogram(const T (&items)[ITEMS_PER_THREAD],
	                                          CounterT histogram[BINS]) const noexcept {
		InitHistogram(histogram);
		__syncthreads();
		Composite(items, histogram);
	}
};

/// kernel of `DeviceHistogram`: every block processes tiles of
/// `BLOCK_THREADS * ITEMS` samples, read with vector loads.
/// \tparam SMEM: if true every block computes a private histogram in shared
/// 	memory, which is added to `d_histogram` at the end. Otherwise all
/// 	blocks directly update `d_histogram`, always aggregated per warp
/// 	(`ALGORITHM` is ignored): the plain global atomics serialize on skewed
/// 	samples (u16 with 4 values, RTX 5090: 16 GB/s instead of 130 GB/s)
/// \param d_samples[in]: input samples
/// \param d_histogram[in/out]: zeroed histogram
/// \param num_samples[in]: number of samples
template<typename T, typename CounterT,
         const int BLOCK_THREADS,
         const BlockHistogramAlgorithm ALGORITHM,
         const bool SMEM>
__global__ void __launch_bounds__(BLOCK_THREADS)
DeviceHistogramKernel(const T *d_samples,
                      CounterT *d_histogram,
                      const size_t num_samples) {
	constexpr int ITEMS = histogram_vector_bytes / sizeof(T);
	constexpr int BINS = histogram_bins<T>;
	constexpr size_t TILE = size_t(BLOCK_THREADS) * ITEMS;
	using BlockHistogramT = BlockHistogram<T, BLOCK_THREADS, ITEMS, BINS, ALGORITHM>;

	// the counters of a block do not overflow, as a block processes less
	// than 2**32 samples (`DeviceHistogram` limits the number of samples)
	__shared__ uint32_t smem_histogram[SMEM ? BINS : 1];
	__shared__ typename BlockHistogramT::TempStorage temp_storage;
	const BlockHistogramT block_histogram(temp_storage);
	if constexpr (SMEM) {
		block_histogram.InitHistogram(smem_histogram);
		__syncthreads();
	}

	// the samples before the first aligned vector and after the last tile
	const size_t misaligned = (reinterpret_cast<uintptr_t>(d_samples) % histogram_vector_bytes) / sizeof(T);
	const size_t head = misaligned ? (num_samples < ITEMS - misaligned ? num_samples : ITEMS - misaligned) : 0;
	const T *aligned = d_samples + head;
	const size_t tiles = (num_samples - head) / TILE;
	const size_t tail = head + tiles * TILE;
	const size_t rest = head + num_samples - tail;
	for (size_t i = size_t(blockIdx.x) * BLOCK_THREADS + threadIdx.x; i < rest;
	     i += size_t(gridDim.x) * BLOCK_THREADS) {
		const T items[1] = {d_samples[i < head ? i : tail + i - head]};
		if constexpr (SMEM) {
			ThreadHistogram<::cuda::thread_scope_block>(items, smem_histogram);
		} else {
			ThreadHistogram<::cuda::thread_scope_device>(items, d_histogram);
		}
	}

	for (size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
		T items[ITEMS];
		ThreadLoadVectorized(aligned + tile * TILE + size_t(threadIdx.x) * ITEMS, items);
		if constexpr (SMEM) {
			block_histogram.Composite(items, smem_histogram);
		} else {
			WarpHistogram<::cuda::thread_scope_device, CounterT>{}.Composite(items, d_histogram);
		}
	}

	if constexpr (SMEM) {
		__syncthreads();
		for (int i = threadIdx.x; i < BINS; i += BLOCK_THREADS) {
			const uint32_t c = smem_histogram[i];
			if (c) {
				::cuda::atomic_ref<CounterT, ::cuda::thread_scope_device>(d_histogram[i]).fetch_add(CounterT(c), ::cuda::std::memory_order_relaxed);
			}
		}
	}
}

/// device-level histogram
struct DeviceHistogram {
	/// computes the histogram over all values of `T`, i.e. `d_histogram`
	/// needs 2**(8*sizeof(T)) counters.
	/// NOTE: asynchronous with respect to the host
	/// NOTE: `d_histogram` is overwritten, it does not need to be zeroed
	/// NOTE: if an element occurs more than 2**(8*sizeof(CounterT)) times,
	/// 	its counter overflows
	/// NOTE: RTX 5090, 2**28 u8 samples: 1.65 TB/s (memory bound) for uniform,
	/// 	4-valued and constant samples. 2**27 u16 samples: 130-145 GB/s, as
	/// 	the 2**16 counters do not fit into shared memory.
	/// \tparam BLOCK_THREADS: number of threads per block
	/// \tparam ALGORITHM: see `BlockHistogramAlgorithm`, only used for 8-bit
	/// 	samples. `BLOCK_HISTO_WARP_AGGREGATE` is slow for uniform samples
	/// 	(RTX 5090: 0.23 instead of 1.65 TB/s)
	/// \param d_samples[in]: input samples (device accessible)
	/// \param d_histogram[out]: histogram (device accessible)
	/// \param num_samples[in]: number of samples
	/// \param stream[in]: stream to launch on
	/// \return the error of the launch
	template<const int BLOCK_THREADS = 512,
	         const BlockHistogramAlgorithm ALGORITHM = BLOCK_HISTO_ATOMIC,
	         typename T, typename CounterT>
	static cudaError_t Histogram(const T *d_samples,
	                             CounterT *d_histogram,
	                             const size_t num_samples,
	                             const ::cuda::stream_ref stream = cudaStream_t{}) noexcept {
		static_assert(::cuda::std::is_unsigned_v<T> && sizeof(T) <= 2, "DeviceHistogram: 8/16-bit unsigned samples only");
		static_assert(::cuda::std::is_same_v<CounterT, uint32_t> ||
		              ::cuda::std::is_same_v<CounterT, unsigned long> ||
		              ::cuda::std::is_same_v<CounterT, unsigned long long>,
		              "DeviceHistogram: 32/64-bit unsigned counters only");
		constexpr bool SMEM = histogram_bins<T> * sizeof(uint32_t) <= histogram_max_smem_bytes;
		const auto kernel = DeviceHistogramKernel<T, CounterT, BLOCK_THREADS, ALGORITHM, SMEM>;
		constexpr size_t TILE = size_t(BLOCK_THREADS) * histogram_vector_bytes / sizeof(T);

		cudaError_t err = cudaMemsetAsync(d_histogram, 0, histogram_bins<T> * sizeof(CounterT), stream.get());
		if (err != cudaSuccess || num_samples == 0) {
			return err;
		}

		// as many blocks as fit onto the device at once, but at most one per
		// tile and enough, such that a block processes less than 2**32 samples
		int device, sms, blocks_per_sm;
		if ((err = cudaGetDevice(&device)) != cudaSuccess ||
		    (err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device)) != cudaSuccess ||
		    (err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, BLOCK_THREADS, 0)) != cudaSuccess) {
			return err;
		}
		const size_t min_blocks = ::cuda::ceil_div(num_samples, size_t(1u) << 31u);
		const size_t max_blocks = size_t(sms) * blocks_per_sm, tiles = ::cuda::ceil_div(num_samples, TILE);
		size_t blocks = tiles < max_blocks ? tiles : max_blocks;
		blocks = blocks < min_blocks ? min_blocks : blocks;

		kernel<<<uint32_t(blocks), BLOCK_THREADS, 0, stream.get()>>>(d_samples, d_histogram, num_samples);
		return cudaGetLastError();
	}
};

/// host wrapper for `parallel_cuda_policy` (see `algorithm/histogram.h`),
/// synchronous with respect to the host. Host memory is staged through
/// device memory, device and managed memory is used directly.
/// NOTE: for host memory the copy dominates (RTX 5090, PCIe, 2**24 u8
/// 	samples: 6 GB/s)
/// NOTE: `cnt` is overwritten, it does not need to be zeroed
/// \param cnt[out]: 2**(8*sizeof(T)) counters
/// \param in[in]: input samples
/// \param inlen[in]: number of samples
/// \param stream[in]: stream to launch on
/// \param device[in]: device to run on
/// \return false on any CUDA error (e.g. no device), then `cnt` is unspecified
template<typename T, typename C>
bool histogram(C *cnt,
               const T *in,
               const size_t inlen,
               cudaStream_t stream,
               const int device) noexcept {
	int prev_device;
	if (cudaGetDevice(&prev_device) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) {
		return false;
	}

	/// \return true if `ptr` must be copied from/to the device
	const auto is_host = [](const void *ptr) {
		cudaPointerAttributes attr;
		if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
			return true;
		}
		return attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged;
	};

	const size_t in_bytes = inlen * sizeof(T), cnt_bytes = histogram_bins<T> * sizeof(C);
	const bool stage_in = inlen > 0 && is_host(in), stage_cnt = is_host(cnt);
	const T *d_in = in;
	C *d_cnt = cnt;
	void *buffer = nullptr;
	cudaError_t err = cudaSuccess;
	if (stage_in) {
		err = cudaMallocAsync(&buffer, in_bytes, stream);
		if (err == cudaSuccess) {
			d_in = static_cast<const T *>(buffer);
			err = cudaMemcpyAsync(buffer, in, in_bytes, cudaMemcpyHostToDevice, stream);
		}
	}
	if (err == cudaSuccess && stage_cnt) {
		err = cudaMallocAsync(&d_cnt, cnt_bytes, stream);
	}
	if (err == cudaSuccess) {
		err = DeviceHistogram::Histogram(d_in, d_cnt, inlen, stream);
	}
	if (err == cudaSuccess && stage_cnt) {
		err = cudaMemcpyAsync(cnt, d_cnt, cnt_bytes, cudaMemcpyDeviceToHost, stream);
	}

	if (stage_cnt && d_cnt != cnt) {
		cudaFreeAsync(d_cnt, stream);
	}
	if (buffer != nullptr) {
		cudaFreeAsync(buffer, stream);
	}
	const cudaError_t sync = cudaStreamSynchronize(stream);
	cudaSetDevice(prev_device);
	return err == cudaSuccess && sync == cudaSuccess;
}

} // end namespace cryptanalysislib::internal::cuda
#endif
