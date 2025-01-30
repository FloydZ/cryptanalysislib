/******************************************************************************
 * mini_ips4o.hpp
 *
 * In-place Parallel Super Scalar Samplesort (IPS⁴o)
 *
 ******************************************************************************
 * BSD 2-Clause License
 *
 * Copyright © 2017, Michael Axtmann <michael.axtmann@gmail.com>
 * Copyright © 2017, Daniel Ferizovic <daniel.ferizovic@student.kit.edu>
 * Copyright © 2017, Sascha Witt <sascha.witt@kit.edu>
 * Copyright © 2021, Google LLC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    using T = int;
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#pragma once

// Requires AVX-512 and github.com/google/highway if 1.
#include <cstdlib>
#define IPS4O_SIMD 0

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <cassert>
#include <memory>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include "memory/memory.h"

#if IPS4O_SIMD
#include "hwy/highway.h"
#endif

#define IPS4O_RESTRICT __restrict__
#define IPS4O_UNLIKELY(expr) __builtin_expect(!!(expr), 0)

#define IPS4OML_ASSUME_NOT(c) \
	if (c) __builtin_unreachable()
#define IPS4OML_IS_NOT(c) assert(!(c))


// TODO allocator + thread template
namespace ips4o {
	namespace detail {

#if IPS4O_SIMD
		using namespace hwy::HWY_NAMESPACE;
#endif

		/**
         * Compute the logarithm to base 2, rounded down.
         */
		inline constexpr unsigned long log2(unsigned long n) {
			return (std::numeric_limits<unsigned long>::digits - 1 - __builtin_clzl(n));
		}

		// TODO(?): make into template to allow other key types.
		struct Cfg {
			using value_type = uint32_t;
			using iterator = value_type *__restrict__;// restrict is important

			// Must be at least 1 vector.
			static constexpr const ptrdiff_t kBaseCaseSize = 32;

			// Must be at least 2 vectors.
			static constexpr const ptrdiff_t kBlockSizeInBytes = 1024;

			static constexpr const std::size_t kDataAlignment = 128;
            
            // Number of splitters that must be equal before equality buckets are enabled.
			static constexpr const ptrdiff_t kEqualBucketsThreshold = 5;

            ///Logarithm of the maximum number of buckets (excluding equality buckets).
			// TODO(?): currently AVX-512 specific to avoid Gather
			static constexpr const int kLogBuckets = 6;

			// TODO(?): adapt to base case sorter (exact match size of sorting network)
			static constexpr const ptrdiff_t kSingleLevelThreshold =
			        kBaseCaseSize * (1ul << kLogBuckets);
			static constexpr const ptrdiff_t kTwoLevelThreshold =
			        kSingleLevelThreshold * (1ul << kLogBuckets);

            /// Computes the logarithm of the number of buckets to use for input size n.
			constexpr static int logBuckets(const ptrdiff_t n) noexcept {
				if (n <= kSingleLevelThreshold) {
					// Only one more level until the base case, reduce the number of buckets
					return std::max(1ul, log2(n / kBaseCaseSize));
				} else if (n <= kTwoLevelThreshold) {
					// Only two more levels until we reach the base case, split the buckets
					// evenly
					return std::max(1ul, (log2(n / kBaseCaseSize) + 1) / 2);
				} else {
					// Use the maximum number of buckets
					return kLogBuckets;
				}
			}

            /// Maximum number of buckets (including equality buckets).
			static constexpr const int kMaxBuckets = 1ul << (kLogBuckets + 1);

            /// Number of elements in one block.
			static constexpr ptrdiff_t kBlockSize =
			        kBlockSizeInBytes / sizeof(value_type);

			// The implementation of the block alignment limits the possible block sizes.
			static_assert((kBlockSize & (kBlockSize - 1)) == 0,
			              "Block size must be a power of two.");

            /// Aligns an offset to the next block boundary, upwards.
			[[nodiscard]] static constexpr ptrdiff_t alignToNextBlock(const ptrdiff_t p) noexcept {
				return (p + kBlockSize - 1) & ~(kBlockSize - 1);
			}
		};

		// Read/write indices
		class BucketPointers {
		public:
            /// \param l_w[in]:
            /// \param m_r[in]:
			constexpr inline void set(const ptrdiff_t l_w,
                                      const ptrdiff_t m_r) noexcept {
				m_ = m_r;
				l_ = l_w;
			}
            
            /// \return 
			[[nodiscard]] constexpr inline ptrdiff_t getWrite() const noexcept { 
                return l_; 
            }

            /// Gets write/read pointers and increases the write pointer.
            /// \return 
			constexpr std::pair<ptrdiff_t, ptrdiff_t> incWrite() noexcept {
				const auto tmp = l_;
				l_ += Cfg::kBlockSize;
				return {tmp, m_};
			}

            /// Gets write/read pointers, decreases the read pointer.
            /// \return 
			std::pair<ptrdiff_t, ptrdiff_t> decRead() noexcept {
				const auto tmp = m_;
				m_ -= Cfg::kBlockSize;
				return {l_, tmp};
			}

		private:
			ptrdiff_t l_;
			ptrdiff_t m_;
		};

        /// A single buffer block: array of values.
		class Block {
		public:
			constexpr inline Cfg::value_type *data() noexcept { 
                return storage_; 
            }

			constexpr inline const Cfg::value_type *data() const noexcept { 
                return storage_; 
            }

			__attribute__((noinline)) 
            void readFrom(const Cfg::iterator src) noexcept {
				// memcpy(storage_, src, Cfg::kBlockSizeInBytes);
				memcpy<Cfg::value_type>(storage_, src, Cfg::kBlockSize);
			}

			__attribute__((noinline))
            void writeToBlock(Block &block) noexcept {
				//memcpy(block.storage_, storage_, Cfg::kBlockSizeInBytes);
				memcpy<Cfg::value_type>(block.storage_, storage_, Cfg::kBlockSize);
			}

			__attribute__((noinline)) 
            void writeTo(Cfg::iterator dest) noexcept {
				// memcpy(dest, storage_, Cfg::kBlockSizeInBytes);
				memcpy<Cfg::value_type>(dest, storage_, Cfg::kBlockSize);
			}

		private:
			Cfg::value_type storage_[Cfg::kBlockSize];
		};

        /// Per-thread buffers for each bucket (array of pointers into one allocation).
        // TODO template<class Allocator = std::allocator<int>>
		class Buffers {
		public:
			Buffers() noexcept
			    : storage_(static_cast<Cfg::value_type *>(
			              std::aligned_alloc(Cfg::kDataAlignment, Cfg::kBlockSizeInBytes * Cfg::kMaxBuckets * 2))) {
				for (size_t idx_bucket = 0; idx_bucket < Cfg::kMaxBuckets; ++idx_bucket) {
					reset(idx_bucket);
					end_[idx_bucket] = indices_[idx_bucket] + Cfg::kBlockSize;
				}
			}

			~Buffers() noexcept { 
                free(storage_);
            }

            /// Resets to initial position.
			constexpr inline void reset(const size_t idx_bucket) noexcept {
				indices_[idx_bucket] = Cfg::kBlockSize * idx_bucket;
			}

            /// Pointer to start of buffer.
			constexpr inline Cfg::value_type *data(const size_t idx_bucket) const noexcept {
				return storage_ + (Cfg::kBlockSize * idx_bucket);
			}

			constexpr inline ptrdiff_t size(const size_t idx_bucket) const noexcept {
				return indices_[idx_bucket] - (Cfg::kBlockSize * idx_bucket);
			}

			// Only these two are time-critical and optimized.
			// TODO(?): to allow SIMD writes, we should add >=1 vector of padding to the
			// end of each block (also allows larger block sizes without suffering 2K
			// aliasing), and check whether the block is full *after* writing, copying
			// any spillover from the padding to the start of the newly flushed block.
			constexpr inline bool isFull(const size_t idx_bucket) const noexcept {
				return indices_[idx_bucket] == end_[idx_bucket];
			}

			constexpr inline void push(const size_t idx_bucket, 
                                       const Cfg::value_type value) noexcept {
				storage_[indices_[idx_bucket]++] = value;
			}

            /// Flushes buffer to input.
			void writeTo(const size_t idx_bucket, Cfg::iterator dest) {
                // rewind to start of buffer
				indices_[idx_bucket] -= Cfg::kBlockSize;

				Cfg::value_type *bucket = storage_ + indices_[idx_bucket];
				//memcpy(dest, bucket, Cfg::kBlockSizeInBytes);
				memcpy(dest, bucket, Cfg::kBlockSize);
			}

		private:
			Cfg::value_type *storage_;
			size_t indices_[Cfg::kMaxBuckets];// into storage_
			size_t end_[Cfg::kMaxBuckets];    // into storage_
			// Blocks should have no extra elements or padding
			static_assert(sizeof(Block) == sizeof(Cfg::value_type) * Cfg::kBlockSize, "");
		};


        /// Branch-free classifier.
		class Classifier {
		public:
			using value_type = typename Cfg::value_type;
			using IdxBucket = ptrdiff_t;

            /// The sorted array of splitters, to be filled externally.
			constexpr inline value_type *getSortedSplitters() noexcept {
				return static_cast<value_type *>(static_cast<void *>(sorted_storage_));
			}

            /// Builds the tree from the sorted splitters.
			constexpr void build(const uint32_t log_buckets) noexcept {
				log_buckets_ = log_buckets;
				num_buckets_ = 1 << log_buckets;
				const auto num_splitters = (1 << log_buckets) - 1;
				IPS4OML_ASSUME_NOT(getSortedSplitters() + num_splitters == nullptr);
                // TODO replace with allocator
				new (getSortedSplitters() + num_splitters)
				        value_type(getSortedSplitters()[num_splitters - 1]);
				build(getSortedSplitters(), getSortedSplitters() + num_splitters, 1);
			}

            /// Classifies a single element.
			template<bool kEqualBuckets>
			IdxBucket classify(const value_type &value) const {
				const uint32_t log_buckets = log_buckets_;
				const IdxBucket num_buckets = num_buckets_;
				IPS4OML_ASSUME_NOT(log_buckets < 1);
				IPS4OML_ASSUME_NOT(log_buckets > Cfg::kLogBuckets + 1);

				IdxBucket b = 1;
				std::less<Cfg::value_type> comp_;
				for (uint32_t l = 0; l < log_buckets; ++l) {
					b = 2 * b + comp_(*splitter(b), value);
                }

				if (kEqualBuckets) {
					b = 2 * b + !comp_(value, *sortedSplitter(b - num_buckets));
                }

				return b - (kEqualBuckets ? 2 * num_buckets : num_buckets);
			}

			constexpr inline const value_type *splitter(const IdxBucket i) const noexcept {
				return &storage_[i];
			}

			constexpr  inline const value_type *sortedSplitter(const IdxBucket i) const noexcept {
				return &sorted_storage_[i];
			}

            /// Recursively builds the tree.
			void build(const value_type *const left,
                       const value_type *const right,
			           const IdxBucket pos) noexcept {
				const auto mid = left + (right - left) / 2;
				IPS4OML_ASSUME_NOT(storage_ + pos == nullptr);
				new (storage_ + pos) value_type(*mid);
				if (2 * pos < num_buckets_) {
					build(left, mid, 2 * pos);
					build(mid, right, 2 * pos + 1);
				}
			}

			size_t log_buckets_ = 0;
			ptrdiff_t num_buckets_ = 0;

			// Filled from 1 to num_buckets_
			value_type storage_[Cfg::kMaxBuckets / 2];
			// Filled from 0 to num_buckets_, last one is duplicated
			value_type sorted_storage_[Cfg::kMaxBuckets / 2];
		};

        /// TODO replace with my prng class 
		class Sampler {
			static constexpr const bool kIs64Bit = sizeof(std::uintptr_t) == 8;

		public:
			Sampler() noexcept {
				std::random_device rdev;
				ptrdiff_t seed = rdev();
				if (kIs64Bit) { seed = (seed << (kIs64Bit * 32)) | rdev(); }
				random_generator.seed(seed);
			}

			void selectInPlace(Cfg::iterator begin, const Cfg::iterator end,
			                   ptrdiff_t num_samples) {
				ptrdiff_t n = end - begin;
				while (num_samples--) {
					const auto i =
					        std::uniform_int_distribution<ptrdiff_t>(0, --n)(random_generator);
					std::swap(*begin, begin[i]);
					++begin;
				}
			}

		private:
			// Random bit generator for sampling
			// LCG using constants by Knuth (for 64 bit) or Numerical Recipes (for 32 bit)
			std::linear_congruential_engine<
			        std::uintptr_t, kIs64Bit ? 6364136223846793005u : 1664525u,
			        kIs64Bit ? 1442695040888963407u : 1013904223u, 0u>
			        random_generator;
		};

		template<class It>
		__attribute__((noinline)) void baseCaseSort(It begin, It end) {
			// Insertion sort
			std::less<Cfg::value_type> comp;
			for (It it = begin + 1; it < end; ++it) {
				Cfg::value_type val = *it;
				if (comp(val, *begin)) {
					std::move_backward(begin, it, it + 1);
					*begin = std::move(val);
				} else {
					auto cur = it;
					for (auto next = it - 1; comp(val, *next); --next) {
						*cur = std::move(*next);
						cur = next;
					}
					*cur = std::move(val);
				}
			}
		}

		class Sorter {
		public:
			using iterator = Cfg::iterator;
			using value_type = Cfg::value_type;
			using IdxBucket = ptrdiff_t;

			/**
   * Recursive entry point for sequential algorithm.
   */
			__attribute__((noinline)) void sequential(const iterator begin,
			                                          const iterator end) {
				// Check for base case
				const ptrdiff_t n = end - begin;
				if (n <= 2 * Cfg::kBaseCaseSize) {
					baseCaseSort(begin, end);
					return;
				}

				sequential_rec(begin, end);
			}

		private:
			__attribute__((noinline)) std::pair<int, bool> partition(
			        const iterator begin, const iterator end, ptrdiff_t *const bucket_start) {
				// Sampling
				bool use_equal_buckets = false;
				{
					std::tie(this->num_buckets_, use_equal_buckets) =
					        buildClassifier(begin, end);
				}

				// Set parameters for this partitioning step
				// Must do this AFTER sampling, because sampling will recurse to sort
				// splitters.
				this->bucket_start_ = bucket_start;
				this->overflow_ = nullptr;
				this->begin_ = begin;
				this->end_ = end;

				sequentialClassification(use_equal_buckets);

				// Compute which bucket can cause overflow
				const int overflow_bucket = computeOverflowBucket();

				// Block Permutation
				if (use_equal_buckets)
					permuteBlocks<true>();
				else
					permuteBlocks<false>();

				// Write remaining elements; save excess elements at right end of stripe.
				writeMargins(0, num_buckets_, overflow_bucket, -1, 0);

				std::fill_n(bucket_size, Cfg::kMaxBuckets, 0);

				return {this->num_buckets_, use_equal_buckets};
			}

			// Called by sequential().
			void sequential_rec(const iterator begin, const iterator end) {
				// Check for base case
				const auto n = end - begin;
				IPS4OML_IS_NOT(n <= 2 * Cfg::kBaseCaseSize);

				ptrdiff_t bucket_start[Cfg::kMaxBuckets + 1];

				// Do the partitioning
				const auto res = partition(begin, end, bucket_start);
				const int num_buckets = std::get<0>(res);
				const bool equal_buckets = std::get<1>(res);

				// Final base case is executed in cleanup step, so we're done here
				if (n <= Cfg::kSingleLevelThreshold) {
					return;
				}

				// Recurse
				for (int i = 0; i < num_buckets; i += 1 + equal_buckets) {
					const auto start = bucket_start[i];
					const auto stop = bucket_start[i + 1];
					if (stop - start > 2 * Cfg::kBaseCaseSize)
						sequential(begin + start, begin + stop);
				}
				if (equal_buckets) {
					const auto start = bucket_start[num_buckets - 1];
					const auto stop = bucket_start[num_buckets];
					if (stop - start > 2 * Cfg::kBaseCaseSize)
						sequential(begin + start, begin + stop);
				}
			}

		private:
			ptrdiff_t bucket_size[Cfg::kMaxBuckets] = {0};
			Buffers buffers;

			Classifier classifier;

			// Bucket information
			BucketPointers bucket_pointers[Cfg::kMaxBuckets];

			Sampler sampler_;

			// TODO(?): for verification, we should narrow the scope/visibility of these.
			// They are set and used in the recursion.
			ptrdiff_t *bucket_start_;
			Block *overflow_;

			iterator begin_;
			iterator end_;
			int num_buckets_;

			Block swap[2];
			Block overflow;

			/**
   * The oversampling factor to be used for input of size n.
   */
			static constexpr double oversamplingFactor(ptrdiff_t n) {
				return (0.2 * log2(n)) < 1.0 ? 1.0 : (0.2 * log2(n));
			}

			/**
   * Builds the classifer.
   * Number of used_buckets is a power of two and at least two.
   */
			std::pair<int, bool> buildClassifier(const iterator begin,
			                                     const iterator end) {
				const auto n = end - begin;
				int log_buckets = Cfg::logBuckets(n);
				int num_buckets = 1 << log_buckets;
				const auto step = std::max<ptrdiff_t>(1, oversamplingFactor(n));
				const auto num_samples = std::min(step * num_buckets - 1, n / 2);

				sampler_.selectInPlace(begin, end, num_samples);

				// Sort the sample
				sequential(begin, begin + num_samples);
				iterator splitter = begin + step - 1;
				auto sorted_splitters = classifier.getSortedSplitters();
				std::less<Cfg::value_type> comp;

				// Choose the splitters
				IPS4OML_ASSUME_NOT(sorted_splitters == nullptr);
				new (sorted_splitters) typename Cfg::value_type(*splitter);
				for (int i = 2; i < num_buckets; ++i) {
					splitter += step;
					// Skip duplicates
					if (comp(*sorted_splitters, *splitter)) {
						IPS4OML_ASSUME_NOT(sorted_splitters + 1 == nullptr);
						new (++sorted_splitters) Cfg::value_type(*splitter);
					}
				}

				// Check for duplicate splitters
				const auto diff_splitters =
				        sorted_splitters - classifier.getSortedSplitters() + 1;
				const bool use_equal_buckets =
				        num_buckets - 1 - diff_splitters >= Cfg::kEqualBucketsThreshold;

				// Fill the array to the next power of two
				log_buckets = log2(diff_splitters) + 1;
				num_buckets = 1 << log_buckets;
				for (int i = diff_splitters + 1; i < num_buckets; ++i) {
					IPS4OML_ASSUME_NOT(sorted_splitters + 1 == nullptr);
					new (++sorted_splitters) Cfg::value_type(*splitter);
				}

				// Build the tree
				classifier.build(log_buckets);

				const int used_buckets = num_buckets * (1 + use_equal_buckets);
				return {used_buckets, use_equal_buckets};
			}

			// TODO(?): this is a bottleneck, should be replaced with SIMD writes.
			__attribute__((noinline)) void ScatterBatch(
			        const uint32_t *IPS4O_RESTRICT bucket_indices, size_t num, iterator begin,
			        iterator &write) {
#pragma unroll(2)
				for (size_t i = 0; i < num; ++i) {
					const IdxBucket idx_bucket = bucket_indices[i];
					// Only flush buffers on overflow
					if (IPS4O_UNLIKELY(buffers.isFull(idx_bucket))) {
						buffers.writeTo(idx_bucket, write);
						write += Cfg::kBlockSize;
						bucket_size[idx_bucket] += Cfg::kBlockSize;
					}
					buffers.push(idx_bucket, begin[i]);
				}
			}
			/**
   * Local classification phase.
   */

			template<bool kEqualBuckets>
			__attribute__((noinline)) ptrdiff_t classifyLocally(iterator begin,
			                                                    iterator end) {
				const size_t log_buckets = classifier.log_buckets_;
				switch (log_buckets) {
					case 1:
						return classifyLocally2<kEqualBuckets, 1>(begin, end);
					case 2:
						return classifyLocally2<kEqualBuckets, 2>(begin, end);
					case 3:
						return classifyLocally2<kEqualBuckets, 3>(begin, end);
					case 4:
						return classifyLocally2<kEqualBuckets, 4>(begin, end);
					case 5:
						return classifyLocally2<kEqualBuckets, 5>(begin, end);
					case 6:
						return classifyLocally2<kEqualBuckets, 6>(begin, end);
				}
				// HWY_ABORT("add case for log_buckets %zu\n", log_buckets);
                assert(false);
                return 0;
			}

#if IPS4O_SIMD
			HWY_INLINE Vec512<int32_t> X2PlusComp(Vec512<int32_t> vb,
			                                      Vec512<int32_t> items,
			                                      Vec512<int32_t> split) {
				const CappedTag<Cfg::value_type, 16> d;
				vb = Add(vb, vb);
				// Poor codegen here with clang, unnecessary shift of the predicate value.
				const auto gt = Gt(items, split);
				vb = Vec512<int32_t>{
				        _mm512_mask_add_epi32(vb.raw, gt.raw, vb.raw, Set(d, 1).raw)};
				//                      b[i] = 2 * b[i] + comp_(splitter(b[i]),
				//                      begin[i]);
				return vb;
			}
#endif

			// Called by classifyLocally; runtime classifier.log_buckets_ is 'frozen' into
			// a compile-time constant to allow skipping code without actually branching.
			template<bool kEqualBuckets, size_t kLogBuckets>
			__attribute__((noinline)) ptrdiff_t classifyLocally2(iterator begin,
			                                                     iterator end) {
				iterator write = begin;
				const ptrdiff_t num_buckets = 1l << (kLogBuckets + kEqualBuckets);
				constexpr const size_t kUnroll = 1; // AVX-512: TODO was 16 for avx 512
				static_assert(kUnroll <= Cfg::kBaseCaseSize, "Need >= 1 iteration");

				alignas(64) uint32_t bucket_indices[kUnroll];
				const size_t num = static_cast<size_t>(end - begin);

#if IPS4O_SIMD
				const CappedTag<Cfg::value_type, 16> d;
				const auto splitter0 = LoadU(d, classifier.splitter(0));
				const auto splitter1 = LoadU(d, classifier.splitter(16));
				const auto splitter2 = LoadU(d, classifier.splitter(32));
				const auto splitter3 = LoadU(d, classifier.splitter(48));
				const auto k1 = Set(d, 1);
				const auto sort_splitter0 = LoadU(d, classifier.sortedSplitter(0));
				const auto sort_splitter1 = LoadU(d, classifier.sortedSplitter(16));
				const auto sort_splitter2 = LoadU(d, classifier.sortedSplitter(32));
				const auto sort_splitter3 = LoadU(d, classifier.sortedSplitter(48));

				for (size_t i = 0; i < num; i += kUnroll) {
					auto vb = Set(d, 1);
					const auto items = LoadU(d, &begin[i]);

					// Before each loop body, vb is in [2^l, 2^(l+1)). Splitters fit in a
					// single AVX3 register:
					for (size_t l = 0; l < HWY_MIN(kLogBuckets, 4); ++l) {
						auto split =
						        Vec512<int32_t>{_mm512_permutexvar_epi32(vb.raw, splitter0.raw)};
						vb = X2PlusComp(vb, items, split);
					}
					// l=4: vb [16,31]
					if (kLogBuckets >= 5) {
						auto split = Vec512<int32_t>{
						        _mm512_permutexvar_epi32(Sub(vb, Set(d, 16)).raw, splitter1.raw)};
						vb = X2PlusComp(vb, items, split);
					}
					// l=5: vb [32,63]
					if (kLogBuckets >= 6) {
						auto split = Vec512<int32_t>{_mm512_permutex2var_epi32(
						        splitter2.raw, Sub(vb, Set(d, 32)).raw, splitter3.raw)};
						vb = X2PlusComp(vb, items, split);
					}

					// vb [2^kLogBuckets, 2^(kLogBuckets+1))
					if (kEqualBuckets) {
						// [0, 2^kLogBuckets)
						const auto idx = Sub(vb, Set(d, num_buckets / 2));
						Vec512<int32_t> split;
						if (kLogBuckets <= 4) {
							// Same speed as _mm512_permutex2var_epi32, but this special case
							// avoids having to load sort_splitter1.
							split = Vec512<int32_t>{
							        _mm512_permutexvar_epi32(idx.raw, sort_splitter0.raw)};
						} else if (kLogBuckets == 5) {
							split = Vec512<int32_t>{_mm512_permutex2var_epi32(
							        sort_splitter0.raw, idx.raw, sort_splitter1.raw)};
						} else {
							const auto splitL = Vec512<int32_t>{_mm512_permutex2var_epi32(
							        sort_splitter0.raw, idx.raw, sort_splitter1.raw)};
							const auto splitH = Vec512<int32_t>{_mm512_permutex2var_epi32(
							        sort_splitter2.raw, idx.raw, sort_splitter3.raw)};
							split = IfThenElse(Lt(idx, Set(d, 32)), splitL, splitH);
						}
						vb = Add(vb, vb);
						auto gt = _mm512_cmpge_epi32_mask(items.raw, split.raw);
						vb = Vec512<int32_t>{_mm512_mask_add_epi32(vb.raw, gt, vb.raw, k1.raw)};
						//   b[i] = 2 * b[i] + !comp_(begin[i], sortedSplitter(b[i] -
						//   num_buckets / 2));
					}

					vb = Sub(vb, Set(d, num_buckets));
					Store(vb, d, reinterpret_cast<int32_t *>(bucket_indices));
					size_t batch_size = HWY_MIN(kUnroll, num - i);
					ScatterBatch(bucket_indices, batch_size, begin + i, write);
				}
#else
				// TODO tail mngt
				for (size_t i = 0; i < num; i += kUnroll) {
					for (size_t j = 0; j < kUnroll; ++j) {
						IdxBucket b = 1;
						std::less<Cfg::value_type> comp_;
						for (size_t l = 0; l < kLogBuckets; ++l)
							b = 2 * b + comp_(*classifier.splitter(b), begin[i + j]);
						if (kEqualBuckets)
							b = 2 * b + !comp_(begin[i + j],
							                   *classifier.sortedSplitter(b - num_buckets / 2));
						bucket_indices[j] = b - num_buckets;
					}
					size_t batch_size = std::min(kUnroll, num - i);
					ScatterBatch(bucket_indices, batch_size, begin + i, write);
				}
#endif

				// Update bucket sizes to account for partially filled buckets
				for (int i = 0; i < num_buckets_; ++i) bucket_size[i] += buffers.size(i);

				return write - begin_;
			}

			__attribute__((noinline)) void sequentialClassification(
			        const bool use_equal_buckets) {
				const auto my_first_empty_block =
				        use_equal_buckets ? classifyLocally<true>(begin_, end_)
				                          : classifyLocally<false>(begin_, end_);

				// Find bucket boundaries
				ptrdiff_t sum = 0;
				bucket_start_[0] = 0;
				for (int i = 0; i < num_buckets_; ++i) {
					sum += bucket_size[i];
					bucket_start_[i + 1] = sum;
				}
				IPS4OML_ASSUME_NOT(bucket_start_[num_buckets_] != end_ - begin_);

				// Set write/read pointers for all buckets
				for (int bucket = 0; bucket < num_buckets_; ++bucket) {
					const auto start = Cfg::alignToNextBlock(bucket_start_[bucket]);
					const auto stop = Cfg::alignToNextBlock(bucket_start_[bucket + 1]);
					bucket_pointers[bucket].set(
					        start,
					        (start >= my_first_empty_block
					                 ? start
					                 : (stop <= my_first_empty_block ? stop : my_first_empty_block)) -
					                Cfg::kBlockSize);
				}
			}

			/**
   * Moves empty blocks to establish invariant:
   * All buckets must consist of full blocks followed by empty blocks.
   */
			__attribute__((noinline)) void moveEmptyBlocks(
			        const ptrdiff_t my_begin, const ptrdiff_t my_end,
			        const ptrdiff_t my_first_empty_block) {
				// Find range of buckets that start in this stripe
				const int bucket_range_start = [&](int i) {
					while (Cfg::alignToNextBlock(bucket_start_[i]) < my_begin) ++i;
					return i;
				}(0);

				/*
     * After classification, a stripe consists of full blocks followed by empty
     * blocks. This means that the invariant above already holds for all buckets
     * except those that cross stripe boundaries.
     *
     * The following cases exist:
     * 1)  The bucket is fully contained within one stripe.
     *     In this case, nothing needs to be done, just set the bucket pointers.
     *
     * 2)  The bucket starts in stripe i, and ends in stripe i+1.
     *     In this case, thread i moves full blocks from the end of the bucket
     * (from the stripe of thread i+1) to fill the holes at the end of its
     * stripe.
     *
     * 3)  The bucket starts in stripe i, crosses more than one stripe boundary,
     * and ends in stripe i+k. This is an extension of case 2. In this case,
     * multiple threads work on the same bucket. Each thread is responsible for
     * filling the empty blocks in its stripe. The left-most thread will take
     * the right-most blocks. Therefore, we count how many blocks are fetched by
     * threads to our left before moving our own blocks.
     */

				// Check if last bucket overlaps the end of the stripe
				const auto bucket_end = Cfg::alignToNextBlock(bucket_start_[num_buckets_]);
				const bool last_bucket_is_overlapping = bucket_end > my_end;

				// Case 1)
				for (int b = bucket_range_start;
				     b < num_buckets_ - last_bucket_is_overlapping; ++b) {
					const auto start = Cfg::alignToNextBlock(bucket_start_[b]);
					const auto stop = Cfg::alignToNextBlock(bucket_start_[b + 1]);
					auto read = stop;
					if (my_first_empty_block <= start) {
						// Bucket is completely empty
						read = start;
					} else if (my_first_empty_block < stop) {
						// Bucket is partially empty
						read = my_first_empty_block;
					}
					bucket_pointers[b].set(start, read - Cfg::kBlockSize);
				}

				// Cases 2) and 3)
				if (last_bucket_is_overlapping) {
					const int overlapping_bucket = num_buckets_ - 1;
					const auto bucket_start =
					        Cfg::alignToNextBlock(bucket_start_[overlapping_bucket]);

					// If it is a very large bucket, other threads will also move blocks
					// around in it (case 3) Count how many filled blocks are in this bucket
					ptrdiff_t flushed_elements_in_bucket = 0;
					if (bucket_start < my_begin) {
						abort();
					}

					// Threads to our left will move this many blocks (0 if we are the
					// left-most)
					// ptrdiff_t elements_reserved = 0;
					if (my_begin > bucket_start) {
						// Threads to the left of us get priority
						// elements_reserved =  my_begin - bucket_start - flushed_elements_in_bucket;

						// Count how many elements we flushed into this bucket
						flushed_elements_in_bucket += my_first_empty_block - my_begin;
					} else if (my_first_empty_block > bucket_start) {
						// We are the left-most thread
						// Count how many elements we flushed into this bucket
						flushed_elements_in_bucket += my_first_empty_block - bucket_start;
					}

					// Find stripe which contains last block of this bucket (off by one)
					// Also continue counting how many filled blocks are in this bucket

					// After moving blocks, this will be the first empty block in this bucket
					const auto first_empty_block_in_bucket =
					        bucket_start + flushed_elements_in_bucket;

					// This is the range of blocks we want to fill
					// auto write_ptr = begin_ + std::max(my_first_empty_block, bucket_start);
					// const auto write_ptr_end = begin_ + std::min(first_empty_block_in_bucket, my_end);

					// Set bucket pointers if the bucket starts in this stripe
					if (my_begin <= bucket_start) {
						bucket_pointers[overlapping_bucket].set(
						        bucket_start, first_empty_block_in_bucket - Cfg::kBlockSize);
					}
				}
			}

			/**
   * Computes which bucket can cause an overflow,
   * i.e., the last bucket that has more than one full block.
   */
			int computeOverflowBucket() {
				int bucket = num_buckets_ - 1;
				while (bucket >= 0 && (bucket_start_[bucket + 1] - bucket_start_[bucket]) <=
				                              Cfg::kBlockSize)
					--bucket;
				return bucket;
			}

			/**
   * Tries to read a block from read_bucket.
   */
			template<bool kEqualBuckets>
			int classifyAndReadBlock(const int read_bucket) {
				BucketPointers &bp = bucket_pointers[read_bucket];

				ptrdiff_t write, read;
				std::tie(write, read) = bp.decRead();

				if (read < write) {
					// No more blocks in this bucket
					return -1;
				}

				// Read block
				swap[0].readFrom(begin_ + read);

				return classifier.template classify<kEqualBuckets>(swap[0].data()[0]);
			}

			/**
   * Finds a slot for the block in the swap buffer. May or may not read another
   * block.
   */
			template<bool kEqualBuckets>
			int swapBlock(const ptrdiff_t max_off, const int dest_bucket,
			              const bool current_swap) {
				ptrdiff_t write, read;
				int new_dest_bucket;
				BucketPointers &bp = bucket_pointers[dest_bucket];
				do {
					std::tie(write, read) = bp.incWrite();
					if (write > read) {
						// Destination block is empty
						if (write >= max_off) {
							// Out-of-bounds; write to overflow buffer instead
							swap[current_swap].writeToBlock(overflow);
							overflow_ = &overflow;
							return -1;
						}

						// Write block
						swap[current_swap].writeTo(begin_ + write);
						return -1;
					}
					// Check if block needs to be moved
					new_dest_bucket =
					        classifier.template classify<kEqualBuckets>(begin_[write]);
				} while (new_dest_bucket == dest_bucket);

				// Swap blocks
				swap[!current_swap].readFrom(begin_ + write);
				swap[current_swap].writeTo(begin_ + write);

				return new_dest_bucket;
			}

			/**
   * Block permutation phase.
   */
			template<bool kEqualBuckets>
			__attribute__((noinline)) void permuteBlocks() {
				const auto num_buckets = num_buckets_;
				// Distribute starting points of threads
				int read_bucket = 0;
				// Not allowed to write to this offset, to avoid overflow
				const ptrdiff_t max_off =
				        Cfg::alignToNextBlock(end_ - begin_ + 1) - Cfg::kBlockSize;

				// Go through all buckets
				for (int count = num_buckets; count; --count) {
					int dest_bucket;
					// Try to read a block ...
					while ((dest_bucket = classifyAndReadBlock<kEqualBuckets>(read_bucket)) !=
					       -1) {
						bool current_swap = 0;
						// ... then write it to the correct bucket
						while ((dest_bucket = swapBlock<kEqualBuckets>(max_off, dest_bucket,
						                                               current_swap)) != -1) {
							// Read another block, keep going
							current_swap = !current_swap;
						}
					}
					read_bucket = (read_bucket + 1) % num_buckets;
				}
			}

            /// Fills margins from buffers.
			__attribute__((noinline)) void writeMargins(const int first_bucket,
			                                            const int last_bucket,
			                                            const int overflow_bucket,
			                                            const int swap_bucket,
			                                            const ptrdiff_t in_swap_buffer) {
				const bool is_last_level = end_ - begin_ <= Cfg::kSingleLevelThreshold;

				for (int i = first_bucket; i < last_bucket; ++i) {
					// Get bucket information
					const auto bstart = bucket_start_[i];
					const auto bend = bucket_start_[i + 1];
					const auto bwrite = bucket_pointers[i].getWrite();
					// Destination where elements can be written
					auto dst = begin_ + bstart;
					auto remaining = Cfg::alignToNextBlock(bstart) - bstart;

					if (i == overflow_bucket && overflow_) {
						// Is there overflow?

						// Overflow buffer has been written => write pointer must be at end of
						// bucket
						IPS4OML_ASSUME_NOT(Cfg::alignToNextBlock(bend) != bwrite);

						auto src = overflow_->data();
						// There must be space for at least BlockSize elements
						IPS4OML_ASSUME_NOT((bend - (bwrite - Cfg::kBlockSize)) + remaining <
						                   Cfg::kBlockSize);
						auto tail_size = Cfg::kBlockSize - remaining;

						// Fill head
						// memcpy(dst, src, remaining * sizeof(value_type));
						memcpy(dst, src, remaining);
						src += remaining;
						remaining = std::numeric_limits<ptrdiff_t>::max();

						// Write remaining elements into tail
						dst = begin_ + (bwrite - Cfg::kBlockSize);
						dst = std::move(src, src + tail_size, dst);
					} else if (i == swap_bucket && in_swap_buffer) {
						// Bucket of last block in this thread's area => write swap buffer
						auto src = swap[0].data();
						// All elements from the buffer must fit
						IPS4OML_ASSUME_NOT(in_swap_buffer > remaining);

						// Write to head
						dst = std::move(src, src + in_swap_buffer, dst);
						remaining -= in_swap_buffer;
					} else if (bwrite > bend && bend - bstart > Cfg::kBlockSize) {
						// Final block has been written => move excess elements to head
						IPS4OML_ASSUME_NOT(Cfg::alignToNextBlock(bend) != bwrite);

						auto src = begin_ + bend;
						auto head_size = bwrite - bend;
						// Must fit, no other empty space left
						IPS4OML_ASSUME_NOT(head_size > remaining);

						// Write to head
						dst = std::move(src, src + head_size, dst);
						remaining -= head_size;
					}

					// Write elements from buffers
					for (int t = 0; t < 1; ++t) {
						auto src = buffers.data(i);
						auto count = buffers.size(i);

						if (count <= remaining) {
							dst = std::move(src, src + count, dst);
							remaining -= count;
						} else {
							// memcpy(dst, src, remaining * sizeof(value_type));
							memcpy(dst, src, remaining);
							src += remaining;
							count -= remaining;
							remaining = std::numeric_limits<ptrdiff_t>::max();

							dst = begin_ + bwrite;
							dst = std::move(src, src + count, dst);
						}

						buffers.reset(i);
					}

					// Perform final base case sort here, while the data is still cached
					if (is_last_level || ((bend - bstart <= 2 * Cfg::kBaseCaseSize))) {
						baseCaseSort(begin_ + bstart, begin_ + bend);
					}
				}
			}
		};

		template<class It>
		__attribute__((noinline)) bool sortSimpleCases(It begin, It end) {
			if (begin == end) {
				return true;
			}

			std::less<Cfg::value_type> comp;

			// If last element is not smaller than first element,
			// test if input is sorted (input is not reverse sorted).
			if (!comp(*(end - 1), *begin)) {
				if (std::is_sorted(begin, end)) {
					return true;
				}
			} else {
				// Check whether the input is reverse sorted.
				for (It it = begin; (it + 1) != end; ++it) {
					if (comp(*it, *(it + 1))) {
						return false;
					}
				}
				std::reverse(begin, end);
				return true;
			}

			return false;
		}

	}// namespace detail

	template<class It>
	void sort(It begin, It end) {
		// Negligible cost.
		if (detail::sortSimpleCases(begin, end)) {
			return;
		}

		if ((end - begin) <= 16 * detail::Cfg::kBaseCaseSize) {
			detail::baseCaseSort(begin, end);
			return;
		}

		alignas(64) detail::Sorter sorter;
        // TODO fix this iterator nonsence
		sorter.sequential(&(*begin), &(*end));
	}

}// namespace ips4o
