#pragma once
// source https://github.com/bertdobbelaere/SorterHunter
// TODO improvements:  https://github.com/HJLebbink/SorterHunter (256 hannels)
// TODO add a function to `Pair_t` which checks if its a simd swap pattern
//          so this is only include swaps of equal size.
// TODO config is contantexpr
// Mutation which is a swap pattern

#include <stdint.h>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <vector>
#include <algorithm>

#include "random.h"

using namespace cryptanalysislib;


#define NMAX (64)
#define PARWORDSIZE (64)

using std::size_t;

typedef uint64_t SortWord_t;///< Needs to contain at least NMAX bits
typedef uint64_t BPWord_t;  ///< Bit-parallel operation word, needs to contain at least PARWORDSIZE bits
typedef uint32_t u32;
typedef uint8_t u8;

/// Compare and Exchange (CE) representation
struct Pair_t {
public:
	///< low and high line indices connected by the element
	[[nodiscard]] constexpr inline bool operator==(const Pair_t &p) const noexcept { return (lo == p.lo) && (hi == p.hi); }
	[[nodiscard]] constexpr inline bool operator!=(const Pair_t &p) const noexcept { return (lo != p.lo) || (hi != p.hi); }

	u8 lo=0, hi=0;
};


// forward decl
class ClusterGroup;
class OCH_t;

typedef std::vector<Pair_t> Network_t;
typedef std::vector<SortWord_t> SinglePatternList_t;
typedef std::vector<BPWord_t> BitParallelList_t;

static u32 Verbosity;

///< "Alphabet" of possible CEs defined by their vertical positions.
static Network_t alphabet;

///  Test vectors filled with input data sets fed to parallel sorter tester
BitParallelList_t parallelpatterns_from_prefix;

///< Number of different mutation types
#define NMUTATIONTYPES 6 

///< Helper variable to quickly pick a mutation with the requested probability.
std::vector<u8> mutationSelector; 

///< Current core network: evolving section between prefix and postfix. For symmetric networks, mirrored pair (if not coinciding) is omitted.
Network_t pairs; 
Network_t se; ///< Symmetrical expansion of current network
Network_t newpairs;
Network_t prefix; ///< Fixed, greedy, hybrid or empty prefix network
Network_t postfix; ///< Fixed or empty postfix network

// Random generation defs
///< Random index from vector
#define RANDIDX(v) (rng(v.size()))
///< Random element from vector
#define RANDELEM(v) (v[RANDIDX(v)])


/// Orthogonal Convex Hull, to keep track of unmatched (size,depth) combinations of the networks we found
class OCH_t {
public:
	OCH_t() {}

	///  Clear OCH data
	constexpr void clear() noexcept {
		och.clear();
	}

	/// Add a (size, depth) pair to the OCH computation
	/// \param l length of network found
	/// \param d depth of network found
	/// \param true if the network is an "improvement" i.e. belongs to the
	///     updated set of OCH pairs that minimize both criteria.
	bool improved(u32 size, u32 depth) {
		bool matched = false;

		for (size_t k = 0; k < och.size(); k++) {
			if ((size >= och[k].size) && (depth >= och[k].depth))
				matched = true;
		}

		if (matched)
			return false;

		std::vector<OCH_Entry> newch;
		OCH_Entry ce;
		ce.size = size;
		ce.depth = depth;
		newch.push_back(ce);

		for (size_t k = 0; k < och.size(); k++) {
			if ((och[k].size < size) || (och[k].depth < depth)) {
				newch.push_back(och[k]);
			}
		}

		och = newch;
		return true;
	}

	/// Print best performing (length, depth) pairs found so far
	void print() const {
		printf("Most performant: [");
		for (size_t k = 0; k < och.size(); k++) {
			printf("(%u,%u)", och[k].size, och[k].depth);
			printf("%c", (k < (och.size() - 1)) ? ',' : ']');
		}
		printf("\r\n");
	}

private:
	struct OCH_Entry {
		u32 size = 0;
		u32 depth = 0;
	};

	std::vector<OCH_Entry> och{};
};

///< "Best performing" network list found so far
OCH_t conv_hull;


class SortHunterConfig {
public:
	///< Treat sorting network as symmetric or not
	const bool use_symmetry = true;

	///< "Uphill" step inserts duplicate CE if not in final layer.
	const bool force_valid_uphill_step = true;

	///< Problem dimension, i.e. number of inputs to be sorted
	const u8 N = 20;

	///< Adds a random pair (and its symmetric complement for symmetric networks)
	/// every x iterations
	const u32 EscapeRate = 1000;

	///< Maximum allowed number of mutations in evolution step
	const u32 MaxMutations = 3;


	///< Fixed prefix to use (if applicable)
	const Pair_t FixedPrefix[20] = {{.lo=0,.hi=1},{.lo=2,.hi=3},{.lo=4,.hi=5},{.lo=6,.hi=7},{.lo=8,.hi=9},{.lo=10,.hi=11},{.lo=12,.hi=13},{.lo=14,.hi=15},{.lo=16,.hi=17},{.lo=18,.hi=19},{.lo=0,.hi=2},{.lo=1,.hi=3},{.lo=4,.hi=6},{.lo=5,.hi=7},{.lo=8,.hi=10},{.lo=9,.hi=11},{.lo=12,.hi=14},{.lo=13,.hi=15},{.lo=16,.hi=18},{.lo=17,.hi=19}};
	///< Type of prefix used (0=none, 1=fixed, 2=greedy)
	const u32 PrefixType = 2;

    // Pair_t InitialNetwork[0];
	///< Initial starting point of network
	Pair_t InitialNetwork[121] = {{.lo=4,.hi=17},{.lo=6,.hi=19},{.lo=15,.hi=22},{.lo=1,.hi=8},{.lo=14,.hi=16},{.lo=7,.hi=9},{.lo=7,.hi=14},{.lo=9,.hi=16},{.lo=0,.hi=2},{.lo=21,.hi=23},{.lo=10,.hi=11},{.lo=12,.hi=13},{.lo=1,.hi=15},{.lo=8,.hi=22},{.lo=13,.hi=17},{.lo=6,.hi=10},{.lo=11,.hi=19},{.lo=4,.hi=12},{.lo=9,.hi=15},{.lo=8,.hi=14},{.lo=14,.hi=15},{.lo=8,.hi=9},{.lo=3,.hi=18},{.lo=5,.hi=20},{.lo=20,.hi=23},{.lo=0,.hi=3},{.lo=1,.hi=7},{.lo=16,.hi=22},{.lo=2,.hi=18},{.lo=5,.hi=21},{.lo=2,.hi=13},{.lo=10,.hi=21},{.lo=11,.hi=20},{.lo=3,.hi=12},{.lo=12,.hi=21},{.lo=2,.hi=11},{.lo=17,.hi=18},{.lo=5,.hi=6},{.lo=3,.hi=6},{.lo=17,.hi=20},{.lo=0,.hi=4},{.lo=19,.hi=23},{.lo=18,.hi=23},{.lo=0,.hi=5},{.lo=1,.hi=5},{.lo=18,.hi=22},{.lo=14,.hi=20},{.lo=3,.hi=9},{.lo=15,.hi=21},{.lo=2,.hi=8},{.lo=0,.hi=1},{.lo=22,.hi=23},{.lo=9,.hi=11},{.lo=12,.hi=14},{.lo=3,.hi=5},{.lo=18,.hi=20},{.lo=6,.hi=7},{.lo=16,.hi=17},{.lo=13,.hi=19},{.lo=4,.hi=10},{.lo=8,.hi=10},{.lo=13,.hi=15},{.lo=17,.hi=19},{.lo=4,.hi=6},{.lo=8,.hi=9},{.lo=14,.hi=15},{.lo=12,.hi=16},{.lo=7,.hi=11},{.lo=1,.hi=3},{.lo=20,.hi=22},{.lo=10,.hi=18},{.lo=5,.hi=13},{.lo=11,.hi=17},{.lo=6,.hi=12},{.lo=2,.hi=4},{.lo=19,.hi=21},{.lo=7,.hi=13},{.lo=10,.hi=16},{.lo=6,.hi=8},{.lo=15,.hi=17},{.lo=9,.hi=12},{.lo=11,.hi=14},{.lo=19,.hi=20},{.lo=3,.hi=4},{.lo=21,.hi=22},{.lo=1,.hi=2},{.lo=2,.hi=3},{.lo=20,.hi=21},{.lo=7,.hi=10},{.lo=13,.hi=16},{.lo=14,.hi=16},{.lo=7,.hi=9},{.lo=18,.hi=19},{.lo=4,.hi=5},{.lo=15,.hi=18},{.lo=5,.hi=8},{.lo=17,.hi=19},{.lo=4,.hi=6},{.lo=19,.hi=20},{.lo=3,.hi=4},{.lo=11,.hi=13},{.lo=10,.hi=12},{.lo=12,.hi=15},{.lo=8,.hi=11},{.lo=5,.hi=7},{.lo=16,.hi=18},{.lo=13,.hi=14},{.lo=9,.hi=10},{.lo=14,.hi=15},{.lo=8,.hi=9},{.lo=10,.hi=11},{.lo=12,.hi=13},{.lo=13,.hi=14},{.lo=9,.hi=10},{.lo=16,.hi=17},{.lo=6,.hi=7},{.lo=11,.hi=12},{.lo=7,.hi=8},{.lo=15,.hi=16},{.lo=5,.hi=6},{.lo=17,.hi=18}};

	///< Size of greedy prefix (if applicable)
	const u32 GreedyPrefixSize = 10;

	///< Random seed
	const uint64_t RandomSeed = 0;

	///< Return to initial conditions each ... iterations (0=never)
	const uint64_t RestartRate = 0;

	///< Overall verbosity level: 0:minimal, 1:moderate, 2:high, >2:debug
	const u32 Verbosity = 1;

    ///< Relative probabilities for each mutation type
    /// # Relative probabilities of mutation types (integer >=0). Not all must 
    /// be 0. Don't use huge values (total<10000 is no issue), as a table is 
    /// allocated for the probability distribution 
    ///
    /// WeigthRemovePair = 1              # Remove a random pair
    /// WeigthSwapPairs  = 1              # Swap two random pairs
    /// WeigthReplacePair = 0             # Replace random pair with another pair
    /// WeightCrossPairs = 1              # Cross two pairs at random positions
    /// WeightSwapIntersectingPairs = 2   # Swap pairs in neighbouring layers sharing a connection
    /// WeightReplaceHalfPair = 1         # Replace one of the two connections of a random pair
    u32 mutation_type_weights[NMUTATIONTYPES] = {1,1,0,1,2,1}; 
};

constexpr static SortHunterConfig config;

constexpr static u32 computeDepth(const Network_t &nw) noexcept {
	std::vector<SortWord_t> layers;
	int nlayers = 0;

	for (size_t k = 0; k < nw.size(); k++) {
		u32 i = nw[k].lo;
		u32 j = nw[k].hi;
		int matchidx = nlayers;
		int idx = nlayers - 1;
		while (idx >= 0) {
			if ((layers[idx] & ((1ULL << i) | (1ULL << j))) == 0) {
				matchidx = idx;
			} else {
				break;
			}
			idx--;
		}
		if (matchidx >= nlayers) {
			layers.push_back(0);
			nlayers++;
		}
		layers[matchidx] |= 1ULL << i;
		layers[matchidx] |= 1ULL << j;
	}

	return nlayers;
}

/// \param nw[in]
constexpr static void printnw(const Network_t &nw) noexcept {
	printf("[");
	for (size_t k = 0; k < nw.size(); k++) {
		printf("(%u,%u)", nw[k].lo, nw[k].hi);
		printf("%c", ((k + 1) < nw.size()) ? ',' : ']');
	}
	printf("}\r\n");
}

/// \param ninputs[in]:
/// \param inpairs[in]:
/// \param outpairs[out]:
constexpr static void symmetricExpansion(u8 ninputs,
                                         const Network_t &inpairs,
                                         Network_t &outpairs) noexcept {
	outpairs.clear();
	for (size_t k = 0; k < inpairs.size(); k++) {
		outpairs.push_back(inpairs[k]);
		// Don't duplicate pair that maps on itself
		if ((inpairs[k].lo + inpairs[k].hi) != (ninputs - 1)) {
			Pair_t sp = {(u8) (ninputs - 1 - inpairs[k].hi), (u8) (ninputs - 1 - inpairs[k].lo)};
			outpairs.push_back(sp);
		}
	}
}

/// \param nw1[in]:
/// \param nw2[in]:
/// \param result[out]:
constexpr static void concatNetwork(const Network_t &nw1,
                                    const Network_t &nw2,
                                    Network_t &result) noexcept {
	result = nw1;
	result.insert(result.end(), nw2.begin(), nw2.end());
}

/// \param dst[out]:
/// \param src[in]:
constexpr static void appendNetwork(Network_t &dst,
                                    const Network_t &src) noexcept {
	if (src.size() > 0) {
		dst.insert(dst.end(), src.begin(), src.end());
	}
}


/// Given a prefix containing of 0 or more network pairs, computes the possible outputs of the (partially ordered) output set.
/// For an empty prefix, the result will contain 2**N patterns.
/// If the prefix is in itself a valid sorter, the result will contain N+1 patterns.
/// @param ninputs Number of inputs to the partially ordered network
/// @param prefix Prefix to process
/// @param patterns [OUT] List of output patterns
void computePrefixOutputs(u8 ninputs,
                          const Network_t &prefix,
                          SinglePatternList_t &patterns) noexcept ;

/// Converts a set of prefix output patterns to a bit parallel data structure to speed up testing of the "postfix" network.
/// The word size for packing is given by PARWORDSIZE
/// \param ninputs Number of inputs to the partially ordered network
/// \param singles Prefix output patterns to convert
/// \param use_symmetry Optimize using symmetry
/// \param parallels [OUT] Bit parallel representations of the patterns
void convertToBitParallel(u8 ninputs,
                          const SinglePatternList_t &singles,
                          bool use_symmetry,
                          BitParallelList_t &parallels) noexcept;

/// Tries to create a partially ordered network that (approximately) minimizes the number of possible outputs.
/// Function is called with the list of fixed pairs (optional, empty list if none).
/// Caller should take care of symmetry of fixed pairs.
/// \param ninputs Number of inputs to the partially ordered network
/// \param maxpairs Maximum number of pairs in the prefix
/// \param use_symmetry Set to true of the computed prefix needs to be symmetrical
/// \param prefix Contains fixed pairs as input (if any) and best prefix as output
/// \param rndgen Random number generator for shuffling
/// \return Number of outputs from partially ordered network (ninputs+1 if fully sorted, 2**ninputs worst case)
SortWord_t createGreedyPrefix(u8 ninputs,
                              u32 maxpairs,
                              bool use_symmetry,
                              Network_t &prefix) noexcept;

/// Replaces a *sorted* list of patterns applied to a network containing a single CE by the sorted list of output patterns of that network.
/// The sort order is low to high, a pattern represents the binary representation of an input/output state
/// Restriction to sorted pattern lists allows to compute the output list in linear time.
/// \param patterns [IN/OUT] Input and output list of patterns, sorted.
/// \param pair Representation of CE to apply
static void swap_sortedpatterns(SinglePatternList_t &patterns,
                                const Pair_t &pair) noexcept;


/**
 * Helper class to efficiently compute partially ordered pattern sets.
 * The inputs of the network are grouped together in clusters that have been connected by CEs
 * Initial clusters contain just one input (no CEs added yet). 
 * Each cluster has a set of output patterns that it leaves behind. The global set of output patterns
 * is at each time defined by the bitwise "ored" combinations of the outputs that all clusters produce together.
 * While adding CEs to the network, clusters are combined into larger clusters with a shrinking total number of
 * output patterns. If the CE is added that combines the last two clusters, only one cluster will remain.
 * If after that sufficient new CEs are added, only ninputs+1 patterns will remain, meaning that the network is fully sorted.
 */
class ClusterGroup {
public:
	/// Initialize an empty cluster group
	ClusterGroup(const u8 n) noexcept {
		ninputs = n;
		patternlists = new SinglePatternList_t[ninputs];
		masks = new SortWord_t[ninputs];
		clusterAlloc = new u8[ninputs];
		clear();
	}

	/// Copy constructor
	ClusterGroup(const ClusterGroup &cg) noexcept {
		ninputs = cg.ninputs;
		patternlists = new SinglePatternList_t[ninputs];
		masks = new SortWord_t[ninputs];
		clusterAlloc = new u8[ninputs];
		for (u32 k = 0; k < ninputs; k++) {
			patternlists[k] = cg.patternlists[k];
			masks[k] = cg.masks[k];
			clusterAlloc[k] = cg.clusterAlloc[k];
		}
	}

	/// Assignment of cluster groups to eachother
	const ClusterGroup &operator=(const ClusterGroup &cg) noexcept {
		ninputs = cg.ninputs;
		for (u32 k = 0; k < ninputs; k++) {
			patternlists[k] = cg.patternlists[k];
			masks[k] = cg.masks[k];
			clusterAlloc[k] = cg.clusterAlloc[k];
		}
		return *this;
	}

	/// Set initial state:
	/// each input corresponds one to one with its own cluster. The cluster
	/// has two possible output patterns: the all 0 pattern, and a single
	/// 1 bit at the bit position of the corresponding input.
	void clear() noexcept {
		for (u32 k = 0; k < ninputs; k++) {
			clusterAlloc[k] = k;
			masks[k] = 1ULL << k;
			patternlists[k].clear();
			patternlists[k].push_back(0);
			patternlists[k].push_back(1ULL << k);
		}
	}

	/// Reduces the number of patterns represented by appending a single CE
	/// to the network. If the CE's lines belong to different clusters, the
	/// clusters are merged first.
	/// \param p CE represented by its input/output lines
	void preSort(Pair_t p) noexcept {
		u32 ci_idx = clusterAlloc[p.lo];
		u32 cj_idx = clusterAlloc[p.hi];

		if (ci_idx != cj_idx) {
			combine(ci_idx, cj_idx);
		}
		swap_sortedpatterns(patternlists[ci_idx], p);
	}

	/// Compute the list of output patterns that can leave the network composed of all
	/// clusters remaining. This is done by "oring" together output combinations of all remaining clusters.
	/// @param patterns [OUT] pattern list created (not lexographically sorted)
	void computeOutputs(SinglePatternList_t &patterns) const noexcept {
		const static SinglePatternList_t *pLists[NMAX];
		int n_to_combine = 0;

		for (u32 k = 0; k < ninputs; k++) {
			if (masks[k] != 0)
				pLists[n_to_combine++] = &patternlists[k];
		}

		assert(n_to_combine > 0);

		int level = 0;
		size_t indices[NMAX] = {0};
		SortWord_t outmasks[NMAX] = {0};
		patterns.clear();

		while (level >= 0) {
			if (indices[level] < pLists[level]->size()) {
				if (level == 0)
					outmasks[level] = (*pLists[level])[indices[level]];
				else
					outmasks[level] = outmasks[level - 1] | (*pLists[level])[indices[level]];
				if (level < (n_to_combine - 1)) {
					indices[level + 1] = 0;
					indices[level]++;
					level++;
				} else {
					patterns.push_back(outmasks[level]);
					indices[level]++;
				}
			} else {
				level--;
			}
		}
	}

	/// Compute number of output patterns that would be produced by call
	/// to computeOutputs
	SortWord_t outputSize() const noexcept {
		SortWord_t prod = 1;

		for (u32 k = 0; k < ninputs; k++) {
			if (masks[k] != 0)
				prod *= patternlists[k].size();
		}

#if 1
		if (prod == 0)// Special case for N=NMAX, dirty hack avoiding wrap-around to 0 of empty network: set size to one less.
			prod -= 1;
#endif

		return prod;
	}

	bool isSameCluster(Pair_t p) const noexcept {
		u32 ci_idx = clusterAlloc[p.lo];
		u32 cj_idx = clusterAlloc[p.hi];
		return ci_idx == cj_idx;
	}
	/// Clean up cluster group
	~ClusterGroup() {
		delete[] patternlists;
		delete[] masks;
		delete[] clusterAlloc;
	}

private:
	/// Combines two clusters to form a larger cluster.
	/// The output pattern list is produced by bitwise "oring" of both
	/// original pattern lists
	/// \param ci_idx First cluster index (new result cluster)
	/// \param cj_idx Second cluster index (will no longer be used)
	void combine(u8 ci_idx,
	             u8 cj_idx) noexcept {
		SinglePatternList_t &p1 = patternlists[ci_idx];
		SinglePatternList_t &p2 = patternlists[cj_idx];

		for (u32 k = 0; k < ninputs; k++)
			if (clusterAlloc[k] == cj_idx)
				clusterAlloc[k] = ci_idx;// ci will take over

		masks[ci_idx] |= masks[cj_idx];
		SinglePatternList_t cp;
		/*
        	 * Combined cluster's output patterns are here simply generated by producing all
        	 * patterns, first disregarding their final order and sorting them afterwards.
        	 * At first, I had an algorithm in place that broke the masks into chunks allowing
        	 * in order generation that had lower theoretical complexity. For practical sizes however
        	 * a quicksort proved a faster and simpler alternative. (and probably has less bugs :-) )
        	 */
		for (size_t i = 0; i < p1.size(); i++)
			for (size_t j = 0; j < p2.size(); j++)
				cp.push_back(p1[i] | p2[j]);
		std::sort(cp.begin(), cp.end());// Keep the new output set sorted
		p1 = cp;
		masks[cj_idx] = 0;
		p2.clear();
	}

    ///< Sorted list of output patterns from each cluster of lines
	SinglePatternList_t *patternlists;

    ///< Masks for each cluster marking the applicable lines for each cluster
	SortWord_t *masks;

    ///< Allocations of lines to clusters
	u8 *clusterAlloc;
    
    ///< Total number of inputs (and outputs) of the network
	u8 ninputs;
};


void computePrefixOutputs(u8 ninputs,
                          const Network_t &prefix,
                          SinglePatternList_t &patterns) noexcept {
	ClusterGroup cg(ninputs);
	Network_t todo = prefix;

	while (todo.size() > 0) {
		cg.preSort(todo[0]);// Process first remaining pair, combine related clusters

		Network_t postponed;
		SortWord_t visitmask = 0;
		for (size_t k = 1; k < todo.size(); k++)// Skip 1st element, we just handled it
		{
			Pair_t el = todo[k];
			SortWord_t elmask = (1ull << el.lo) | (1ull << el.hi);

			if (((visitmask & elmask) == 0) && cg.isSameCluster(el)) {
				// Prioritize elements that can be applied without extra cluster joining.
				// The goal is to reduce memory requirements where possible
				cg.preSort(el);
			} else {
				// Postpone till next iteration any element that requires additional clusters to be joined or has dependencies to unprocessed elements
				postponed.push_back(el);
			}
			visitmask |= elmask;
		}
		todo = postponed;
	}

	cg.computeOutputs(patterns);
}

/**
 * For symmetric networks, any network that sorts a pattern successfully will also sort the reverse of the inverse,
 * i.e. if a symmetric network sorts '00101111', if will also sort '00001011'
 * This function is used to discard the largest of those patterns.
 */
static bool hasSmallerMirror(u8 ninputs, SortWord_t w) noexcept {
	SortWord_t rw = 0u;
	SortWord_t tmp = w;
	for (u32 k = 0; k < ninputs; k++) {
		rw <<= 1;
		rw |= ~tmp & 1u;
		tmp >>= 1;
	}
	return w > rw;
}

///< ninputs lowest bit to be set
static SortWord_t all_n_inputs_mask;
constexpr static bool isSorted(const u8 ninputs, 
                               SortWord_t w) noexcept {
    (void)ninputs;
	w = ~w & all_n_inputs_mask;
	return (w & (w + 1)) == 0;
}


void convertToBitParallel(u8 ninputs,
                          const SinglePatternList_t &singles, 
                          bool use_symmetry, 
                          BitParallelList_t &parallels) noexcept {
	u32 level = 0;
	static BPWord_t buffer[NMAX];
	parallels.clear();

	all_n_inputs_mask = 0ULL;
	for (u32 k = 0; k < ninputs; k++) {
		all_n_inputs_mask |= 1ULL << k;
	}

	for (size_t idx = 0; idx < singles.size(); idx++) {
		SortWord_t w = singles[idx];
		if (use_symmetry && hasSmallerMirror(ninputs, w)) {
            // Complement of reverse word is smaller, skip this vector if the 
            // network is symmetric
			continue;
		}

		if (isSorted(ninputs, w)) {
            // Already sorted pattern will not be affected by sorting operation 
            // - useless as test vector
			continue;
		}

		for (u32 b = 0; b < ninputs; b++) {
			buffer[b] <<= 1;
			buffer[b] |= (w & 1);
			w >>= 1;
		}
		level++;

		if (level >= PARWORDSIZE) {
			for (u32 b = 0; b < ninputs; b++) {
				parallels.push_back(buffer[b]);
				buffer[b] = 0;// Needed ? Probably not, but cleaner.
			}
			level = 0;
		}
	}
	if (level > 0) {
		for (u32 b = 0; b < ninputs; b++) {
			parallels.push_back(buffer[b]);
		}
	}

	if (Verbosity > 2) {
		printf("Debug: Pattern conversion: %lu single inputs -> %lu parallel words (%u * %lu) (symmetry:%d)\n", singles.size(), parallels.size(), ninputs, parallels.size() / ninputs, use_symmetry);
	}
}

/**
 * Initialize alphabet of CEs. 
 * @param ninputs Number of network inputs
 * @param use_symmetry If set to true duplicates due to mirroring will be omitted
 */
static void initAlphabet(u8 ninputs, bool use_symmetry) {
	alphabet.clear();
	for (u32 i = 0; i < (ninputs - 1u); i++)
		for (u32 j = i + 1; j < ninputs; j++) {
			u8 isym = ninputs - 1 - j;
			u8 jsym = ninputs - 1 - i;

			if (!use_symmetry || (isym > i) || ((isym == i) && (jsym >= j))) {
				Pair_t p = {(u8) i, (u8) j};
				alphabet.push_back(p);
			}
		}
}

SortWord_t createGreedyPrefix(u8 ninputs,
                              u32 maxpairs, 
                              bool use_symmetry, 
                              Network_t &prefix) noexcept {
	if (Verbosity > 2) {
		printf("Creating greedy prefix. Initial prefix size = %lu, max prefix size %u.\n", prefix.size(), maxpairs);
	}
	// RandGen_t &rndgen(0);
	ClusterGroup cg(ninputs);
	initAlphabet(ninputs, use_symmetry);

	for (size_t k = 0; k < prefix.size(); k++)
		cg.preSort(prefix[k]);
	SortWord_t currentsize = cg.outputSize();

	while ((prefix.size() < maxpairs) || (use_symmetry && (prefix.size() < (maxpairs - 1)))) {
		Network_t ashuf = alphabet;
		Pair_t best = {0, 1};
		// std::shuffle(ashuf.begin(), ashuf.end(), rndgen);
        // TODO
		std::random_shuffle(ashuf.begin(), ashuf.end());
		SortWord_t minsize = currentsize;

		ClusterGroup cgbest = cg;
		SortWord_t minfuturesize = currentsize;
		for (size_t k = 0; k < alphabet.size(); k++) {
			ClusterGroup cgnew = cg;
			cgnew.preSort(ashuf[k]);
			if (use_symmetry && ((ashuf[k].lo + ashuf[k].hi) != (ninputs - 1))) {
				Pair_t p = {(u8) (ninputs - 1 - ashuf[k].hi), (u8) (ninputs - 1 - ashuf[k].lo)};
				cgnew.preSort(p);
			}
			SortWord_t newsize = cgnew.outputSize();
			SortWord_t futuresize = newsize;
			if (futuresize < minfuturesize) {
				minsize = newsize;
				minfuturesize = futuresize;
				best = ashuf[k];
				cgbest = cgnew;
			}
		}

		if (minsize >= currentsize) {
			// Found no improvement
			if (Verbosity > 2) {
				printf("Greedy algorithm: no further improvement.\n");
			}
			break;
		}
		cg = cgbest;
		if (Verbosity > 2) {
			printf("Greedy: adding pair (%u,%u)\n", best.lo, best.hi);
		}
		prefix.push_back(best);
		if (use_symmetry && ((best.lo + best.hi) != (ninputs - 1))) {
			Pair_t p = {(u8) (ninputs - 1 - best.hi), (u8) (ninputs - 1 - best.lo)};
			if (Verbosity > 2) {
				printf("Greedy: adding symmetric pair (%u,%u)\n", p.lo, p.hi);
			}
			prefix.push_back(p);
		}
		currentsize = minsize;
	}
	return currentsize;
}


/// Replaces a *sorted* list of patterns applied to a network containing a single CE by the sorted list of output patterns of that network.
/// The sort order is low to high, a pattern represents the binary representation of an input/output state
/// Restriction to sorted pattern lists allows to compute the output list in linear time.
/// \param patterns [IN/OUT] Input and output list of patterns, sorted.
/// \param pair Representation of CE to apply
static void swap_sortedpatterns(SinglePatternList_t &patterns,
                                const Pair_t &pair) noexcept {
	SortWord_t p = 1ULL << pair.lo;
	SortWord_t q = 1ULL << pair.hi;
	SortWord_t mask = p | q;

	SinglePatternList_t res;

	size_t idxp = 0;
	size_t idxnp = 0;
	size_t l = patterns.size();
	SortWord_t last = -1;

	while ((idxp < l) && ((patterns[idxp] & mask) != p)) { idxp++; }
	while ((idxnp < l) && ((patterns[idxnp] & mask) == p)) { idxnp++; }

	while ((idxnp < l) && (idxp < l)) {
		SortWord_t a = patterns[idxp] ^ mask;
		SortWord_t b = patterns[idxnp];
		if (a < b) {
			if (a != last) {
				res.push_back(a);
				last = a;
			}
			idxp++;
			while ((idxp < l) && ((patterns[idxp] & mask) != p)) { idxp++; }
		} else {
			if (a != last) {
				res.push_back(b);
				last = b;
			}
			idxnp++;
			while ((idxnp < l) && ((patterns[idxnp] & mask) == p)) { idxnp++; }
		}
	}
	while (idxnp < l) {
		res.push_back(patterns[idxnp++]);
	}
	while (idxp < l) {
		res.push_back(patterns[idxp++] ^ mask);
	}

	patterns = res;
}


/// Send a bit-parallel set of test patterns through a sorting network. Maximum PARWORDSIZE patterns are processed
/// together.
/// 'Data' contains N words. Each bit position corresponds to an independent data set {0,1}^N to be sorted
/// Bit level truth table:
/// In    Out
/// 00 ->  00
/// 01 ->  01
/// 10 ->  01 ("swap")
/// 11 ->  11
/// \param data Input/output vectors
/// \param nw Network to be tested
constexpr void applyBitParallelSort(BPWord_t data[], 
                                    const Network_t &nw) noexcept {
	const size_t l = nw.size();
	for (size_t n = 0; n < l; n++) {
		const u32 i = nw[n].lo;
		const u32 j = nw[n].hi;
		BPWord_t iold = data[i];
		data[i] &= data[j];
		data[j] |= iold;
	}
}

/**
 * Initialise test vectors with patterns produced by the prefix.
 * Test vectors are stored in parallelpatterns_from_prefix
 * @param prefix Network prefix to use
 */
void prepareTestVectorsFromPrefix(const Network_t &prefix) noexcept {
	constexpr bool is_even = ((config.N % 2) == 0);

	SinglePatternList_t singles;
	computePrefixOutputs(config.N, prefix, singles);
    // Shuffle test vectors: improve probability of early rejection of non-sorters
	// TODO std::shuffle(singles.begin(), singles.end(), mtRand);
	std::random_shuffle(singles.begin(), singles.end());

	convertToBitParallel(config.N, singles, config.use_symmetry && is_even, parallelpatterns_from_prefix);
}

/// Initialize "alphabet" of CEs to use
void initalphabet() {
	alphabet.clear();
	for (u32 i = 0; i < (config.N - 1u); i++)
		for (u32 j = i + 1; j < config.N; j++) {
			u32 isym = config.N - 1 - j;
			u32 jsym = config.N - 1 - i;

			if (!config.use_symmetry || (isym > i) || ((isym == i) && (jsym >= j))) {
				Pair_t p = {(u8) i, (u8) j};
				alphabet.push_back(p);
			}
		}
}

/**
 * Heuristic test vector reordering - attempt to speed up rejection of failing networks.
 * Core idea is to move the test vectors that most likely reject a non-sorter to the front of the list. 
 * Withing the first group of PARWORDSIZE test vectors, the individual vectors are competing for the lowest bit position in a ladder tournament.
 * Within that group, each time the vector with the lowest failing index is moving one step closer towards bit 0 by swapping it with its neighbour.
 * Vectors within the 2nd group are competing with the highest bit position i.e. the "degradation candidate" of the 1st group. Vectors in higher 
 * numbered groups (3rd group or later) are not individually rewarded, but the whole group is swapped with a group that is evaluated earlier in the ranking.
 * As the network evolves, so will the selection of "best" vectors for detecting failing mutant networks. The method described attempts to dynamically
 * optimize the order to the evolving situation. Note that to accept a sorting network, still all test vectors need to pass, no shortcuts are taken. 
 * @param bpl List of test vectors matching the prefix (regrouped for parallel execution)
 * @param failvector Index of first failing vector
 */
void bumpVectorPosition(BitParallelList_t &bpl, size_t failvector) {
	size_t groupno = failvector / PARWORDSIZE;
	size_t idx = config.N * groupno;

	if (groupno > 1) {
		size_t delta = config.N * ((groupno + 7) / 8);
		// Move up failing vector group about 1/8 the distance to the front
		for (size_t k = 0; k < config.N; k++) {
			BPWord_t z = bpl[idx + k - delta];
			bpl[idx + k - delta] = bpl[idx + k];
			bpl[idx + k] = z;
		}
	} else if (groupno == 1) {
		// Swap with last bit position of group 0
		BPWord_t m0 = 1ull << (PARWORDSIZE - 1);
		BPWord_t m1 = 1ull << (failvector % PARWORDSIZE);
		int shift = (PARWORDSIZE - 1) - (failvector % PARWORDSIZE);
		for (size_t k = 0; k < config.N; k++) {
			BPWord_t old0 = bpl[k];
			BPWord_t old1 = bpl[k + config.N];
			bpl[k] = (old0 & ~m0) | ((old1 & m1) << shift);
			bpl[k + config.N] = (old1 & ~m1) | ((old0 & m0) >> shift);
		}
	} else if (failvector > 0)// groupno==0, bit position >0
	{
		//assert(failvector<PARWORDSIZE);
		// Swap with neighbouring bit position within group 0
		BPWord_t m0 = 1ull << (failvector - 1);
		BPWord_t m1 = 1ull << failvector;
		for (size_t k = 0; k < config.N; k++) {
			BPWord_t old = bpl[k];
			bpl[k] = (old & ~m0 & ~m1) | ((old & m1) >> 1) | ((old & m0) << 1);
		}
	}
}


/**
 * Test a candidate network complementing the prefix.
 * This function is called during the regular evolution loop and attempts to
 * optimize the future order of test vectors in the background
 * @param pairs Candidated network
 * @param bpl List of test vectors matching the prefix
 * @return true if prefix+pairs form a valid sorter
 */
bool testpairsFromPrefixOutput(const Network_t &pairs, 
                               BitParallelList_t &bpl) noexcept {
	size_t idx = 0;
	size_t failvector = 0;

	while (idx < bpl.size()) {
		static BPWord_t data[NMAX];
		BPWord_t accum = 0;

		for (size_t k = 0; k < config.N; k++) {
			data[k] = bpl[idx + k];
        }

		applyBitParallelSort(data, pairs);

		for (size_t k = 0; k < (config.N - 1u); k++)
			accum |= data[k] & ~data[k + 1];// Scan for forbidden 1 -> 0 transition
		if (accum != 0ULL) {
			while ((accum & 1ull) == 0) {
				accum >>= 1;
				failvector++;
			}

			bumpVectorPosition(bpl, failvector);

			return false;
		}
		idx += config.N;
		failvector += PARWORDSIZE;
	}
	return true;
}

/// Test a candidate network complementing the prefix.
/// This function is called during the search for an initial sorter
/// \param pairs Candidated network
/// \param bpl List of test vectors matching the prefix
/// \param failed_output_pattern First unsorted output pattern detected. Used
///     to determine candidate elements to be appended.
/// \return true if prefix+pairs form a valid sorter
bool testInitialPairsFromPrefixOutput(const Network_t &pairs,
                                      const BitParallelList_t &bpl,
                                      SortWord_t &failed_output_pattern) noexcept {
	size_t idx = 0;
	failed_output_pattern = 0;

	while (idx < bpl.size()) {
		static BPWord_t data[NMAX];
		BPWord_t accum = 0;

		for (size_t k = 0; k < config.N; k++)
			data[k] = bpl[idx + k];

		applyBitParallelSort(data, pairs);

		for (size_t k = 0; k < (config.N - 1u); k++)
			accum |= data[k] & ~data[k + 1];// Scan for forbidden 1 -> 0 transition
		if (accum != 0ULL) {
			while ((accum & 1ull) == 0) {
				accum >>= 1;
				for (size_t k = 0; k < config.N; k++)
					data[k] >>= 1;
			}

			for (size_t k = 0; k < config.N; k++)
				failed_output_pattern |= (data[k] & 1) << k;

			return false;
		}
		idx += config.N;
	}
	return true;
}


/// Filter a network to obtain only the pairs that are in range 0..ninputs-1 
/// and properly sorted
/// \param nw input network
/// \param ninputs config.Number of inputs
/// \return Filtered input network
static const Network_t copyValidPairs(const Network_t &nw,
                                      const u32 ninputs) noexcept {
	static Network_t result;
	result.clear();
	for (Network_t::const_iterator it = nw.begin(); it != nw.end(); it++) {
		if ((it->hi < ninputs) && (it->lo < it->hi)) {
			result.push_back(*it);
		}
	}
	return result;
}

/// Create a prefix network using greedy algorithm A.
/// \param prefix [OUT] generated prefix
/// \param npairs config.Number of inputs to the network
constexpr 
void fillprefixGreedyA(Network_t &prefix, 
                       const u32 npairs) noexcept {
	prefix.clear();
	SortWord_t sizetmp = createGreedyPrefix(config.N, npairs, config.use_symmetry, prefix);
	if (Verbosity > 1) {
		printf("Greedy prefix size %lu, span %lu.\n", prefix.size(), (size_t) sizetmp);
	}
}

/// Create a hybrid prefix network using first the fixed prefix, then append 
///     elements with greedy algorithm A.
/// \param prefix [OUT] generated prefix
/// \param npairs config.Number of inputs to the network
static
void fillprefixFixedThenGreedyA(Network_t &prefix, 
                                const u32 npairs) noexcept {
    std::vector<Pair_t> t(sizeof(config.FixedPrefix));
    for (uint32_t i = 0; i < 1; i++) {
       t[i] = config.FixedPrefix[i];
    }
	prefix = copyValidPairs(t, config.N);
	SortWord_t sizetmp = createGreedyPrefix(config.N, npairs + prefix.size(), config.use_symmetry, prefix);
	if (Verbosity > 2) {
		printf("Hybrid prefix size %lu, span %lu.\n", prefix.size(), (size_t) sizetmp);
	}
}


/// Attempt to apply a single mutation to the network. If the mutation is a 
///     priory rejected, 0 is returned and we will try again.
/// \param newpairs [IN/OUT] candidate network
/// \return Positive integer identifying type of mutation applied, or 0 if none.
static
u32 attemptMutation(Network_t &newpairs) {
	u32 applied = 0;// config.Nothing
	u32 mtype = 1 + RANDELEM(mutationSelector);

	switch (mtype) {
		case 1:
            // Removal of random pair from list
			if (newpairs.size() > 0) {
				u32 a = RANDIDX(newpairs);
				newpairs.erase(newpairs.begin() + a);
				applied = mtype;
			}
			break;
		case 2:
            // Swap two pairs at random positions in list
			if (newpairs.size() > 1) {
				u32 a = RANDIDX(newpairs);
				u32 b = RANDIDX(newpairs);
				if (a > b) {
					u32 z = a;
					a = b;
					b = z;
				}
				if (newpairs[a] != newpairs[b]) {
					bool dependent = false;
					u8 alo = newpairs[a].lo;
					u8 ahi = newpairs[a].hi;
					u8 blo = newpairs[b].lo;
					u8 bhi = newpairs[b].hi;

					// Pairs should either intersect, or another pair should 
                    // exist between them that uses one of the same 4 inputs. 
                    // Otherwise, comparisons can be executed in parallel and
					// swapping them has no effect.
					if ((blo == alo) || (blo == ahi) || (bhi == alo) || (bhi == ahi))
						dependent = true;
					else {
						for (u32 k = a + 1; k < b; k++) {
							u8 clo = newpairs[k].lo;
							u8 chi = newpairs[k].hi;
							if ((clo == alo) || (clo == ahi) || (chi == alo) || (chi == ahi) ||
							    (clo == blo) || (clo == bhi) || (chi == blo) || (chi == bhi)) {
								dependent = true;
								break;
							}
						}
					}
					if (dependent) {
						Pair_t z = newpairs[a];
						newpairs[a] = newpairs[b];
						newpairs[b] = z;
						applied = mtype;
					}
				}
			}
			break;
		case 3:
            // Replace a pair at a random position with another random pair
			if (newpairs.size() > 0) {
				u32 a = RANDIDX(newpairs);
				Pair_t p = RANDELEM(alphabet);
				if (newpairs[a] != p) {
					newpairs[a] = p;
					applied = mtype;
				}
			}
			break;
		case 4:
            // Cross two pairs at random positions in list
			if (newpairs.size() > 1) {
				u32 a = RANDIDX(newpairs);
				u32 b = RANDIDX(newpairs);
				u8 alo = newpairs[a].lo;
				u8 ahi = newpairs[a].hi;
				u8 blo = newpairs[b].lo;
				u8 bhi = newpairs[b].hi;

				if ((alo != blo) && (alo != bhi) && (ahi != blo) && (ahi != bhi)) {
					u32 r2 = rng() % 2;
					u32 x = r2 ? bhi : blo;
					u32 y = r2 ? blo : bhi;
					newpairs[a].lo = std::min((u32)alo, x);
					newpairs[a].hi = std::max((u32)alo, x);
					newpairs[b].lo = std::min((u32)ahi, y);
					newpairs[b].hi = std::max((u32)ahi, y);
					applied = mtype;
				}
			}
			break;
		case 5:
            // Swap neighbouring intersecting pairs - special case of type r=2.
			if (newpairs.size() > 1) {
				u32 a = RANDIDX(newpairs);
				u8 alo = newpairs[a].lo;
				u8 ahi = newpairs[a].hi;
				for (u32 b = a + 1; b < newpairs.size(); b++) {
					u8 blo = newpairs[b].lo;
					u8 bhi = newpairs[b].hi;
					if ((blo == alo) || (blo == ahi) || (bhi == alo) || (bhi == ahi)) {
						if (newpairs[a] != newpairs[b]) {
							Pair_t z = newpairs[a];
							newpairs[a] = newpairs[b];
							newpairs[b] = z;
							applied = mtype;
						}
						break;
					}
				}
			}
			break;
		case 6:
            // Change one half of a pair - special case of type r=3.
			if (newpairs.size() > 0) {
				u32 a = RANDIDX(newpairs);
				Pair_t p = newpairs[a];
				Pair_t q;
				do {
					q = RANDELEM(alphabet);
				} while ((q.lo != p.lo) && (q.hi != p.lo) && 
                         (q.lo != p.hi) && (q.hi != p.hi));

				if (q != p) {
					newpairs[a] = q;
					applied = mtype;
				}
			}
			break;
		default:
			break;
	}

	return applied;
}

/// Report sorting network if it is an improved (size,depth) combination
/// \param nw Valid sorting network
static void checkImproved(const Network_t &nw) noexcept {
	u32 depth = computeDepth(nw);
	if (conv_hull.improved(nw.size(), depth)) {
		// Print only if the sorter is an improved (size,depth) combination
        // Reduce rubbish listing. Should at least compete with bubble sort before reporting
		if ((Verbosity > 1) || (nw.size() <= ((config.N * (config.N - 1u)) / 2u))) {
			printf(" {'N':%u,'L':%lu,'D':%u,'ESC':%u,'Prefix':%lu,'Postfix':%lu,'nw':",
            config.N, nw.size(), depth,  config.EscapeRate, prefix.size(), postfix.size());
			printnw(nw);
			conv_hull.print();
		}
	}
}

