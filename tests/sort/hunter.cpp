#include "sort/sorting_network//hunter.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <ctime>

uint64_t itercount = 0;
uint64_t iter_next_report = 1;
uint64_t iter_last_report = 0;
time_t t0 = clock();
time_t t1 = t0;

/// SorterHunter main routine
int main() {
	for (u32 n = 0; n < NMUTATIONTYPES; n++) {
		for (u32 k = 0; k < config.mutation_type_weights[n]; k++)
			mutationSelector.push_back(n);
	}
	if (mutationSelector.size() == 0) {
		printf("No mutation types selected.\n");
		exit(1);
	}

    static_assert(!(config.N%2 && config.use_symmetry),"option 'Symmetric' ignored for odd number of inputs\n");

	// Initialize set of CEs to pick from //
	initalphabet();

    std::vector<Pair_t> t(sizeof(config.FixedPrefix));
    for (uint32_t i = 0; i < t.size(); i++) {
        t[i] = config.FixedPrefix[i];
    }
    std::vector<Pair_t> init(sizeof(config.InitialNetwork));
    for (uint32_t i = 0; i < t.size(); i++) {
        init[i] = config.InitialNetwork[i];
    }
	/* Create initial prefix network */
	switch (config.PrefixType) {
		case 1:// Fixed prefix
			prefix = copyValidPairs(t, config.N);
			break;
		case 2:// Greedy algorithm A
			fillprefixGreedyA(prefix, config.GreedyPrefixSize);
			break;
		case 3:// Hybrid prefix
			fillprefixFixedThenGreedyA(prefix, config.GreedyPrefixSize);
			break;
		default:// config.No prefix
			prefix.clear();
			break;
	}

	if (Verbosity > 0) {
		printf("Prefix size: %lu\n", prefix.size());
	}

	/* Prepare a set of test vectors matching the prefix */
	prepareTestVectorsFromPrefix(prefix);


	for (;;)// Outer loop - restart from here if restart is triggered (only applies if RestartRate!=0)
	{
		pairs = copyValidPairs(init, config.N);

		// Produce initial solution, simply by adding random pairs until we 
        // found a valid network. In case no postfix is present, we demand that
        // the added pair fixes at least one of the output inversions in the 
        // first detected error output vector, so it does at least some useful
        // work to help sorting the outputs. In case there is a postfix network, 
        // this check is not implemented.
		for (;;) {
			if (config.use_symmetry)
				symmetricExpansion(config.N, pairs, se);
			else
				se = pairs;

			appendNetwork(se, postfix);

			SortWord_t failed_output_pattern;

			if (testInitialPairsFromPrefixOutput(se, parallelpatterns_from_prefix, failed_output_pattern))
				break;

			Pair_t p;

			if (postfix.size() == 0)// Empty postfix: find a pattern that fixes an arbitrary inversion in the first failed output
			{
				bool found_useful_ce = false;
				do {
					p = RANDELEM(alphabet);

					if ((((failed_output_pattern >> p.lo) & 1) == 1) && (((failed_output_pattern >> p.hi) & 1) == 0))
						found_useful_ce = true;

					if (config.use_symmetry) {
						if ((((failed_output_pattern >> ((config.N - 1) - p.hi)) & 1) == 1) && (((failed_output_pattern >> ((config.N - 1) - p.lo)) & 1) == 0))
							found_useful_ce = true;
					}

				} while (!found_useful_ce);
			} else// In case of postfix: just append a random initial pair to the core network, cannot directly determine good candidate from failed output pattern.
			{
				p = RANDELEM(alphabet);
			}

			pairs.push_back(p);
		}

		Network_t totalnw;
		concatNetwork(prefix, se, totalnw);

		if (Verbosity > 1) {
			printf("Initial network size: %lu\n", totalnw.size());
		}

		checkImproved(totalnw);

		for (;;)// Program never ends, keep trying to improve, we may restart in the outer loop however.
		{
			if (Verbosity > 2) {
				itercount++;
				if (itercount >= iter_next_report) {
					clock_t t2 = clock();

					if ((t2 > t1) && (t2 > t0)) {
						double t = (t2 - t0) / (double) CLOCKS_PER_SEC;
						double dt = (t2 - t1) / (double) CLOCKS_PER_SEC;
						printf("Iteration %lu  t=%.3lf s     %.1lf it/s\n", itercount, t, (iter_next_report - iter_last_report) / dt);
					}

					t1 = t2;
					iter_last_report = iter_next_report;
					iter_next_report += (1 + iter_next_report / 10);// Report about each 10% increase of iteration count, avoid all too frequent output
				}
			}
			/* Determine number of mutations to use in this iteration */
			u32 nmods = 1;

			if (config.MaxMutations > 1) {
				// nmods += mtRand() % MaxMutations;
                nmods += rng(config.MaxMutations);
			}

			/* Create a copy of the accepted set of pairs */
			newpairs = pairs;

			/* Apply the mutations */
			u32 modcount = 0;
			while (modcount < nmods) {
				u32 r = attemptMutation(newpairs);
				if (r != 0) {
					modcount++;
				}
			}

			/// Create a symmetric expansion of the modified pairs (or just a 
            /// copy if non-symmetric network)
			if (config.use_symmetry) {
				symmetricExpansion(config.N, newpairs, se);
			} else {
				se = newpairs;
			}

			appendNetwork(se, postfix);

			/* Test whether the new postfix network yields a valid sorter when combined with the prefix */
			if ((se.size() > 0) && testpairsFromPrefixOutput(se, parallelpatterns_from_prefix)) {
				concatNetwork(prefix, se, totalnw);

				/* Accept the new postfix */
				pairs = newpairs;

				checkImproved(totalnw);
			}

			/// With low probability, add another pair random pair at a random 
            /// place. Attempt to escape from local optimum.
			// if ((config.EscapeRate > 0) && ((mtRand() % EscapeRate) == 0)) {
			if ((config.EscapeRate > 0) && (rng(config.EscapeRate) == 0)) {
				// int a = mtRand() % (pairs.size() + 1);// Random insertion position
				int a = rng(pairs.size() + 1);// Random insertion position
				Pair_t p = RANDELEM(alphabet);

				// Determine if the random pair p could be added in the last layer
				bool hit_successor = false;
				for (Network_t::const_iterator it = pairs.begin() + a; it != pairs.end(); it++) {
					if ((it->lo == p.lo) || (it->hi == p.lo) || (it->lo == p.hi) || (it->hi == p.hi)) {
						hit_successor = true;
						break;
					}
				}

				if (config.force_valid_uphill_step && hit_successor) {
					pairs.insert(pairs.begin() + a, pairs[a]);// Prepend duplicate of existing pair right in front of it => Sorter with redundant pair will remain valid
				} else {
					pairs.insert(pairs.begin() + a, p);// Add random pair at the end of the network
				}
			}

			if ((config.RestartRate > 0) && (rng(config.RestartRate) == 0)) {
				if (Verbosity > 1) {
					printf("Restart.\n");
				}
			    
                // Recompute prefix if not fixed
                switch (config.PrefixType) {
					case 1:// Fixed prefix - no update: vectors remain the same after restart
						break;
					case 2:// Greedy algorithm A
						fillprefixGreedyA(prefix, config.GreedyPrefixSize);
						prepareTestVectorsFromPrefix(prefix);
						break;
					case 3:// Hybrid prefix
						fillprefixFixedThenGreedyA(prefix, config.GreedyPrefixSize);
						prepareTestVectorsFromPrefix(prefix);
						break;
					default:// config.No prefix - no update: vectors remain the same after restart
						break;
				}

				break;// Restart using outer loop
			}
		}
	}

	return 0;
}
