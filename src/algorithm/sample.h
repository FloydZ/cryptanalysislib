#pragma once

#include <iterator>
#include <random>
#include <algorithm>
#include <cmath>
#include "thread/thread.h"

namespace cryptanalysislib {
    /// Selects n elements from the sequence [first, last) using the given URBG generator
    /// such that each possible sample has equal probability of appearance.
    /// The algorithm used is the Reservoir sampling algorithm.
    ///
    /// \tparam PopulationIt Input iterator type for the population range
    /// \tparam SampleIt Output iterator type for the sample range
    /// \tparam Distance Integer type for the sample size
    /// \tparam URBG Uniform random bit generator type
    /// \param first Iterator to the first element in the population range
    /// \param last Iterator one past the last element in the population range
    /// \param out Iterator to the beginning of the destination range
    /// \param n Number of elements to sample
    /// \param g Uniform random bit generator used as the source of randomness
    /// \return Iterator to the element after the last sample written
    template<class PopulationIt,
             class SampleIt,
             class Distance,
             class URBG>
#if __cplusplus > 201709L
    requires std::input_iterator<PopulationIt> &&
             std::output_iterator<SampleIt, typename std::iterator_traits<PopulationIt>::value_type>
#endif
    SampleIt sample(PopulationIt first,
                    PopulationIt last,
                    SampleIt out,
                    Distance n,
                    URBG&& g) {
        using diff_t = typename std::iterator_traits<PopulationIt>::difference_type;
        using value_type = typename std::iterator_traits<PopulationIt>::value_type;
        using dist_t = std::uniform_int_distribution<diff_t>;
        using param_t = typename dist_t::param_type;
        
        diff_t population_size = std::distance(first, last);
        n = std::min(static_cast<diff_t>(n), population_size);
        
        if (n <= 0) {
            return out;
        }
        
        // For input iterators, use reservoir sampling algorithm
        if constexpr (std::is_same_v<typename std::iterator_traits<PopulationIt>::iterator_category, 
                                    std::input_iterator_tag>) {
            // Fill the reservoir with the first n elements
            for (diff_t i = 0; i < n && first != last; ++i, ++first) {
                *out++ = *first;
            }
            
            // If we exhausted the input before filling the reservoir, return
            if (first == last) {
                return out;
            }
            
            // Continue with reservoir sampling
            dist_t dist;
            diff_t processed = n;
            
            while (first != last) {
                ++processed;
                const diff_t k = dist(g, param_t(0, processed - 1));
                if (k < n) {
                    // Replace the k-th element with the current one
                    auto replace_it = out - n;
                    std::advance(replace_it, k);
                    *replace_it = *first;
                }
                ++first;
            }
            
            return out;
        } else if constexpr (std::is_same_v<typename std::iterator_traits<PopulationIt>::iterator_category, 
                                          std::random_access_iterator_tag>) {
            // For random access iterators, use selection sampling algorithm
            
            // If the sample size is a large fraction of the population,
            // use Fisher-Yates shuffle and take the first n elements
            if (n > population_size / 4) {
                // Make a copy of the input sequence that we can shuffle
                std::vector<value_type> population(first, last);
                
                // Shuffle the first n elements
                for (diff_t i = 0; i < n; ++i) {
                    dist_t dist;
                    const diff_t j = dist(g, param_t(i, population_size - 1));
                    if (i != j) {
                        std::swap(population[i], population[j]);
                    }
                }
                
                // Copy the first n elements to the output
                return std::copy_n(population.begin(), n, out);
            } else {
                // For small samples relative to population, track selected indices
                std::vector<bool> selected(population_size, false);
                dist_t dist;
                
                // Randomly select n distinct elements
                for (diff_t i = 0; i < n; ++i) {
                    diff_t index;
                    do {
                        index = dist(g, param_t(0, population_size - 1));
                    } while (selected[index]);
                    
                    selected[index] = true;
                    *out++ = *(first + index);
                }
                
                return out;
            }
        } else {
            // For forward and bidirectional iterators, 
            // use the reservoir sampling approach
            
            // Fill the reservoir with the first n elements
            diff_t i = 0;
            for (; i < n && first != last; ++i, ++first) {
                *out++ = *first;
            }
            
            // If we exhausted the input before filling the reservoir, return
            if (first == last) {
                return out - i;
            }
            
            // Continue with reservoir sampling
            dist_t dist;
            diff_t processed = n;
            
            while (first != last) {
                ++processed;
                const diff_t k = dist(g, param_t(0, processed - 1));
                if (k < n) {
                    // Replace the k-th element with the current one
                    auto replace_it = out - n;
                    std::advance(replace_it, k);
                    *replace_it = *first;
                }
                ++first;
            }
            
            return out;
        }
    }
    
    /// Selects n elements from the sequence [first, last) using the given URBG generator
    /// with parallel execution if possible.
    ///
    /// \tparam ExecPolicy Execution policy type
    /// \tparam PopulationIt Input iterator type for the population range
    /// \tparam SampleIt Output iterator type for the sample range
    /// \tparam Distance Integer type for the sample size
    /// \tparam URBG Uniform random bit generator type
    /// \param policy Execution policy
    /// \param first Iterator to the first element in the population range
    /// \param last Iterator one past the last element in the population range
    /// \param out Iterator to the beginning of the destination range
    /// \param n Number of elements to sample
    /// \param g Uniform random bit generator used as the source of randomness
    /// \return Iterator to the element after the last sample written
    template<class ExecPolicy,
             class PopulationIt,
             class SampleIt,
             class Distance,
             class URBG>
#if __cplusplus > 201709L
    requires std::random_access_iterator<PopulationIt> &&
             std::output_iterator<SampleIt, typename std::iterator_traits<PopulationIt>::value_type>
#endif
    SampleIt sample(ExecPolicy&& policy,
                    PopulationIt first,
                    PopulationIt last,
                    SampleIt out,
                    Distance n,
                    URBG&& g) {
        if (is_seq<ExecPolicy>(policy)) {
            return sample(first, last, out, n, std::forward<URBG>(g));
        }
        
        using diff_t = typename std::iterator_traits<PopulationIt>::difference_type;
        using value_type = typename std::iterator_traits<PopulationIt>::value_type;
        
        diff_t population_size = std::distance(first, last);
        n = std::min(static_cast<diff_t>(n), population_size);
        
        if (n <= 0) {
            return out;
        }
        
        // For parallel execution, use Fisher-Yates approach
        // Make a copy of the input sequence
        std::vector<value_type> population(first, last);
        
        // Create a vector of indices
        std::vector<diff_t> indices(population_size);
        for (diff_t i = 0; i < population_size; ++i) {
            indices[i] = i;
        }
        
        // Shuffle the first n indices
        for (diff_t i = 0; i < n; ++i) {
            std::uniform_int_distribution<diff_t> dist(i, population_size - 1);
            const diff_t j = dist(g);
            if (i != j) {
                std::swap(indices[i], indices[j]);
            }
        }
        
        // Use only the first n indices to build the sample
        for (diff_t i = 0; i < n; ++i) {
            *out++ = population[indices[i]];
        }
        
        return out;
    }
}; // end namespace cryptanalysislib
