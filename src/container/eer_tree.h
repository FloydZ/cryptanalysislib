#ifndef CRYPTANALYSISLIB_CONTAINER_EER_TREE_H
#define CRYPTANALYSISLIB_CONTAINER_EER_TREE_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cryptanalysislib {
	/// Eertree (palindromic tree, Rubinchik & Shur 2015): the distinct
	/// palindromic substrings of a string, built online in O(n log sigma).
	/// Every node is a palindrome, its edge `c` leads to the palindrome `c P c`,
	/// its suffix link to its longest proper palindromic suffix.
	/// Node 0 is the imaginary root of length -1 (children: odd palindromes),
	/// node 1 the empty palindrome (children: even palindromes).
	/// Usage:
	/// \code
	/// 	eer_tree<char> t;
	/// 	for (const char c : "abacaba") { t.push_back(c); }
	/// 	t.distinct_palindromes(); // 7: a, b, c, aba, aca, bacab, abacaba
	/// \endcode
	/// \tparam T symbol type (compared with ==)
	template<typename T = char>
	class eer_tree {
	public:
		/// a palindrome
		struct node {
			/// length, -1 for the imaginary root
			int64_t len;
			/// longest proper palindromic suffix
			uint32_t link;
			/// number of palindromic suffixes (including itself)
			uint32_t depth;
			/// first outgoing edge, `npos` if none
			uint32_t edges;
		};

		constexpr static uint32_t npos = uint32_t(-1);

	private:
		/// outgoing edge `c` of a node, the edges of a node are a list
		struct edge {
			T c;
			uint32_t to;
			uint32_t next;
		};

		std::vector<T> s;
		std::vector<node> nodes;
		std::vector<edge> edges;
		/// longest palindromic suffix of `s`
		uint32_t last = 1;
		/// number of palindromic substrings of `s`, with multiplicity
		uint64_t total = 0;

		/// \return the child `c` of node `v`, `npos` if none
		[[nodiscard]] uint32_t child(const uint32_t v, const T &c) const noexcept {
			for (uint32_t e = nodes[v].edges; e != npos; e = edges[e].next) {
				if (edges[e].c == c) {
					return edges[e].to;
				}
			}
			return npos;
		}

		/// \return the longest palindromic suffix `P` of node `v` (following
		/// 	the suffix links), s.t. `c P c` is a suffix of `s`, where `c` is
		/// 	the last symbol
		[[nodiscard]] uint32_t extendable(uint32_t v) const noexcept {
			const size_t i = s.size() - 1;
			while (true) {
				const int64_t j = int64_t(i) - nodes[v].len - 1;
				if ((j >= 0) && (s[size_t(j)] == s[i])) {
					return v;
				}
				v = nodes[v].link;
			}
		}

	public:
		eer_tree() noexcept {
			clear();
		}

		/// removes all symbols
		void clear() noexcept {
			s.clear();
			nodes.clear();
			edges.clear();
			nodes.push_back(node{-1, 0, 0, npos});
			nodes.push_back(node{0, 0, 0, npos});
			last = 1;
			total = 0;
		}

		/// \param n[in]: expected number of symbols
		void reserve(const size_t n) {
			s.reserve(n);
			nodes.reserve(n + 2);
			edges.reserve(n);
		}

		/// appends `c` to the string
		/// \return true if a new distinct palindrome (a suffix of the new
		/// 	string) was found
		bool push_back(const T &c) {
			s.push_back(c);
			const uint32_t p = extendable(last);
			const uint32_t q = child(p, c);
			if (q != npos) {
				last = q;
				total += nodes[q].depth;
				return false;
			}

			// the new palindrome `c P c`; its suffix link is the longest
			// extendable proper suffix (or the empty palindrome for len 1)
			const uint32_t v = uint32_t(nodes.size());
			const int64_t len = nodes[p].len + 2;
			uint32_t link = 1;
			if (len > 1) {
				link = child(extendable(nodes[p].link), c);
			}
			nodes.push_back(node{len, link, nodes[link].depth + 1, npos});
			edges.push_back(edge{c, v, nodes[p].edges});
			nodes[p].edges = uint32_t(edges.size() - 1);

			last = v;
			total += nodes[v].depth;
			return true;
		}

		/// \return number of symbols
		[[nodiscard]] size_t size() const noexcept {
			return s.size();
		}

		/// \return number of distinct (non empty) palindromic substrings
		[[nodiscard]] size_t distinct_palindromes() const noexcept {
			return nodes.size() - 2;
		}

		/// \return number of palindromic substrings, counted with multiplicity
		[[nodiscard]] uint64_t total_palindromes() const noexcept {
			return total;
		}

		/// \return length of the longest palindromic suffix
		[[nodiscard]] size_t longest_palindromic_suffix() const noexcept {
			return size_t(nodes[last].len);
		}

		/// \return node `i`, 0 and 1 are the roots
		[[nodiscard]] const node &operator[](const size_t i) const noexcept {
			return nodes[i];
		}

		/// \return number of nodes (including the two roots)
		[[nodiscard]] size_t nr_nodes() const noexcept {
			return nodes.size();
		}
	};
} // end namespace cryptanalysislib
#endif
