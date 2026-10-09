#pragma once 


#include <string>

namespace cryptanalysislib {
    const int SIGMA = 26;
    struct trie {
        bool word; trie **adj;
        trie() : word(false), adj(new trie*[SIGMA]) {
            for (int i = 0; i < SIGMA; i++) adj[i] = NULL;
        }

        // NOTE: before, no node was ever freed
        ~trie() noexcept {
            for (int i = 0; i < SIGMA; i++) { delete adj[i]; }
            delete[] adj;
        }

        // NOTE: a copy would share (and then double free) the children
        trie(const trie &) = delete;
        trie &operator=(const trie &) = delete;

        /// \return the child index of `ch`, or -1 if `ch` is not in [a-z]
        static int index(const char ch) noexcept {
            return (ch >= 'a' && ch <= 'z') ? (ch - 'a') : -1;
        }

        /// \return false (and nothing is inserted) if `str` contains a
        ///     character outside of [a-z]. Before, this indexed out of bounds.
        bool addWord(const std::string &str) {
            for (char ch : str) {
                if (index(ch) < 0) { return false; }
            }

            trie *cur = this;
            for (char ch : str) {
                const int i = index(ch);
                if (!cur->adj[i]) { cur->adj[i] = new trie(); }
                cur = cur->adj[i];
            }
            cur->word = true;
            return true;
        }

        bool isWord(const std::string &str) const noexcept {
            const trie *cur = this;
            for (char ch : str) {
                const int i = index(ch);
                if ((i < 0) || !cur->adj[i]) { return false; }
                cur = cur->adj[i];
            }
            return cur->word;
        }
    };
};
