#pragma once 


#include <string>

namespace cryptanalysislib {
    const int SIGMA = 26;
    struct trie {
        bool word; trie **adj;
        trie() : word(false), adj(new trie*[SIGMA]) {
            for (int i = 0; i < SIGMA; i++) adj[i] = NULL;
        }

        void addWord(const std::string &str) {
            trie *cur = this;
            for (char ch : str) {
                int i = ch - 'a';
                if (!cur->adj[i]) { cur->adj[i] = new trie(); }
                cur = cur->adj[i];
            }
            cur->word = true;
        }

        bool isWord(const std::string &str) {
            trie *cur = this;
            for (char ch : str) {
                int i = ch - 'a';
                if (!cur->adj[i]) { return false; }
                cur = cur->adj[i];
            }
            return cur->word;
        }
    };
};
