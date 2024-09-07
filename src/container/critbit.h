#ifndef CRYPTANALYSISLIV_CONTAINER_CRITBIT_H
#define CRYPTANALYSISLIV_CONTAINER_CRITBIT_H

#include <cstdint>
#include <cstdlib>

/// Source
/// https://github.com/glk/critbit 

struct critbit_node {
	critbit_node *child[2];
	uint32_t byte;
	uint8_t otherbits;
};

struct critbit_tree {
	void *root;
	size_t keylen;
	void *free_arg;
};
#endif
