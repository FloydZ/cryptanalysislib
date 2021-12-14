#define EXTERNAL_MAIN
#define BINARY_CONTAINER_ALIGNMENT 256

#include "include.h"
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}