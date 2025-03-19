#include "access_profiler.h"
#include <stdio.h>

struct test : cryptanalysislib::instrument_type<test> {
	test() : a(0), b(0) {}
	char array[50];
	int a;
	long long b;
};

int main(int argc, char* argv[]) {
	test* t1 = new test;

	for (int i = 0; i < 10; ++i)
	{
		++t1->a;
		t1->b += t1->a;
	}

	printf("%d\n", t1->b);

	delete t1;
}
