TEST(A, copy) {
	T *a1 = (T *) calloc(size, sizeof(T));
	T *a2 = (T *) malloc(size * sizeof(T));
	for (size_t i = 0; i < size; ++i) {
		a2[i] = i;
	}

	cryptanalysislib::memcpy(a2, a1, size);
	for (size_t i = 0; i < size; ++i) {
		EXPECT_EQ(a2[i], 0);
	}

	free(a1);
	free(a2);
}

TEST(A, set) {
	T *a1 = (T *) calloc(size, sizeof(T));
	T a = 1;

	cryptanalysislib::memset(a1, a, size);
	for (size_t i = 0; i < size; ++i) {
		EXPECT_EQ(a1[i], a);
	}

	free(a1);
}
