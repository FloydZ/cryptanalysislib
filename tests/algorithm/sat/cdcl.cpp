#include <gtest/gtest.h>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "algorithm/sat/cdcl.h"

using ::testing::InitGoogleTest;

/// the .cnf files next to this test
static std::string cnf_path(const char *name) {
	std::string f = __FILE__;
	return f.substr(0, f.rfind('/') + 1) + name;
}

/// independent reader of the DIMACS clauses
static std::vector<std::vector<int>> read_cnf(const std::string &path, int &nvars) {
	std::ifstream in(path);
	std::string line;
	std::vector<std::vector<int>> clauses;
	std::vector<int> cur;
	while (std::getline(in, line)) {
		if (line.empty() || line[0] == 'c') { continue; }
		std::istringstream ss(line);
		if (line[0] == 'p') { std::string p, cnf; int m; ss >> p >> cnf >> nvars >> m; continue; }
		int lit;
		while (ss >> lit) {
			if (lit == 0) { clauses.push_back(cur); cur.clear(); } else { cur.push_back(lit); }
		}
	}
	return clauses;
}

/// solve with the CDCL solver and check the result: a SAT model must satisfy
/// every clause, UNSAT is checked by enumerating all assignments
static void check(const char *name) {
	int nvars = 0;
	const auto clauses = read_cnf(cnf_path(name), nvars);
	ASSERT_LE(nvars, 20);

	struct solver S{};
	const bool sat = (parse(&S, cnf_path(name).c_str()) == SAT) && (solve(&S) == SAT);
	ASSERT_NE(S.DB, nullptr);

	if (sat) {
		for (const auto &c : clauses) {
			bool ok = false;
			for (const int l : c) { ok |= (S._false[-l] != 0); } // literal l true <=> -l false
			EXPECT_TRUE(ok) << name;
		}
	} else {
		for (uint32_t x = 0; x < (1u << nvars); ++x) {
			bool all = true;
			for (const auto &c : clauses) {
				bool ok = false;
				for (const int l : c) { const bool v = (x >> (std::abs(l) - 1)) & 1u; ok |= (l > 0) ? v : !v; }
				all &= ok;
			}
			EXPECT_FALSE(all) << name << " claimed UNSAT, but " << x << " satisfies it";
		}
	}
}

TEST(cdcl, simple)   { check("simple.cnf"); }
TEST(cdcl, problem1) { check("problem1.cnf"); }

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
