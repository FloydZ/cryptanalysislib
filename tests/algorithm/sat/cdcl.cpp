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

/// the same with the `CDCL` class
static void check_class(const char *name) {
	using namespace cryptanalysislib::sat;
	int nvars = 0;
	const auto clauses = read_cnf(cnf_path(name), nvars);
	ASSERT_LE(nvars, 20);

	CDCL<> s(nvars, clauses.size());
	bool ok = true;
	for (const auto &c : clauses) {
		std::vector<Literal> lits;
		for (const int l : c) { lits.emplace_back(l); }
		ok &= s.add_clause(lits);
	}
	const bool sat = ok && (s.solve() == SAT);

	if (sat) {
		for (const auto &c : clauses) {
			bool r = false;
			for (const int l : c) { r |= s.value(Literal(l)); }
			EXPECT_TRUE(r) << name;
		}
	} else {
		for (uint32_t x = 0; x < (1u << nvars); ++x) {
			bool all = true;
			for (const auto &c : clauses) {
				bool r = false;
				for (const int l : c) { const bool v = (x >> (std::abs(l) - 1)) & 1u; r |= (l > 0) ? v : !v; }
				all &= r;
			}
			EXPECT_FALSE(all) << name << " claimed UNSAT, but " << x << " satisfies it";
		}
	}
}

TEST(cdcl, class_simple)   { check_class("simple.cnf"); }
TEST(cdcl, class_problem1) { check_class("problem1.cnf"); }

/// pigeon hole: 3 pigeons, 2 holes is UNSAT, 2 pigeons, 2 holes is SAT
TEST(cdcl, class_pigeonhole) {
	using namespace cryptanalysislib::sat;
	for (const uint32_t p : {2u, 3u}) {
		constexpr uint32_t h = 2;
		// variable of pigeon i in hole j
		const auto var = [](const uint32_t i, const uint32_t j) { return 1 + i * h + j; };
		CDCL<> s(p * h, 0);
		for (uint32_t i = 0; i < p; ++i) {
			s.add_clause({Literal::pos(var(i, 0)), Literal::pos(var(i, 1))});
		}
		for (uint32_t j = 0; j < h; ++j) {
			for (uint32_t i = 0; i < p; ++i) {
				for (uint32_t k = i + 1; k < p; ++k) {
					s.add_clause({Literal::neg(var(i, j)), Literal::neg(var(k, j))});
				}
			}
		}
		EXPECT_EQ(s.solve(), (p <= h) ? SAT : UNSAT) << p;
	}

	const Literal l = Literal::neg(5);
	EXPECT_EQ(l.var(), 5u);
	EXPECT_TRUE(l.negated());
	EXPECT_EQ((-l).dimacs(), 5);
	EXPECT_EQ(-l, Literal::pos(5));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
