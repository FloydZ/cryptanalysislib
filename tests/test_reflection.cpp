#include <gtest/gtest.h>
#include <iostream>

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

#if defined(__clang__) && defined(__x86_64__) && !defined(__DARWIN__)
#include "reflection/reflection.h"


struct Person {
	std::string first_name;
	std::string last_name = "Simpson";
	std::string town = "Springfield";
	unsigned int age;
	std::vector<Person> children;
};

TEST(simple, my_type) {
	for (const auto& f : cryptanalysislib::reflection::fields<Person>()) {
		std::cout << "name: " << f.name() << ", type: " << f.type() << std::endl;
	}
}

TEST(reflection, replace) {
	const auto lisa = Person{.first_name = "Lisa", .last_name = "Simpson"};

	// Returns a deep copy of "lisa" with the first_name replaced.
	const auto maggie = cryptanalysislib::reflection::replace(
	        lisa, cryptanalysislib::reflection::internal::make_field<"first_name">(std::string("Maggie")));
	std::cout << maggie.first_name << " " << maggie.last_name << std::endl;
}

TEST(reflection, to_view) {
	auto lisa = Person{.first_name = "Lisa", .last_name = "Simpson", .age = 8};

	const auto view = cryptanalysislib::reflection::to_view(lisa);

	// view.values() is a std::tuple containing
	// pointers to the original fields.
	// This will modify the struct `lisa`:
	*std::get<0>(view.values()) = "Maggie";

	// All of this is supported as well:
	// *view.get<1>() = "Simpson";
	// *view.get<"age">() = 0;
	*cryptanalysislib::reflection::get<0>(view) = "Maggie";
	*cryptanalysislib::reflection::get<"first_name">(view) = "Maggie";
}

struct A {
	std::string f1;
	std::string f2;
};

struct B {
	std::string f3;
	std::string f4;
};

struct C {
	std::string f1;
	std::string f2;
	std::string f4;
};

TEST(reflection, as) {
	//constexpr static auto a = A{.f1 = "Hello", .f2 = "World"};
	//constexpr static auto b = B{.f3 = "Hello", .f4 = "World"};

	// f1 and f2 are taken from a f4 is taken from b, f3 is ignored.
	// TODO darwin const auto c = cryptanalysislib::reflection::as<C>(a, b);
	// TODO darwin std::cout << c.f1 << " " << c.f2 << " " << c.f4 << std::endl;
}

TEST(reflection, replace2) {
	const auto a = A{.f1 = "Hello", .f2 = "World"};
	const auto c = C{.f1 = "C++", .f2 = "is", .f4 = "great"};

	// The fields f1 and f2 are replaced with the fields f1 and f2 in a.
	const auto c2 = cryptanalysislib::reflection::replace(c, a);
	std::cout << c2.f1 << " " << c2.f2 << " " << c2.f4 << std::endl;
}

TEST(reflection, replace3) {
	//constexpr static auto a = A{.f1 = "Hello", .f2 = "World"};
	//constexpr static auto c = C{.f1 = "C++", .f2 = "is", .f4 = "great"};

	// The fields f1 and f2 are replaced with the fields f1 and f2 in a.
	// TODO darwin const auto c2 = cryptanalysislib::reflection::replace(c, a);
	// TODO darwin std::cout << c2.f1 << " " << c2.f2 << " " << c2.f4 << std::endl;
}
#endif
int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
