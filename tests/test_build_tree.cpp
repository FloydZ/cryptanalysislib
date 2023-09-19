#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "helper.h"
#include "kAry_type.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

#ifdef USE_FPLLL

TEST(TreeTest, BuildTreeTest1) {
    //FixME: Testing mit Identity does only work for q<1<<8 as Value type is uint8_t, also filtering makes problems for higher q
    //maybe implement a switch to turn of filtering for debugging purposes?
    const uint64_t d=1;
    unsigned int basesize=10;
	fplll::ZZ_mat<kAryType> A_(n, n);
	//A_.gen_uniform(4);
	//A_.fill(0);
	A_.gen_identity(n);
	Matrix_T<fplll::ZZ_mat<kAryType>> A{A_};

//    t[0].set_load(0);
//    t[1].set_load(0);
//    t[0].generate_base(1u << 5u, A);
//    t[1].generate_base(1u << 5u, A);
//    t[0].sort_level(0);
//    t[1].sort_level(0);
//    t.join_stream(0);
    __level_translation_array[1]=5;
    __level_translation_array[2]=10;
    __level_translation_array[3]=15;
    __level_translation_array[4]=20;

	Tree t{d, A, basesize, __level_translation_array, __level_filter_array};
	uint64_t k_lower, k_higher;

	Label target {};
    target.zero();
    target.random();
    //std::cout<<"target is "<<target<<"\n";
    t.build_tree(target);
	//std::cout << "bla" << t[0] << "\n";

    //std::cout<<"\nResultsize: "<<t[d+1].get_load();
    auto right=true;
    int wrong=0;
    for(uint64_t i = 0;i<t[d+1].get_load();++i) {
        t[d+1][i].recalculate_label(A);
        //std::cout<<"\n"<<t[d+1][i];
        for(int j =0;j<d;++j) {
	        translate_level(&k_lower, &k_higher, j, __level_translation_array);

	        if (!(Label::cmp(t[d + 1][i].get_label(), target, k_lower, k_higher))) {
		        right = false;
		        // std::cout << "\n" << t[d + 1][i].get_label();
		        wrong++;
	        }
        }
    }

    EXPECT_EQ(0, wrong);
    EXPECT_EQ(right, true);
    EXPECT_GT(t[d+1].get_load(),1u<<(basesize-1));
    EXPECT_LT(t[d+1].get_load(),1u<<(basesize+1));
}

TEST(TreeTest, RestoreBaselists) {
    const uint64_t d=4;
    unsigned int basesize=10;
	fplll::ZZ_mat<kAryType> A_(n, n);
	//A.gen_uniform(4);
	//A.fill(0);
	A_.gen_identity(n);
	const Matrix_T<fplll::ZZ_mat<kAryType>> A{A_};

    __level_translation_array[1]=5;
    __level_translation_array[2]=10;
    __level_translation_array[3]=15;
    __level_translation_array[4]=20;

	Tree t{d, A, basesize, __level_translation_array, __level_filter_array};
	Tree t2{d, A, basesize, __level_translation_array, __level_filter_array};

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u<<basesize, A);
    t[1].generate_base_random(1u<<basesize, A);
    t[0].sort_level(-1, __level_translation_array);
    t[1].sort_level(-1, __level_translation_array);
    t2[0].generate_base_random(1u<<basesize, A);
    t2[1].generate_base_random(1u<<basesize, A);
    Label target {};
    target.random();
    for(uint64_t i=0;i<t[0].get_load();++i) {
        t2[0][i] = t[0][i];
        t2[1][i] = t[1][i];
    }
    t.build_tree(target, false);
    uint64_t wrong1=0;
    uint64_t wrong2=0;
    bool correct=true;
    t[0].sort_level(-1, __level_translation_array);
    t[1].sort_level(-1, __level_translation_array);
    for(uint64_t i=0;i<t[0].get_load();++i) {
        if (!(t2[0][i].is_equal(t[0][i]))) {
            correct = false;
            wrong1++;
        }
        if(!(t2[1][i].is_equal(t[1][i]))){
            correct = false;
            wrong2++;
        }
    }

    EXPECT_EQ(correct,true);
    EXPECT_EQ(wrong1,0);
    EXPECT_EQ(wrong2,0);
}

TEST(TreeTest, RestoreLabelSingleList) {
    const uint64_t d=4;
    unsigned int basesize=10;

	fplll::ZZ_mat<kAryType> A_(n, n);
	//A_.gen_uniform(4);
	//A_.fill(0);
	A_.gen_identity(n);
	const Matrix_T<fplll::ZZ_mat<kAryType>> A{A_};

    __level_translation_array[1]=5;
    __level_translation_array[2]=10;
    __level_translation_array[3]=15;
    __level_translation_array[4]=20;

	Tree t{d, A, basesize, __level_translation_array, __level_filter_array};

	t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u<<basesize, A);
    t[1].generate_base_random(1u<<basesize, A);

    Label target {};
    target.random();
    Element tmp {};
    t.build_tree(target, false);

    uint64_t wrong=0;
    bool correct=true;
    for(uint64_t i=0;i<t[d+1].get_load();++i) {
        tmp.get_label()=t[d+1][i].get_label();
        t[d+1][i].recalculate_label(A_);

        if (!(t[d+1][i].is_equal(tmp))) {
            correct = false;
            wrong++;
        }

    }

    EXPECT_EQ(correct,true);
    EXPECT_EQ(wrong,0);
}

TEST(TreeTest, RestoreLabelTwoLists) {
	// @ Andre diese test gibt halt irgendwie keinen Sinn. Da ja der letzt join auf level 0, das heißt die koordinaten 15-20 nicht gematch werden.
	// Es geht unten weiter
    const uint64_t d=4;
    unsigned int basesize=10;
	fplll::ZZ_mat<kAryType> A_(n, n);
	//A_.gen_uniform(4);
	//A_.fill(0);
	A_.gen_identity(n);
	const Matrix_T<fplll::ZZ_mat<kAryType>> A{A_};

    __level_translation_array[1]=5;
    __level_translation_array[2]=10;
    __level_translation_array[3]=15;
    __level_translation_array[4]=20;

	Tree t{d, A, basesize, __level_translation_array, __level_filter_array};

	t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u<<basesize, A);
    t[1].generate_base_random(1u<<basesize, A);

	Label target{};
    target.random();
    Element tmp {};
    t.build_tree(target, false,true);

    uint64_t wrong=0;
    bool correct=true;
    for (int a=0;a<2;++a)
        for(uint64_t i=0;i<t[d+a].get_load();++i) {
            tmp.get_label() = t[d+a][i].get_label();
	        t[d+a][i].recalculate_label(A_);

	        // Da nicht auf den letzten 5 Koordinaten nicht gematched wird ist die Zeile:
	        //      if (!(t[d+a][i].is_equal(tmp, -1))) {
	        // nicht richtig, da ja noch nciht auf der vollen länge gematched wurde. Das witzige is a=1 wurde schon komplett gematched.
	        if (!(t[d+a][i].is_equal(tmp, 0, 15))) {
            	// std::cout << tmp;
	            // std::cout << t[d+a][i];
	            // std::cout << "\n\n";

	            correct = false;
                wrong++;
            }
        }

    EXPECT_EQ(correct,true);
    EXPECT_EQ(wrong,0);
}

#endif

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
    return RUN_ALL_TESTS();
}
