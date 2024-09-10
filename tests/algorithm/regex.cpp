#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/regex.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

#define OK    ((char*) 1)
#define NOK   ((char*) 0)

char* test_vector[][4] = {
    { OK,  (char *)"\\d",                       (char *)"5",                (char*) 1},
    { OK,  (char *)"\\w+",                      (char *)"hej",              (char*) 3},
    { OK,  (char *)"\\s",                       (char *)"\t \n",            (char*) 1},
    { NOK, (char *)"\\S",                       (char *)"\t \n",            (char*) 0},
    { OK,  (char *)"[\\s]",                     (char *)"\t \n",            (char*) 1},
    { NOK, (char *)"[\\S]",                     (char *)"\t \n",            (char*) 0},
    { NOK, (char *)"\\D",                       (char *)"5",                (char*) 0},
    { NOK, (char *)"\\W+",                      (char *)"hej",              (char*) 0},
    { OK,  (char *)"[0-9]+",                    (char *)"12345",            (char*) 5},
    { OK,  (char *)"\\D",                       (char *)"hej",              (char*) 1},
    { NOK, (char *)"\\d",                       (char *)"hej",              (char*) 0},
    { OK,  (char *)"[^\\w]",                    (char *)"\\",               (char*) 1},
    { OK,  (char *)"[\\W]",                     (char *)"\\",               (char*) 1},
    { NOK, (char *)"[\\w]",                     (char *)"\\",               (char*) 0},
    { OK,  (char *)"[^\\d]",                    (char *)"d",                (char*) 1},
    { NOK, (char *)"[\\d]",                     (char *)"d",                (char*) 0},
    { NOK, (char *)"[^\\D]",                    (char *)"d",                (char*) 0},
    { OK,  (char *)"[\\D]",                     (char *)"d",                (char*) 1},
    { OK,  (char *)"^.*\\\\.*$",                (char *)"c:\\Tools",        (char*) 8},
    { OK,  (char *)"^[\\+-]*[\\d]+$",           (char *)"+27",              (char*) 3},
    { OK,  (char *)"[abc]",                     (char *)"1c2",              (char*) 1},
    { NOK, (char *)"[abc]",                     (char *)"1C2",              (char*) 0},
    { OK,  (char *)"[1-5]+",                    (char *)"0123456789",       (char*) 5},
    { OK,  (char *)"[.2]",                      (char *)"1C2",              (char*) 1},
    { OK,  (char *)"a*$",                       (char *)"Xaa",              (char*) 2},
    { OK,  (char *)"a*$",                       (char *)"Xaa",              (char*) 2},
    { OK,  (char *)"[a-h]+",                    (char *)"abcdefghxxx",      (char*) 8},
    { NOK, (char *)"[a-h]+",                    (char *)"ABCDEFGH",         (char*) 0},
    { OK,  (char *)"[A-H]+",                    (char *)"ABCDEFGH",         (char*) 8},
    { NOK, (char *)"[A-H]+",                    (char *)"abcdefgh",         (char*) 0},
    { OK,  (char *)"[^\\s]+",                   (char *)"abc def",          (char*) 3},
    { OK,  (char *)"[^fc]+",                    (char *)"abc def",          (char*) 2},
    { OK,  (char *)"[^d\\sf]+",                 (char *)"abc def",          (char*) 3},
    { OK,  (char *)"\n",                        (char *)"abc\ndef",         (char*) 1},
    { OK,  (char *)"b.\\s*\n",                  (char *)"aa\r\nbb\r\ncc\r\n\r\n",(char*) 4      },
    { OK,  (char *)".*c",                       (char *)"abcabc",           (char*) 6      },
    { OK,  (char *)".+c",                       (char *)"abcabc",           (char*) 6      },
    { OK,  (char *)"[b-z].*",                   (char *)"ab",               (char*) 1      },
    { OK,  (char *)"b[k-z]*",                   (char *)"ab",               (char*) 1      },
    { NOK, (char *)"[0-9]",                     (char *)"  - ",             (char*) 0      },
    { OK,  (char *)"[^0-9]",                    (char *)"  - ",             (char*) 1      },
    { NOK, (char *)"\\d\\d:\\d\\d:\\d\\d",      (char *)"0s:00:00",         (char*) 0      },
    { NOK, (char *)"\\d\\d:\\d\\d:\\d\\d",      (char *)"000:00",           (char*) 0      },
    { NOK, (char *)"\\d\\d:\\d\\d:\\d\\d",      (char *)"00:0000",          (char*) 0      },
    { NOK, (char *)"\\d\\d:\\d\\d:\\d\\d",      (char *)"100:0:00",         (char*) 0      },
    { NOK, (char *)"\\d\\d:\\d\\d:\\d\\d",      (char *)"00:100:00",        (char*) 0      },
    { NOK, (char *)"\\d\\d:\\d\\d:\\d\\d",      (char *)"0:00:100",         (char*) 0      },
    { OK,  (char *)"\\d\\d?:\\d\\d?:\\d\\d?",   (char *)"0:0:0",            (char*) 5      },
    { OK,  (char *)"\\d\\d?:\\d\\d?:\\d\\d?",   (char *)"0:00:0",           (char*) 6      },
    { OK,  (char *)"\\d\\d?:\\d\\d?:\\d\\d?",   (char *)"0:0:00",           (char*) 5      },
    { OK,  (char *)"\\d\\d?:\\d\\d?:\\d\\d?",   (char *)"00:0:0",           (char*) 6      },
    { OK,  (char *)"\\d\\d?:\\d\\d?:\\d\\d?",   (char *)"00:00:0",          (char*) 7      },
    { OK,  (char *)"\\d\\d?:\\d\\d?:\\d\\d?",   (char *)"00:0:00",          (char*) 6      },
    { OK,  (char *)"\\d\\d?:\\d\\d?:\\d\\d?",   (char *)"0:00:00",          (char*) 6      },
    { OK,  (char *)"\\d\\d?:\\d\\d?:\\d\\d?",   (char *)"00:00:00",         (char*) 7      },
    { OK,  (char *)"[Hh]ello [Ww]orld\\s*[!]?", (char *)"Hello world !",    (char*) 12     },
    { OK,  (char *)"[Hh]ello [Ww]orld\\s*[!]?", (char *)"hello world !",    (char*) 12     },
    { OK,  (char *)"[Hh]ello [Ww]orld\\s*[!]?", (char *)"Hello World !",    (char*) 12     },
    { OK,  (char *)"[Hh]ello [Ww]orld\\s*[!]?", (char *)"Hello world!   ",  (char*) 11     },
    { OK,  (char *)"[Hh]ello [Ww]orld\\s*[!]?", (char *)"Hello world  !",   (char*) 13     },
    { OK,  (char *)"[Hh]ello [Ww]orld\\s*[!]?", (char *)"hello World    !", (char*) 15     },
    { NOK, (char *)"\\d\\d?:\\d\\d?:\\d\\d?",   (char *)"a:0",              (char*) 0      },
    { OK,  (char *)"[^\\w][^-1-4]",     		(char *)")T",          		(char*) 2      },
    { OK,  (char *)"[^\\w][^-1-4]",     		(char *)")^",          		(char*) 2      },
    { OK,  (char *)"[^\\w][^-1-4]",     		(char *)"*)",          		(char*) 2      },
    { OK,  (char *)"[^\\w][^-1-4]",     		(char *)"!.",          		(char*) 2      },
    { OK,  (char *)"[^\\w][^-1-4]",     		(char *)" x",          		(char*) 2      },
    { OK,  (char *)"[^\\w][^-1-4]",     		(char *)"$b",          		(char*) 2      },
    { OK,  (char *)".?bar",                     (char *)"real_bar",        (char*) 4      },
    { NOK, (char *)".?bar",                     (char *)"real_foo",        (char*) 0      },
    { NOK, (char *)"X?Y",                       (char *)"Z",               (char*) 0      },
    { OK,  (char *)"[a-z]+\nbreak",              (char *)"blahblah\nbreak",  (char*) 14     },
    { OK,  (char *)"[a-z\\s]+\nbreak",           (char *)"bla bla \nbreak",  (char*) 14     },
};


TEST(regex, simple) {
	char* text;
	char* pattern;
	int should_fail;
	int length;
	int correctlen;
	size_t ntests = sizeof(test_vector) / sizeof(*test_vector);
	size_t nfailed = 0;
	size_t i;

	for (i = 0; i < ntests; ++i) {
		pattern = test_vector[i][1];
		text = test_vector[i][2];
		should_fail = (test_vector[i][0] == NOK);
		correctlen = (int)(uintptr_t)(test_vector[i][3]);

		int m = regex::re_match(pattern, text, &length);

		if (should_fail) {
			if (m != (-1)) {
				printf("\n");
				const auto p = regex::re_compile(pattern);
				regex::re_print(p);
				fprintf(stderr, "[%lu/%lu]: pattern '%s' matched '%s' unexpectedly, matched %i chars. \n", (i+1), ntests, pattern, text, length);
				nfailed += 1;
			}
		} else {
			if (m == (-1)) {
				printf("\n");
				regex::re_print(regex::re_compile(pattern));
				fprintf(stderr, "[%lu/%lu]: pattern '%s' didn't match '%s' as expected. \n", (i+1), ntests, pattern, text);
				nfailed += 1;
			} else if (length != correctlen) {
				fprintf(stderr, "[%lu/%lu]: pattern '%s' matched '%i' chars of '%s'; expected '%i'. \n", (i+1), ntests, pattern, length, text, correctlen);
				nfailed += 1;
			}
		}
	}

	// printf("\n");
	printf("%lu/%lu tests succeeded.\n", ntests - nfailed, ntests);
	printf("\n");
	printf("\n");
	printf("\n");
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
