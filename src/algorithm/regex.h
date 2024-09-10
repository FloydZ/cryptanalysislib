#ifndef CRYPTANALYSISLIB_ALGORITHM_REGEX
#define CRYPTANALYSISLIB_ALGORITHM_REGEX

// Mini regex-module inspired by Rob Pike's regex code described in:
//   http://www.cs.princeton.edu/courses/archive/spr09/cos333/beautiful.html
// org source code from:
//   https://github.com/kokke/tiny-regex-c
//
// Supports:
// ---------
//   '.'        Dot, matches any character
//   '^'        Start anchor, matches beginning of string
//   '$'        End anchor, matches end of string
//   '*'        Asterisk, match zero or more (greedy)
//   '+'        Plus, match one or more (greedy)
//   '?'        Question, match zero or one (non-greedy)
//   '[abc]'    Character class, match if one of {'a', 'b', 'c'}
//   '[^abc]'   Inverted class, match if NOT one of {'a', 'b', 'c'} -- NOTE: feature is currently broken!
//   '[a-zA-Z]' Character ranges, the character set of the ranges { a-z | A-Z }
//   '\s'       Whitespace, \t \f \r \n \v and spaces
//   '\S'       Non-whitespace
//   '\w'       Alphanumeric, [a-zA-Z0-9_]
//   '\W'       Non-alphanumeric
//   '\d'       Digits, [0-9]
//   '\D'       Non-digits
//   '\xXX'     Hex-encoded byte
//   '|'        Branch Or, e.g. a|A, \w|\s
//   '{n}'      Match n times
//   '{n,}'     Match n or more times
//   '{,m}'     Match m or less times
//   '{n,m}'    Match n to m times
//
// FIXME:
//   - '(...)'    Group
//	 - c++ interface via strings

#include <cstdio>
#include <cstdlib>
#include <ctype.h>


#ifdef DEBUG
#define DEBUG_P(...)// fprintf(stderr, __VA_ARGS__)
#else
#define DEBUG_P(...)
#endif

enum regexype_e {
	UNUSED,
	DOT,
	BEGIN,
	END,
	QUESTIONMARK,
	STAR,
	PLUS,
	CHAR,
	CHAR_CLASS,
	INV_CHAR_CLASS,
	DIGIT,
	NOT_DIGIT,
	ALPHA,
	NOT_ALPHA,
	WHITESPACE,
	NOT_WHITESPACE,
	BRANCH,
	GROUP,
	GROUPEND,
	TIMES,
	TIMES_N,
	TIMES_M,
	TIMES_NM
};

struct regex {
	enum regexype_e type; /* CHAR, STAR, etc.                      */
	union {
		char ch;                   /*      the character itself             */
		char *ccl;                 /*  OR  a pointer to characters in class */
		unsigned char group_num;   /*  OR the number of group patterns. */
		unsigned char group_start; /*  OR for GROUPEND, the start index of the group. */
		struct {
			unsigned short n; /* match n times */
			unsigned short m; /* match n to m times */
		};
	} u;

private:
	// TODO move into config
	constexpr static bool dot_matches_newline = true;

	// Max number of regex symbols in expression
	constexpr static size_t MAX_REGEXP_OBJECTS = 30u;

	// Max length of character-class buffer in.
	constexpr static size_t MAX_CHAR_CLASS_LEN = 40u;

	constexpr static int hex(const char c) noexcept {
		if (c >= 'a' && c <= 'f')
			return c - 'a' + 10;
		else if (c >= 'A' && c <= 'F')
			return c - 'A' + 10;
		else if (c >= '0' && c <= '9')
			return c - '0';
		else
			return -1;
	}

	constexpr static inline int matchdigit(const char c) noexcept {
		return isdigit((unsigned char) c);
	}

	constexpr static inline int matchalpha(const char c) noexcept {
		return isalpha((unsigned char) c);
	}

	constexpr static inline int matchwhitespace(const char c) noexcept {
		return isspace((unsigned char) c);
	}

	constexpr static inline int matchalphanum(const char c) noexcept {
		return ((c == '_') || matchalpha(c) || matchdigit(c));
	}

	constexpr static inline int matchrange(const char c,
	                                       const char *str) noexcept {
		return ((c != '-') && (str[0] != '\0') && (str[0] != '-') &&
		        (str[1] == '-') && (str[2] != '\0') && ((c >= str[0]) && (c <= str[2])));
	}

	constexpr static inline int matchdot(const char c) noexcept {
		if constexpr (dot_matches_newline) {
			(void) c;
			return 1;
		} else {
			return c != '\n' && c != '\r';
		}
	}

	constexpr static inline int ismetachar(char c) noexcept {
		return ((c == 's') || (c == 'S') || (c == 'w') ||
		        (c == 'W') || (c == 'd') || (c == 'D'));
	}

	constexpr static int matchmetachar(const char c,
	                                   const char *str) noexcept {
		switch (str[0]) {
			case 'd':
				return matchdigit(c);
			case 'D':
				return !matchdigit(c);
			case 'w':
				return matchalphanum(c);
			case 'W':
				return !matchalphanum(c);
			case 's':
				return matchwhitespace(c);
			case 'S':
				return !matchwhitespace(c);
			default:
				return (c == str[0]);
		}
	}

	constexpr static int matchcharclass(const char c,
	                                    const char *str) noexcept {
		do {
			if (matchrange(c, str)) {
				return 1;
			} else if (str[0] == '\\') {
				/* Escape-char: increment str-ptr and match on next char */
				str += 1;
				if (matchmetachar(c, str)) {
					return 1;
				} else if ((c == str[0]) && !ismetachar(c)) {
					return 1;
				}
			} else if (c == str[0]) {
				if (c == '-') {
					return ((str[-1] == '\0') || (str[1] == '\0'));
				} else {
					return 1;
				}
			}
		} while (*str++ != '\0');

		return 0;
	}

	constexpr static int matchone(const regex &p,
	                              const char c) noexcept {
		switch (p.type) {
			case DOT:
				return matchdot(c);
			case CHAR_CLASS:
				return matchcharclass(c, (const char *) p.u.ccl);
			case INV_CHAR_CLASS:
				return !matchcharclass(c, (const char *) p.u.ccl);
			case DIGIT:
				return matchdigit(c);
			case NOT_DIGIT:
				return !matchdigit(c);
			case ALPHA:
				return matchalphanum(c);
			case NOT_ALPHA:
				return !matchalphanum(c);
			case WHITESPACE:
				return matchwhitespace(c);
			case NOT_WHITESPACE:
				return !matchwhitespace(c);
			case GROUPEND:
				return 1;
			default:
				return (p.u.ch == c);
		}
	}


	static int matchstar(regex p, regex *pattern, const char *text, int *matchlength) {
		int num_patterns = 0;
		return matchplus(p, pattern, text, matchlength) ||
		       matchpattern(pattern, text, matchlength, &num_patterns);
	}

	static int matchplus(regex p, regex *pattern, const char *text, int *matchlength) {
		int num_patterns = 0;
		const char *prepoint = text;
		while ((text[0] != '\0') && matchone(p, *text)) {
			DEBUG_P("+ matches %s\n", text);
			text++;
		}
		for (; text > prepoint; text--) {
			if (matchpattern(pattern, text, matchlength, &num_patterns)) {
				*matchlength += text - prepoint;
				return 1;
			}
			DEBUG_P("+ pattern does not match %s\n", &text[1]);
		}
		DEBUG_P("+ pattern did not match %s\n", prepoint);
		return 0;
	}

	static int matchquestion(regex p, regex *pattern, const char *text, int *matchlength) {
		int num_patterns = 0;
		if (p.type == UNUSED)
			return 1;
		if (matchpattern(pattern, text, matchlength, &num_patterns)) {
			return 1;
		}
		if (*text && matchone(p, *text++)) {
			if (matchpattern(pattern, text, matchlength, &num_patterns)) {
				(*matchlength)++;
				return 1;
			}
		}
		return 0;
	}

	static int matchtimes(regex p, unsigned short n, const char *text, int *matchlength) {
		unsigned short i = 0;
		int pre = *matchlength;
		/* Match the pattern n to m times */
		while (*text && matchone(p, *text++) && i < n) {
			(*matchlength)++;
			i++;
		}
		if (i == n)
			return 1;
		*matchlength = pre;
		return 0;
	}

	static int matchtimes_n(regex p, unsigned short n, const char *text, int *matchlength) {
		unsigned short i = 0;
		int pre = *matchlength;
		/* Match the pattern n or more times */
		while (*text && matchone(p, *text++)) {
			(*matchlength)++;
			i++;
		}
		if (i >= n)
			return 1;
		*matchlength = pre;
		return 0;
	}

	static int matchtimes_m(regex p, unsigned short m, const char *text, int *matchlength) {
		unsigned short i = 0;
		/* Match the pattern max m times */
		while (*text && matchone(p, *text++) && i < m) {
			(*matchlength)++;
			i++;
		}
		return 1;
	}

	static int matchtimes_nm(regex p, unsigned short n, unsigned short m, const char *text, int *matchlength) {
		unsigned short i = 0;
		int pre = *matchlength;
		/* Match the pattern n to m times */
		while (*text && matchone(p, *text++) && i < m) {
			(*matchlength)++;
			i++;
		}
		if (i >= n && i <= m)
			return 1;
		*matchlength = pre;
		return 0;
	}

	static int matchbranch(regex p, regex *pattern, const char *text, int *matchlength) {
		int num_patterns = 0;
		const char *prepoint = text;
		if (p.type == UNUSED)
			return 1;
		/* Match the current p (previous) */
		if (*text && matchone(p, *text++)) {
			(*matchlength)++;
			return 1;
		}
		if (pattern->type == UNUSED)
			// empty branch "0|" allows NULL text
			return 1;
		/* or the next branch */
		if (matchpattern(pattern, prepoint, matchlength, &num_patterns))
			return 1;
		return 0;
	}

	static int matchgroup(regex *p, const char *text, int *matchlength) {
		int pre = *matchlength;
		int num_patterns = 0, length = pre;
		const regex *groupend = &p[p->u.group_num + 1];
		DEBUG_P("does GROUP (%u) match %s?\n", (unsigned) p->u.group_num, text);
		p++;
		while (p < groupend) {
			if (p->type == UNUSED)// only with invalid external compiles
				return 0;
			if (!matchpattern(p, text, &length, &num_patterns)) {
				DEBUG_P("GROUP did not match %.*s (len %d, patterns %d)\n", length, text - *matchlength, *matchlength, num_patterns);
				*matchlength = pre;
				return 0;
			}
			DEBUG_P("GROUP did match %.*s (len %d, patterns %d)\n", length, text - *matchlength, *matchlength, num_patterns);
			text += length;
			p += num_patterns;
			*matchlength += length;
		}
		DEBUG_P("ENDGROUP did match %s (len %d, patterns %d)\n", text - *matchlength, *matchlength, num_patterns);
		return 1;
	}

	/* Iterative matching */
	static int matchpattern(regex *pattern, const char *text, int *matchlength, int *num_patterns) {
		int pre = *matchlength;
		do {
			if ((pattern[0].type == UNUSED) || (pattern[1].type == QUESTIONMARK)) {
				int i = (pattern[1].type == GROUPEND) ? pattern[1].u.group_start : 0;
				return matchquestion(pattern[i], &pattern[2], text, matchlength);
			} else if (pattern[1].type == STAR) {
				int i = (pattern[1].type == GROUPEND) ? pattern[1].u.group_start : 0;
				return matchstar(pattern[i], &pattern[2], text, matchlength);
			} else if (pattern[1].type == PLUS) {
				DEBUG_P("PLUS match %s?\n", text);
				int i = (pattern[1].type == GROUPEND) ? pattern[1].u.group_start : 0;
				return matchplus(pattern[i], &pattern[2], text, matchlength);
			} else if (pattern[1].type == TIMES) {
				int i = (pattern[1].type == GROUPEND) ? pattern[1].u.group_start : 0;
				return matchtimes(pattern[i], pattern[1].u.n, text, matchlength);
			} else if (pattern[1].type == TIMES_N) {
				return matchtimes_n(pattern[0], pattern[1].u.n, text, matchlength);
			} else if (pattern[1].type == TIMES_M) {
				return matchtimes_m(pattern[0], pattern[1].u.m, text, matchlength);
			} else if (pattern[1].type == TIMES_NM) {
				int i = (pattern[1].type == GROUPEND) ? pattern[1].u.group_start : 0;
				return matchtimes_nm(pattern[i], pattern[1].u.n, pattern[1].u.m, text,
				                     matchlength);
			} else if (pattern[1].type == BRANCH) {
				int i = (pattern[1].type == GROUPEND) ? pattern[1].u.group_start : 0;
				return matchbranch(pattern[i], &pattern[2], text, matchlength);
			} else if (pattern[0].type == GROUPEND) {
				(*num_patterns)++;
				DEBUG_P("GROUPEND matches %.*s (len %d, patterns %d)\n", *matchlength, text - *matchlength, *matchlength, *num_patterns);
				return 1;
			} else if (pattern[0].type == GROUP) {
				*num_patterns = pattern[0].u.group_num + 1;// plus GROUPEND
				return matchgroup(&pattern[0], text, matchlength);
			} else if ((pattern[0].type == END) && pattern[1].type == UNUSED) {
				return (text[0] == '\0');
			}
			(*matchlength)++;
			(*num_patterns)++;
		} while ((text[0] != '\0') && matchone(*pattern++, *text++));

		*matchlength = pre;
		return 0;
	}

public:
	constexpr static int re_match(const char *pattern, const char *text, int *matchlength) noexcept {
		return re_matchp(re_compile(pattern), text, matchlength);
	}

	constexpr static int re_matchp(regex *pattern, const char *text, int *matchlength) noexcept {
		int num_patterns = 0;
		*matchlength = 0;
		if (pattern != 0) {
			if (pattern[0].type == BEGIN) {
				return ((matchpattern(&pattern[1], text, matchlength, &num_patterns)) ? 0 : -1);
			} else {
				int idx = -1;

				do {
					idx += 1;

					if (matchpattern(pattern, text, matchlength, &num_patterns)) {
						// empty branch matches null (i.e. ok, but *matchlength == 0)
						if (*matchlength && text[0] == '\0')
							return -1;

						return idx;
					}
				} while (*text++ != '\0');
			}
		}
		return -1;
	}

	constexpr static regex *re_compile(const char *pattern) noexcept {
		//The sizes of the three static arrays below substantiates the static RAM
     	//usage of this module.
     	//MAX_REGEXP_OBJECTS is the max number of symbols in the expression.
     	//MAX_CHAR_CLASS_LEN determines the size of the buffer for chars in all
        //char-classes in the expression.
		static regex re_compiled[MAX_REGEXP_OBJECTS];
		static char ccl_buf[MAX_CHAR_CLASS_LEN];
		size_t ccl_bufidx = 1;

		char c;    /* current char in pattern   */
		size_t i = 0; /* index into pattern        */
		size_t j = 0; /* index into re_compiled    */

		while (pattern[i] != '\0' && (j + 1 < MAX_REGEXP_OBJECTS)) {
			c = pattern[i];

			switch (c) {
				/* Meta-characters: */
				case '^': {
					re_compiled[j].type = BEGIN;
				} break;
				case '$': {
					re_compiled[j].type = END;
				} break;
				case '.': {
					re_compiled[j].type = DOT;
				} break;
				case '|': {
					re_compiled[j].type = BRANCH;
				} break;
				case '*': {
					if (j > 0)
						re_compiled[j].type = STAR;
					else// nothing to repeat at position 0
						return nullptr;
				} break;
				case '+': {
					if (j > 0)
						re_compiled[j].type = PLUS;
					else// nothing to repeat at position 0
						return nullptr;
				} break;
				case '?': {
					if (j > 0)
						re_compiled[j].type = QUESTIONMARK;
					else// nothing to repeat at position 0
						return nullptr;
				} break;

				case '(': {
					const char *p = strrchr(&pattern[i], ')');
					if (p && *(p - 1) != '\\') {
						re_compiled[j].type = GROUP;
						re_compiled[j].u.group_num = 0;
					} else {
						/* '(' without matching ')' */
						return nullptr;
					}

					break;
				}
				case ')': {
					int nestlevel = 0;
					size_t k = j;
					/* search back to next innermost groupstart */
					for (; k >= 0; k--) {
						if (k < j && re_compiled[k].type == GROUPEND)
							nestlevel++;
						else if (re_compiled[k].type == GROUP) {
							if (nestlevel == 0) {
								re_compiled[k].u.group_num = j - k - 1;
								re_compiled[j].type = GROUPEND;
								re_compiled[j].u.group_start = k;// index of group
								break;
							}
							nestlevel--;
						}
					}
					/* ')' without matching '(' */
					if (k < 0)
						return nullptr;
					break;
				}
				case '{': {
					unsigned short n, m;
					const char *p = strchr(&pattern[i + 1], '}');
					re_compiled[j].type = CHAR;
					re_compiled[j].u.ch = c;
					if (!p || j == 0)// those invalid quantifiers are compiled as is
					{                // (in python and perl)
						re_compiled[j].type = CHAR;
						re_compiled[j].u.ch = c;
					} else if (2 != sscanf(&pattern[i], "{%hd,%hd}", &n, &m)) {
						if (1 != sscanf(&pattern[i], "{%hd,}", &n) ||
						    n == 0 || n > 32767) {
							if (1 != sscanf(&pattern[i], "{,%hd}", &m) ||
							    *(p - 1) == ',' || m == 0 || m > 32767) {
								if (1 == sscanf(&pattern[i], "{%hd}", &n) &&
								    n > 0 && n <= 32767) {
									re_compiled[j].type = TIMES;
									re_compiled[j].u.n = n;
								}
							} else {
								re_compiled[j].type = TIMES_M;
								re_compiled[j].u.m = m;
							}
						} else {
							re_compiled[j].type = TIMES_N;
							re_compiled[j].u.n = n;
						}
					} else {
						// m must be greater than n, and none of them may be 0 or negative.
						if (!(n == 0 || m == 0 || n > 32767 || m > 32767 || m <= n || *(p - 1) == ',')) {
							re_compiled[j].type = TIMES_NM;
							re_compiled[j].u.n = n;
							re_compiled[j].u.m = m;
						}
					}
					if (re_compiled[j].type != CHAR)
						i += (p - &pattern[i]);
					break;
				}
				/* Escaped character-classes (\s \S \w \W \d \D \*): */
				case '\\': {
					if (pattern[i + 1] != '\0') {
						/* Skip the escape-char '\\' */
						i += 1;
						/* ... and check the next */
						switch (pattern[i]) {
							/* Meta-characters: */
							case 'd': {
								re_compiled[j].type = DIGIT;
							} break;
							case 'D': {
								re_compiled[j].type = NOT_DIGIT;
							} break;
							case 'w': {
								re_compiled[j].type = ALPHA;
							} break;
							case 'W': {
								re_compiled[j].type = NOT_ALPHA;
							} break;
							case 's': {
								re_compiled[j].type = WHITESPACE;
							} break;
							case 'S': {
								re_compiled[j].type = NOT_WHITESPACE;
							} break;
							case 'x': {
								/* \xXX */
								re_compiled[j].type = CHAR;
								i++;
								int h = hex(pattern[i]);
								if (h == -1) {
									re_compiled[j].u.ch = '\\';
									re_compiled[j].type = CHAR;
									re_compiled[++j].u.ch = 'x';
									re_compiled[j].type = CHAR;
									re_compiled[++j].u.ch = pattern[i];
									re_compiled[j].type = CHAR;
									break;
								}
								re_compiled[j].u.ch = h << 4;
								h = hex(pattern[++i]);
								if (h != -1)
									re_compiled[j].u.ch += h;
								else {
									re_compiled[j].u.ch = '\\';
									re_compiled[j].type = CHAR;
									re_compiled[++j].u.ch = 'x';
									re_compiled[j].type = CHAR;
									re_compiled[++j].u.ch = pattern[i - 1];
									re_compiled[j].type = CHAR;
									if (pattern[i]) {
										re_compiled[++j].u.ch = pattern[i];
										re_compiled[j].type = CHAR;
									}
								}
							} break;

							/* Escaped character, e.g. '.', '$' or '\\' */
							default: {
								re_compiled[j].type = CHAR;
								re_compiled[j].u.ch = pattern[i];
							} break;
						}
					}
					/* '\\' as last char without previous \\ -> invalid regular expression. */
					else
						return nullptr;
				} break;

				/* Character class: */
				case '[': {
					/* Remember where the char-buffer starts. */
					int buf_begin = ccl_bufidx;

					/* Look-ahead to determine if negated */
					if (pattern[i + 1] == '^') {
						re_compiled[j].type = INV_CHAR_CLASS;
						i += 1;                  /* Increment i to avoid including '^' in the char-buffer */
						if (pattern[i + 1] == 0) /* incomplete pattern, missing non-zero char after '^' */
						{
							return nullptr;
						}
					} else {
						re_compiled[j].type = CHAR_CLASS;
					}

					/* Copy characters inside [..] to buffer */
					while ((pattern[++i] != ']') && (pattern[i] != '\0')) /* Missing ] */
					{
						if (pattern[i] == '\\') {
							if (ccl_bufidx >= MAX_CHAR_CLASS_LEN - 1) {
								return nullptr;
							}
							if (pattern[i + 1] == 0) /* incomplete pattern, missing non-zero char after '\\' */
							{
								return nullptr;
							}
							ccl_buf[ccl_bufidx++] = pattern[i++];
						} else if (ccl_bufidx >= MAX_CHAR_CLASS_LEN) {
							return nullptr;
						}
						ccl_buf[ccl_bufidx++] = pattern[i];
					}
					if (ccl_bufidx >= MAX_CHAR_CLASS_LEN) {
						/* Catches cases such as [00000000000000000000000000000000000000][ */
						return nullptr;
					}
					/* Null-terminate string end */
					ccl_buf[ccl_bufidx++] = 0;
					re_compiled[j].u.ccl = &ccl_buf[buf_begin];
				} break;

				case '\0':// EOL (dead-code)
					return nullptr;

				/* Other characters: */
				default: {
					re_compiled[j].type = CHAR;
					// cbmc: arithmetic overflow on signed to unsigned type conversion in c
					re_compiled[j].u.ch = c;
				} break;
			}
			i += 1;
			j += 1;
		}
		/* 'UNUSED' is a sentinel used to indicate end-of-pattern */
		re_compiled[j].type = UNUSED;

		return (regex *) re_compiled;
	}


	constexpr static void re_print(regex *pattern) noexcept {
		const char *const types[] = {
		        "UNUSED", "DOT", "BEGIN", "END", "QUESTIONMARK", "STAR",
		        "PLUS", "CHAR", "CHAR_CLASS", "INV_CHAR_CLASS", "DIGIT",
		        "NOT_DIGIT", "ALPHA", "NOT_ALPHA", "WHITESPACE", "NOT_WHITESPACE",
		        "BRANCH", "GROUP", "GROUPEND", "TIMES", "TIMES_N", "TIMES_M", "TIMES_NM"};
		unsigned char i;
		unsigned char j;
		unsigned char group_end = 0;
		char c;

		if (!pattern) {
			return;
		}

		for (i = 0; i < MAX_REGEXP_OBJECTS; ++i) {
			if (pattern[i].type == UNUSED) {
				break;
			}

			//if (group_end && i == group_end)
			//  printf("      )\n");
			if (pattern[i].type <= TIMES_NM)
				printf("type: %s", types[pattern[i].type]);
			else
				printf("invalid type: %d", pattern[i].type);

			if (pattern[i].type == CHAR_CLASS || pattern[i].type == INV_CHAR_CLASS) {
				printf(" [");
				if (pattern[i].type == INV_CHAR_CLASS)
					printf("^");
				for (j = 0; j < MAX_CHAR_CLASS_LEN; ++j) {
					c = pattern[i].u.ccl[j];
					if ((c == '\0') || (c == ']')) {
						break;
					}
					printf("%c", c);
				}
				printf("]");
			} else if (pattern[i].type == CHAR) {
				printf(" '%c'", pattern[i].u.ch);
			} else if (pattern[i].type == TIMES) {
				printf("{%hu}", pattern[i].u.n);
			} else if (pattern[i].type == TIMES_N) {
				printf("{%hu,}", pattern[i].u.m);
			} else if (pattern[i].type == TIMES_M) {
				printf("{,%hu}", pattern[i].u.n);
			} else if (pattern[i].type == TIMES_NM) {
				printf("{%hu,%hu}", pattern[i].u.n, pattern[i].u.m);
			} else if (pattern[i].type == GROUP) {
				group_end = i + pattern[i].u.group_num;
				if (group_end >= MAX_REGEXP_OBJECTS)
					return;
				printf(" (");
			} else if (pattern[i].type == GROUPEND) {
				printf(" )");
			}
			printf("\n");
		}
	}
};

#endif
