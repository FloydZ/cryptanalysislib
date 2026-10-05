#pragma once

namespace cryptanalysislib {
	//    1.2. Tonelli-Shanks algorithm. Given prime p and integer
	//1 ≤ n < p, returns the square root r of n modulo p. There is
	//also another solution given by −r modulo p.
	template<typename T>
	T legendre(T a, T p) {
		if (a % p == 0) return 0;
		{
			return p == 2 || mod_pow(a, (p - 1) / 2, p) == 1 ? 1 : -1;
		}
	}

	template<typename T>
	T tonelli_shanks(T n, T p) {
		assert(legendre(n, p) == 1);
		if (p == 2) return 1;
		T s = 0, q = p - 1, z = 2;
		while (~q & 1) s++, q >>= 1;
		if (s == 1) return mod_pow(n, (p + 1) / 4, p);
		while (legendre(z, p) != -1) z++;
		T c = mod_pow(z, q, p),
		  r = mod_pow(n, (q + 1) / 2, p),
		  t = mod_pow(n, q, p),
		  m = s;
		while (t != 1) {
			T i = 1, ts = (T) t * t % p;
			while (ts != 1) i++, ts = ((T) ts * ts) % p;
			T b = mod_pow(c, 1ull << (m - i - 1), p);
			r = (T) r * b % p;
			t = (T) t * b % p * b % p;
			c = (T) b * b % p;
			m = i;
		}
		return r;
	}

}// namespace cryptanalysislib
