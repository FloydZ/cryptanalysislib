#pragma once 

// TODO finish 

// Solving linear recurrences. Given some brute-forced
// sequence s[0], s[1], . . . , s[2n − 1], Berlekamp-Massey finds the
// shortest possible recurrence relation in O(n2). After that,
// lin_rec finds s[k] in O(n2 log k).
// Given a sequence s[0], ..., s[2n-1] finds the smallest linear recurrence↪→
// of size <= n compatible with s.
vl BerlekampMassey(const vl &s, ll mod) {
    int n = sz(s), L = 0, m = 0;
    vl C(n), B(n), T;
    C[0] = B[0] = 1;
    ll b = 1;
    REP(i, n) {
    ++m;
    ll d = s[i] % mod;
    rep(j, 1, L+1) d = (d + C[j] * s[i - j]) % mod;
    if (!d) continue;
    T = C;
    ll coef = d * modpow(b, mod-2, mod) % mod;
    rep(j,m,n) C[j] = (C[j] - coef * B[j-m]) % mod;
    if (2 * L > i) continue;
    L = i + 1 - L;
    B = T; b = d; m = 0;
    }
    C.resize(L + 1);
    C.erase(C.begin());
    for (auto &x : C) x = (mod - x) % mod;
    return C;
}

// Input: A[0,...,n-1], C[0,...,n-1] satisfying
// A[i] = \sum_{j=1}^{n} C[j-1] A[i-j],
// Outputs A[k]
ll lin_rec(const vl &A, const vl &C, ll k, ll mod){
    int n = sz(A);
    auto combine = [&](vl a, vl b) {
    vl res(sz(a) + sz(b) - 1, 0);
    REP(i, sz(a)) REP(j, sz(b))
    res[i+j] = (res[i+j] + a[i]*b[j]) % mod;
    for (int i = 2*n; i > n; --i) REP(j,n)
    res[i-1-j] = (res[i-1-j] + res[i]*C[j]) % mod;
    res.resize(n + 1);
    return res;
    };
    vl pol(n + 1), e(pol);
    pol[0] = e[1] = 1;
    for (++k; k; k /= 2) {
    if (k % 2) pol = combine(pol, e);
    e = combine(e, e);
    }
    ll res = 0;
    REP(i, n) res = (res + pol[i + 1] * A[i]) % mod;
    return res;
}
