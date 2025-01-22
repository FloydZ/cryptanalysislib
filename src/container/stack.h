#pragma once

#include <stack>

template<typename T>
struct min_stack {
	std::stack<int> S, M;
	void push(int x) {
		S.push(x);
		M.push(M.empty() ? x : std::min(M.top(), x));
	}

	int top() { 
        return S.top(); 
    }

	int mn() { 
        return M.top(); 
    }
	void pop() {
		S.pop();
		M.pop();
	}
	bool empty() { return S.empty(); }
};

template<typename T>
struct min_queue {
	min_stack<T> inp, outp;
	void push(int x) { inp.push(x); }
	void fix() {
		if (outp.empty())
			while (!inp.empty())
				outp.push(inp.top()), inp.pop();
	}
	int top() {
		fix();
		return outp.top();
	}
	int mn() {
		if (inp.empty()) return outp.mn();
		if (outp.empty()) return inp.mn();
		return min(inp.mn(), outp.mn());
	}
	void pop() {
		fix();
		outp.pop();
	}
	bool empty() { return inp.empty() && outp.empty(); }
};
