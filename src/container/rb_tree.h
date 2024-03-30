#ifndef CRYPTANALYSISLIB_RB_TREE_H
#define CRYPTANALYSISLIB_RB_TREE_H

#include <cstddef>
#include <cstdlib>
#include <vector>

#include "helper.h"
#include <bits/stl_tree.h>

enum RB_Tree_Color {
	RED,
	BLACK,
};
struct _Rb_tree_node_base {
	typedef _Rb_tree_node_base *_Base_ptr;
	typedef const _Rb_tree_node_base *_Const_Base_ptr;

	RB_Tree_Color _M_color;
	_Base_ptr _M_parent;
	_Base_ptr _M_left;
	_Base_ptr _M_right;

	constexpr inline static _Base_ptr
	_S_minimum(_Base_ptr __x) noexcept {
		while (__x->_M_left != 0) __x = __x->_M_left;
		return __x;
	}

	constexpr inline static _Const_Base_ptr
	_S_minimum(_Const_Base_ptr __x) noexcept {
		while (__x->_M_left != 0) __x = __x->_M_left;
		return __x;
	}

	constexpr inline static _Base_ptr
	_S_maximum(_Base_ptr __x) noexcept {
		while (__x->_M_right != 0) __x = __x->_M_right;
		return __x;
	}

	constexpr inline static _Const_Base_ptr
	_S_maximum(_Const_Base_ptr __x) noexcept {
		while (__x->_M_right != 0) __x = __x->_M_right;
		return __x;
	}
};

// Helper type offering value initialization guarantee on the compare functor.
template<typename _Key_compare>
struct _Rb_tree_key_compare {
	_Key_compare _M_key_compare;

	constexpr _Rb_tree_key_compare() noexcept : _M_key_compare() {}
	constexpr _Rb_tree_key_compare(const _Key_compare &__comp) noexcept : _M_key_compare(__comp) {}
};

// Helper type to manage default initialization of node count and header.
struct _Rb_tree_header {
	_Rb_tree_node_base _M_header;
	size_t _M_node_count;// Keeps track of size of tree.

	// base constructor
	constexpr _Rb_tree_header() noexcept {
		_M_header._M_color = RED;
		_M_reset();
	}

	// copy constructor
	constexpr _Rb_tree_header(_Rb_tree_header &&__x) noexcept {
		if (__x._M_header._M_parent != nullptr) {
			_M_move_data(__x);
		} else {
			_M_header._M_color = RED;
			_M_reset();
		}
	}

	constexpr void _M_move_data(_Rb_tree_header &__from) noexcept {
		_M_header._M_color = __from._M_header._M_color;
		_M_header._M_parent = __from._M_header._M_parent;
		_M_header._M_left = __from._M_header._M_left;
		_M_header._M_right = __from._M_header._M_right;
		_M_header._M_parent->_M_parent = &_M_header;
		_M_node_count = __from._M_node_count;
		__from._M_reset();
	}

	constexpr inline void _M_reset() noexcept {
		_M_header._M_parent = 0;
		_M_header._M_left = &_M_header;
		_M_header._M_right = &_M_header;
		_M_node_count = 0;
	}
};

template<typename V>
struct _Rb_tree_node : public _Rb_tree_node_base {
private:
	alignas(alignof(V)) V _M_storage;

public:
	typedef _Rb_tree_node<V> *_Link_type;

	constexpr inline V *ptr() { return &_M_storage; }
};


template<typename K,
         typename V,
         typename _KeyOfValue,
         typename _Compare = std::equal_to<K>,
         typename _Alloc = std::allocator<V>>
class RB_Tree {
	typedef typename __gnu_cxx::__alloc_traits<_Alloc>::template rebind<_Rb_tree_node<V>>::other _Node_allocator;
	typedef __gnu_cxx::__alloc_traits<_Node_allocator> _Alloc_traits;

protected:
	typedef _Rb_tree_node_base *_Base_ptr;
	typedef const _Rb_tree_node_base *_Const_Base_ptr;
	typedef _Rb_tree_node<V> *_Link_type;
	typedef const _Rb_tree_node<V> *_Const_Link_type;

private:
	// Functor recycling a pool of nodes and using allocation once the pool
	// is empty.
	struct _Reuse_or_alloc_node {
		constexpr _Reuse_or_alloc_node(RB_Tree &__t) noexcept
		    : _M_root(__t._M_root()), _M_nodes(__t._M_rightmost()), _M_t(__t) {
			if (_M_root) {
				_M_root->_M_parent = 0;

				if (_M_nodes->_M_left) {
					_M_nodes = _M_nodes->_M_left;
				}
			} else {
				_M_nodes = 0;
			}
		}

		constexpr _Reuse_or_alloc_node(const _Reuse_or_alloc_node &) = delete;
		constexpr ~_Reuse_or_alloc_node() { _M_t._M_erase(static_cast<_Link_type>(_M_root)); }

		template<typename _Arg>
		constexpr _Link_type
		operator()(_Arg &&__arg) noexcept {
			_Link_type __node = static_cast<_Link_type>(_M_extract());
			if (__node) {
				_M_t._M_destroy_node(__node);
				_M_t._M_construct_node(__node, _GLIBCXX_FORWARD(_Arg, __arg));
				return __node;
			}

			return _M_t._M_create_node(_GLIBCXX_FORWARD(_Arg, __arg));
		}

	private:
		constexpr _Base_ptr _M_extract() noexcept {
			if (!_M_nodes) {
				return _M_nodes;
			}

			_Base_ptr __node = _M_nodes;
			_M_nodes = _M_nodes->_M_parent;
			if (_M_nodes) {
				if (_M_nodes->_M_right == __node) {
					_M_nodes->_M_right = 0;

					if (_M_nodes->_M_left) {
						_M_nodes = _M_nodes->_M_left;

						while (_M_nodes->_M_right) {
							_M_nodes = _M_nodes->_M_right;
						}

						if (_M_nodes->_M_left) {
							_M_nodes = _M_nodes->_M_left;
						}
					}
				} else {// __node is on the left.
					_M_nodes->_M_left = 0;
				}
			} else {
				_M_root = 0;
			}

			return __node;
		}

		_Base_ptr _M_root;
		_Base_ptr _M_nodes;
		RB_Tree &_M_t;
	};// end Reuse of Alloc Node

	// Functor similar to the previous one but without any pool of nodes to
	// recycle.
	struct _Alloc_node {
		constexpr _Alloc_node(RB_Tree &__t) noexcept : _M_t(__t) {}

		template<typename _Arg>
		constexpr _Link_type
		operator()(_Arg &&__arg) const {
			return _M_t._M_create_node(_GLIBCXX_FORWARD(_Arg, __arg));
		}

	private:
		RB_Tree &_M_t;
	};// end _Alloc_Node

public:
	typedef K key_type;
	typedef V value_type;
	typedef value_type *pointer;
	typedef const value_type *const_pointer;
	typedef value_type &reference;
	typedef const value_type &const_reference;
	typedef size_t size_type;
	typedef ptrdiff_t difference_type;
	typedef _Alloc allocator_type;

protected:
	constexpr inline _Link_type
	_M_get_node() {
		return _Alloc_traits::allocate(_M_get_Node_allocator(), 1);
	}

	constexpr inline void
	_M_put_node(_Link_type __p) noexcept {
		_Alloc_traits::deallocate(_M_get_Node_allocator(), __p, 1);
	}

	/// allocator stuff
	constexpr inline _Node_allocator &
	_M_get_Node_allocator() noexcept { return this->_M_impl; }

	constexpr inline const _Node_allocator &
	_M_get_Node_allocator() const noexcept { return this->_M_impl; }

	constexpr inline allocator_type
	get_allocator() const noexcept { return allocator_type(_M_get_Node_allocator()); }


	template<typename... _Args>
	void
	_M_construct_node(_Link_type __node, _Args &&...__args) {
		::new (__node) _Rb_tree_node<value_type>;
		_Alloc_traits::construct(_M_get_Node_allocator(),
		                         __node->_M_valptr(),
		                         std::forward<_Args>(__args)...);
	}

	template<typename... _Args>
	_Link_type
	_M_create_node(_Args &&...__args) {
		_Link_type __tmp = _M_get_node();
		_M_construct_node(__tmp, std::forward<_Args>(__args)...);
		return __tmp;
	}

	/// base allocator class
	template<typename _Key_compare>
	struct _Rb_tree_impl
	    : public _Node_allocator,
	      public _Rb_tree_key_compare<_Key_compare>,
	      public _Rb_tree_header {

		typedef _Rb_tree_key_compare<_Key_compare> _Base_key_compare;

		constexpr _Rb_tree_impl() noexcept
		    : _Node_allocator() {}

		constexpr _Rb_tree_impl(const _Rb_tree_impl &__x) noexcept
		    : _Node_allocator(_Alloc_traits::_S_select_on_copy(__x)), _Base_key_compare(__x._M_key_compare), _Rb_tree_header() {}

		_Rb_tree_impl(_Rb_tree_impl &&) noexcept(std::is_nothrow_move_constructible<_Base_key_compare>::value) = default;

		explicit _Rb_tree_impl(_Node_allocator &&__a)
		    : _Node_allocator(std::move(__a)) {}

		_Rb_tree_impl(_Rb_tree_impl &&__x, _Node_allocator &&__a)
		    : _Node_allocator(std::move(__a)),
		      _Base_key_compare(std::move(__x)),
		      _Rb_tree_header(std::move(__x)) {}

		_Rb_tree_impl(const _Key_compare &__comp, _Node_allocator &&__a)
		    : _Node_allocator(std::move(__a)), _Base_key_compare(__comp) {}
	};

	/// This is really interesting.
	_Rb_tree_impl<_Compare> _M_impl;








	template<typename _Arg>
	pair<iterator, bool>
	_M_insert_unique(_Arg &&__x);

	template<typename _Arg>
	iterator
	_M_insert_equal(_Arg &&__x);

	template<typename _Arg, typename _NodeGen>
	iterator
	_M_insert_unique_(const_iterator __pos, _Arg &&__x, _NodeGen &);

	template<typename _Arg>
	iterator
	_M_insert_unique_(const_iterator __pos, _Arg &&__x) {
		_Alloc_node __an(*this);
		return _M_insert_unique_(__pos, std::forward<_Arg>(__x), __an);
	}

	template<typename _Arg, typename _NodeGen>
	iterator
	_M_insert_equal_(const_iterator __pos, _Arg &&__x, _NodeGen &);

	template<typename _Arg>
	iterator
	_M_insert_equal_(const_iterator __pos, _Arg &&__x) {
		_Alloc_node __an(*this);
		return _M_insert_equal_(__pos, std::forward<_Arg>(__x), __an);
	}

	template<typename... _Args>
	pair<iterator, bool>
	_M_emplace_unique(_Args &&...__args);

	template<typename... _Args>
	iterator
	_M_emplace_equal(_Args &&...__args);

	template<typename... _Args>
	iterator
	_M_emplace_hint_unique(const_iterator __pos, _Args &&...__args);

	template<typename... _Args>
	iterator
	_M_emplace_hint_equal(const_iterator __pos, _Args &&...__args);

	template<typename _Iter>
	using __same_value_type = is_same<value_type, typename iterator_traits<_Iter>::value_type>;

	template<typename _InputIterator>
	__enable_if_t<__same_value_type<_InputIterator>::value>
	_M_insert_range_unique(_InputIterator __first, _InputIterator __last) {
		_Alloc_node __an(*this);
		for (; __first != __last; ++__first)
			_M_insert_unique_(end(), *__first, __an);
	}

	template<typename _InputIterator>
	__enable_if_t<!__same_value_type<_InputIterator>::value>
	_M_insert_range_unique(_InputIterator __first, _InputIterator __last) {
		for (; __first != __last; ++__first)
			_M_emplace_unique(*__first);
	}

	template<typename _InputIterator>
	__enable_if_t<__same_value_type<_InputIterator>::value>
	_M_insert_range_equal(_InputIterator __first, _InputIterator __last) {
		_Alloc_node __an(*this);
		for (; __first != __last; ++__first)
			_M_insert_equal_(end(), *__first, __an);
	}

	template<typename _InputIterator>
	__enable_if_t<!__same_value_type<_InputIterator>::value>
	_M_insert_range_equal(_InputIterator __first, _InputIterator __last) {
		for (; __first != __last; ++__first)
			_M_emplace_equal(*__first);
	}






};// end of tree class


#endif
