What?
-------
Instead of just running a program and pass the tree/list/value/ options via the command line, we heavily rely on
compile time optimisations. That's why we outsource all options to a config header file. 
You'll find all possible options in `template.h`

Why?
----
compile time optimisations and `C++17` `constexpr` features

How?
---
Just create a copy of `template.h` in this folder e.g. `new_approach.h`. Set all parameters as you wish within it. 
Subsequent make sure that you add `#include "config/new_approach.h"` before all other configs in `helper.h`. And thats it.