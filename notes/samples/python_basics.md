# Python Basics

## Variables and types
Python is dynamically typed: a variable's type is decided at run time. Common built-in types are int, float, str, bool, list, tuple, dict and set.

## Lists vs tuples
A list is mutable, so items can be added or changed. A tuple is immutable, so it cannot change after creation. Tuples can be used as dictionary keys; lists cannot.

## For loops
A for loop repeats a block once for each item in an iterable. range(5) produces 0, 1, 2, 3, 4. The break statement exits the loop early and continue skips to the next item.

## Functions
Functions are defined with def. Default argument values are evaluated once, when the function is defined, so a mutable default like a list is shared between calls. Use None as the default instead.

## List comprehensions
A list comprehension builds a list in one line: [x * 2 for x in numbers if x > 0]. It is usually faster and clearer than an equivalent for loop with append.
