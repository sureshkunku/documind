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

## OOPS 
Object oriented programing language 

samples/neural_networks.md           1176 chars  hash=a20994e89ec841d2
samples/python_basics.md              949 chars  hash=853af4581a1e3577
samples/sql_basics.md                 900 chars  hash=4868dcc1a381f9a8
samples/transformers.md              1069 chars  hash=fb2b198e9e67381a