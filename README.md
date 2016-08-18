TPPB
----

the thibm purpose partition balancer balances an array into K partitions of
roughly equal sums. it guesses a likely candidate partitioning using
one of several schemes and then pushes on the partition boundaries until
no local improvements can be made.

`python tppb.py N K` creates several test arrays of size N and splits them into
K partitions. if you have cython installed, use `python main.py N K` instead.
