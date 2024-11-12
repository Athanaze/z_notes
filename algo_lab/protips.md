 In particular, we want to be able to carry out the
following two operations efficiently: (1) removing a weight, and (2) finding the heaviest weight
which is still at most s[i].
Fortunately for us, there is a data structure in the C++ STL that can do both of these things
efficiently: std::multiset. Both removing elements and answering queries of the form in (2)
takes O(log N) time in a std::multiset with N elements.