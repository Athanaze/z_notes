# hydra

### **Understanding the Problem**

Imagine the Hydra has **n** heads labeled from **0** to **n - 1**. Hercules has a list of **m** eradication patterns, each of length **k**. Each pattern is a sequence of head labels that, if matched by the last **k** cuts, can eradicate a specific head.

Our challenge is to:

- **Eradicate all heads** in the correct order (from head **0** to head **n - 1**).
- **Minimize the total number of cuts**.
- **Avoid cutting eradicated heads**, since once a head is eradicated, it cannot be cut again.

### **Strategizing with Dynamic Programming**

To tackle this, we'll employ **Dynamic Programming (DP)**, a method perfectly suited for problems with overlapping subproblems and optimal substructure.

#### **1. Enumerating Possible Patterns**

First, Hercules organizes Athena's eradication patterns by the head they eradicate. Let's denote:

- **P[h]**: A list of all eradication patterns that can eradicate head **h**.

This step ensures Hercules knows all possible ways to eradicate each head.

#### **2. Checking for Impossible Cases**

Before proceeding, Hercules checks:

- **If any head lacks eradication patterns**, it's impossible to eradicate all heads.

In code:

```cpp
// Check if every head has at least one eradication pattern
bool impossible = false;
for(int h = 0; h < n; h++){
    if(P[h].empty()){
        impossible = true;
        break;
    }
}
if(impossible){
    cout << "Impossible!\n";
    continue;
}
```

#### **3. Calculating Overlaps Between Patterns**

Hercules realizes that to minimize cuts, he should **overlap patterns** where possible. Overlapping reduces the number of new cuts needed when moving from eradicating head **i** to head **i + 1**.

He defines a function:

```cpp
int getOverlap(const vector<int> &p, const vector<int> &q, int k, int tillHead)
```

This function computes the maximum overlap **t** between patterns **p** and **q** such that:

- The last **t** elements of **p** match the first **t** elements of **q**.
- The **new cuts** (positions after the overlap in **q**) do not involve eradicated heads (labels less than **tillHead**).

This mirrors Hercules's cautious approach: when moving to the next head, he ensures he doesn't cut any already eradicated heads.

#### **4. Setting Up Dynamic Programming State**

Hercules sets up a DP table:

- **dp[i][u]**: The minimum total number of cuts to eradicate heads **0** to **i**, ending with using the **u-th** pattern in **P[i]** to eradicate head **i**.

He initializes the base case:

- **dp[0][u] = k** for all patterns **u** in **P[0]**, since no heads have been eradicated yet.

In code:

```cpp
for(int u = 0; u < P[0].size(); u++){
    dp[0][u] = k;
}
```

#### **5. Filling the DP Table**

For each subsequent head **i** from **1** to **n - 1**, Hercules considers all possible patterns to eradicate head **i**. He updates **dp[i][v]** by considering all patterns **u** used to eradicate head **i - 1**, and computes:

- The overlap **ov** between patterns **P[i - 1][u]** and **P[i][v]**.
- The total cuts required: **dp[i - 1][u] + (k - ov)**.

He ensures that:

- The overlap is valid (i.e., **ov != -1**).
- The new cuts do not involve eradicated heads.

In code:

```cpp
for(int i = 1; i < n; i++){
    for(int v = 0; v < P[i].size(); v++){
        int best = INT_MAX;
        for(int u = 0; u < P[i-1].size(); u++){
            if(dp[i-1][u] == INT_MAX) continue;
            int ov = overlapCost[i-1][u][v];
            if(ov == -1) continue;
            int cuts = dp[i-1][u] + (k - ov);
            best = min(best, cuts);
        }
        dp[i][v] = best;
    }
}
```

This represents Hercules's meticulous planning to chain his eradication patterns efficiently.

#### **6. Determining the Final Answer**

Finally, Hercules looks for the minimum total cuts required to eradicate all heads:

- **ans = min(dp[n - 1][u])** for all **u** in **P[n - 1]**.

If **ans == INT_MAX**, it means it's impossible to eradicate all heads under the given constraints.

In code:

```cpp
int ans = *min_element(dp[n-1].begin(), dp[n-1].end());
if(ans == INT_MAX) cout << "Impossible!\n";
else cout << ans << "\n";
```

### **Putting It All Together**

Here's the complete, corrected code with comments:

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
  getOverlap:
  Returns the largest t in [0..k] such that
  the suffix of p of length t equals the prefix of q of length t,
  and the new cuts in q (beyond the overlap) do not involve eradicated heads (heads < tillHead).
  Returns -1 if no valid overlap is possible.
*/
int getOverlap(const vector<int> &p, const vector<int> &q, int k, int tillHead) {
    for(int t = k; t >= 0; t--){
        bool match = true;
        // Check overlapping positions (no need to check for eradicated heads here)
        for(int s = 0; s < t; s++){
            if(p[k - t + s] != q[s]){
                match = false;
                break;
            }
        }
        if(!match) continue;
        // Check new cuts in q (positions beyond the overlap)
        for(int s = t; s < k; s++){
            int h_q = q[s];
            if(h_q < tillHead){
                match = false;
                break;
            }
        }
        if(match) return t;
    }
    return -1; // No valid overlap
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T; cin >> T;
    while(T--){
        int n,m; cin >> n >> m;
        int k,d; cin >> k >> d; // d is not directly needed in this solution

        // P[h] = all patterns of length k that eradicate head h
        vector<vector<vector<int>>> P(n);

        for(int i=0; i<m; i++){
            vector<int> pat(k);
            for(int &x : pat) cin >> x;
            int last = pat[k-1];
            // Add pat to P[last].
            if(last >= 0 && last < n){
                P[last].push_back(pat);
            }
        }

        // If any head has no patterns, it's impossible
        bool impossible = false;
        for(int h=0; h<n; h++){
            if(P[h].empty()){
                impossible = true;
                break;
            }
        }
        if(impossible){
            cout << "Impossible!\n";
            continue;
        }

        // Precompute overlaps between patterns of consecutive heads
        vector<vector<vector<int>>> overlapCost(n-1);
        for(int i=0; i+1<n; i++){
            int szA = P[i].size();
            int szB = P[i+1].size();
            overlapCost[i].assign(szA, vector<int>(szB, -1));
            for(int u = 0; u < szA; u++){
                for(int v = 0; v < szB; v++){
                    // Heads 0 to i have been eradicated
                    overlapCost[i][u][v] = getOverlap(P[i][u], P[i+1][v], k, i+1);
                }
            }
        }

        // DP table
        vector<vector<int>> dp(n);
        for(int i=0; i<n; i++){
            dp[i].resize(P[i].size(), INT_MAX);
        }

        // Base case for head 0
        for(int u=0; u<P[0].size(); u++){
            dp[0][u] = k;
        }

        // Fill DP table
        for(int i=1; i<n; i++){
            for(int v=0; v<P[i].size(); v++){
                int best = INT_MAX;
                for(int u=0; u<P[i-1].size(); u++){
                    if(dp[i-1][u] == INT_MAX) continue;
                    int ov = overlapCost[i-1][u][v];
                    if(ov == -1) continue;
                    int cuts = dp[i-1][u] + (k - ov);
                    best = min(best, cuts);
                }
                dp[i][v] = best;
            }
        }

        // Final answer
        int ans = *min_element(dp[n-1].begin(), dp[n-1].end());
        if(ans == INT_MAX) cout << "Impossible!\n";
        else cout << ans << "\n";
    }
    return 0;
}
```

### **An Illustrative Example**

Let's walk through an example to solidify our understanding.

**Input:**

```
1
3 4
2 2
0 0
2 0
1 1
0 2
```

- **n = 3** (heads 0 to 2)
- **m = 4** eradication patterns
- **k = 2**
- Patterns:
  - `[0, 0]` eradicates head 0
  - `[2, 0]` eradicates head 0
  - `[1, 1]` eradicates head 1
  - `[0, 2]` eradicates head 2

**Processing:**

- **P[0]:** `[[0, 0], [2, 0]]`
- **P[1]:** `[[1, 1]]`
- **P[2]:** `[[0, 2]]`

**We need to determine if it's possible to eradicate all heads.**

- **Head 0:** Can be eradicated using patterns in **P[0]**. Let's pick `[0, 0]`.
- **Head 1:** Needs patterns in **P[1]**. It requires cuts `[1, 1]`. Since head 0 has been eradicated, we proceed.
- **Head 2:** Needs pattern `[0, 2]`. It requires cuts on head 0, which has been eradicated.

**Problem:** We cannot perform new cuts on eradicated heads. Therefore, we cannot eradicate head 2 with the available patterns.

**Conclusion:** It's **impossible** to eradicate all heads.

**Output:**

```
Impossible!
```

### **Key Takeaways**

- **Dynamic Programming** is used to find the optimal sequence of cuts while respecting constraints.
- **Overlap of patterns** reduces the number of cuts by reusing previous cuts.
- **Ensuring no cuts on eradicated heads** is crucial to meet the problem's conditions.
- **Precomputing overlaps** efficiently allows us to avoid redundant computations.

**Remember:** Just like Hercules needed both strength and wit to defeat the Hydra, we often need both efficient algorithms and careful consideration of constraints to solve complex problems.

### **Final Thoughts**

By approaching the problem methodically, considering all constraints, and employing dynamic programming, we provide Hercules with a strategic plan to slay the Hydra with minimal effort.

As you tackle similar challenges, think of Hercules's labor as a metaphor:

- **Identify the constraints.**
- **Break the problem into manageable subproblems.**
- **Seek opportunities to optimize, like overlapping patterns.**
- **Ensure your solution adheres strictly to the problem's rules.**

May this tale of Hercules and the Hydra serve as a memorable guide on your programming journey!

**Explanation of Changes:**

- **`getOverlap` Function:**

  - We no longer check for eradicated heads in overlapping positions.
  - We only check that the new cuts (positions beyond the overlap in `q`) do not involve eradicated heads.
  - This allows us to use previous cuts (even on eradicated heads) to form the required patterns.

- **Overlap Computation:**

  - When computing `overlapCost`, we use the updated `getOverlap` function.
  - We ensure that the new cuts added when moving from one pattern to the next do not require cutting eradicated heads.

- **Base Case Initialization:**

  - For head 0, all patterns are valid since no heads have been eradicated yet.
  - We can initialize `dp[0][u] = k` for all patterns of head 0.

- **DP Transition:**

  - In the DP update, we only consider overlaps where `ov != -1`, meaning the overlap is valid and new cuts do not involve eradicated heads.
  - We calculate the total cuts by adding the number of new cuts `(k - ov)` to the previous total.


# Graph exploration

## Iterating over all the out edges

```cpp
#include <boost/graph/adjacency_list.hpp>
#include <iostream>

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
  boost::no_property, boost::property<boost::edge_weight_t, int> > weighted_graph;

typedef boost::graph_traits<weighted_graph>::vertex_descriptor vertex_desc;
typedef boost::graph_traits<weighted_graph>::edge_descriptor edge_desc;
typedef boost::property_map<weighted_graph, boost::edge_weight_t>::type weight_map;

int main() {
    weighted_graph G(4);
    weight_map weights = boost::get(boost::edge_weight, G);

    edge_desc e;
    e = boost::add_edge(0, 1, G).first; weights[e] = 0;
    e = boost::add_edge(1, 2, G).first; weights[e] = 2;
    e = boost::add_edge(2, 3, G).first; weights[e] = 1;
    e = boost::add_edge(3, 0, G).first; weights[e] = 3;
    e = boost::add_edge(0, 2, G).first; weights[e] = 3;

    // Print all vertices connected to vertex 0 (undirected graph)
    vertex_desc v = 0;
    std::cout << "Vertices connected to vertex " << v << " (undirected graph):\n";
    auto out_edges = boost::out_edges(v, G);
    for (auto it = out_edges.first; it != out_edges.second; ++it) {
        vertex_desc target = boost::target(*it, G);
        std::cout << target << "\n";
    }

    return 0;
}
```

# Delaunay triangulation

```cpp

K::FT get_face_radius(Triangulation &t, Triangulation::Face_handle f) {
  if(t.is_infinite(f)) return K::FT(LONG_MAX);
  return Circle(f->vertex(0)->point(),f->vertex(1)->point(),f->vertex(2)->point()).squared_radius();
}

for(auto e = t.finite_edges_begin(); e != t.finite_edges_end(); ++e) {
    auto v_left = e->first->vertex((e->second + 1) % 3);
    auto v_right = e->first->vertex((e->second + 2) % 3); 
    
    Point midpoint = CGAL::midpoint(v_left->point(), v_right->point());
    int v_nearest = t.nearest_vertex(midpoint)->info();
    
    auto f1 = e->first;
    // returns the same edge seen from the other adjacent face.
    auto f2 = t.mirror_edge(*e).first;


// We can compare vertices in the triangulation like this:
 Point midpoint = CGAL::midpoint(v_left->point(), v_right->point());
    int v_nearest = t.nearest_vertex(midpoint)->info();
 if(v_nearest != v_left->info() ...) {

```



``` In particular, we want to be able to carry out the
following two operations efficiently: (1) removing a weight, and (2) finding the heaviest weight
which is still at most s[i].
Fortunately for us, there is a data structure in the C++ STL that can do both of these things
efficiently: std::multiset. Both removing elements and answering queries of the form in (2)
takes O(log N) time in a std::multiset with N elements.
```

# Get max element of vector:

 *std::max_element(occs.begin(), occs.end())

# When searching a value think about order, and binary search if applicable 🧠

# Connected components sizes

```

// Find connected components
    std::vector<int> component_map(n);
    int num_components = boost::connected_components(
        G, boost::make_iterator_property_map(component_map.begin(), boost::get(boost::vertex_index, G)));

    // Count the size of each connected component
    std::vector<int> component_sizes(num_components, 0);
    for (int i = k; i < n; ++i) {
        component_sizes[component_map[i]]++;
    }

    // Find the maximum component size
    int max_size = *std::max_element(component_sizes.begin(), component_sizes.end());

```
