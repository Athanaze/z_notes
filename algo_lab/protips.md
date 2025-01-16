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
