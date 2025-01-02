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
