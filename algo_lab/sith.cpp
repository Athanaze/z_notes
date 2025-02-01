#include <limits>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <algorithm>
#include <vector>
#include <tuple>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_face_base_2.h>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef std::size_t Index;
typedef CGAL::Triangulation_vertex_base_with_info_2<Index, K> Vb;
typedef CGAL::Triangulation_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds> Delaunay;

typedef K::Point_2 P;
typedef std::pair<P, Index> IPoint;

// Boost Graph Library Definitions
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor VertexDesc;

long n, r;

int rebels(std::vector<IPoint>& points, const int k) {
    // Create a Delaunay triangulation
    Delaunay t;
    t.insert(points.begin() + k, points.end());

    // Create a graph
    Graph G(n);

    // Add edges to the graph based on distance
    for (auto e = t.finite_edges_begin(); e != t.finite_edges_end(); ++e) {
        Index i1 = e->first->vertex((e->second + 1) % 3)->info();
        Index i2 = e->first->vertex((e->second + 2) % 3)->info();

        if (t.segment(e).squared_length() <= r * r) {
            boost::add_edge(i1, i2, G);
        }
    }

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
    return std::min(k, max_size);
}

void testcase() {
    std::cin >> n >> r;

    std::vector<IPoint> points(n);
    for (Index i = 0; i < n; ++i) {
        int x, y;
        std::cin >> x >> y;
        points[i] = {P(x, y), i};
    }

    // Binary search over the number of rebels
    int start = 1, end = (n / 2) + 1;
    while (start < end) {
        int mid = start + (end - start) / 2;
        int r1 = rebels(points, mid), r2 = rebels(points, mid + 1);

        if (r1 < r2)
            start = mid + 1;
        else
            end = mid;
    }

    std::cout << start << std::endl;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int t;
    std::cin >> t;
    for (int i = 0; i < t; ++i) {
        testcase();
    }

    return 0;
}
