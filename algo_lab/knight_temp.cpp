// Algolab BGL Tutorial 2 (Max flow, by taubnert@ethz.ch)
// Flow example demonstrating how to use push_relabel_max_flow using a custom
// edge adder to manage the interior graph properties required for flow
// algorithms
#include <iostream>

// BGL include
#include <boost/graph/adjacency_list.hpp>

// BGL flow include *NEW*
#include <boost/graph/push_relabel_max_flow.hpp>

// Graph Type with nested interior edge properties for flow algorithms
typedef boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS>
    traits;
typedef boost::adjacency_list<
    boost::vecS, boost::vecS, boost::directedS, boost::no_property,
    boost::property<boost::edge_capacity_t, long,
                    boost::property<boost::edge_residual_capacity_t, long,
                                    boost::property<boost::edge_reverse_t,
                                                    traits::edge_descriptor>>>>
    graph;

typedef traits::vertex_descriptor vertex_desc;
typedef traits::edge_descriptor edge_desc;

// Custom edge adder class, highly recommended
class edge_adder {
  graph &G;

public:
  explicit edge_adder(graph &G) : G(G) {}

  void add_edge(int from, int to, long capacity) {
    auto c_map = boost::get(boost::edge_capacity, G);
    auto r_map = boost::get(boost::edge_reverse, G);
    const auto e = boost::add_edge(from, to, G).first;
    const auto rev_e = boost::add_edge(to, from, G).first;
    c_map[e] = capacity;
    c_map[rev_e] = 0; // reverse edge has no capacity!
    r_map[e] = rev_e;
    r_map[rev_e] = e;
  }
};

#define FROM_2D_TO_1D(x, y, x_size) y *x_size + x
#define add_bi_edge(a,b, C) adder.add_edge(a, b, C);  adder.add_edge(b, a, C)
int main() {

  int n_test;
  std::cin >> n_test;

  while (n_test--) {
    int m, n, k, C;
    std::cin >> m >> n >> k >> C;

    auto grid_size_x = 3 * m + 2;
    auto grid_size_y = 3 * n + 2;

    std::vector<std::pair<int, int>> knight_vec(k);
    for (int i = 0; i < k; i++) {
      int knight_x, knight_y;
      std::cin >> knight_x >> knight_y;
      knight_vec[i] = {knight_x, knight_y};
    }

    // Build the graph
    graph G(grid_size_x * grid_size_y);

    edge_adder adder(G);

    // Adds vertical edges
    for (int x = 0; x < grid_size_x; x++) {
      for (int y = 0; y < grid_size_y; y++) {
        if ((x + 1) % 3 == 0) {

          if ((y) % 3 == 0) {
            // Add edge with capacity 1, straight down
            add_bi_edge(FROM_2D_TO_1D(x, y, grid_size_x),
                           FROM_2D_TO_1D(x, y + 1, grid_size_x),
                           1); // from, to, capacity
          }
          if ((y) % 3 == 1) {
            // Add edge with capacity C, straight down
           add_bi_edge(FROM_2D_TO_1D(x, y, grid_size_x),
                           FROM_2D_TO_1D(x, y + 1, grid_size_x), C);
          }
          if ((y) % 3 == 2) {
            // Add edge with capacity C, straight down
            add_bi_edge(FROM_2D_TO_1D(x, y, grid_size_x),
                           FROM_2D_TO_1D(x, y + 1, grid_size_x), C);
          }
        }
      }
    }
    // Adds horizotal edges
    for (int y = 0; y < grid_size_y; y++) {
      for (int x = 0; x < grid_size_x; x++) {
        if ((y + 1) % 3 == 0) {

          if ((x) % 3 == 0) {
            // Add edge with capacity 1, to the right
            add_bi_edge(FROM_2D_TO_1D(x, y, grid_size_x),
                           FROM_2D_TO_1D(x + 1, y, grid_size_x), 1);

                          
          }
          if ((x) % 3 == 1) {
            // Add edge with capacity C, to the right
            add_bi_edge(FROM_2D_TO_1D(x, y, grid_size_x),
                           FROM_2D_TO_1D(x + 1, y, grid_size_x), C);

                           
          }
          if ((x) % 3 == 2) {
            // Add edge with capacity C, to the right
            add_bi_edge(FROM_2D_TO_1D(x, y, grid_size_x),
                           FROM_2D_TO_1D(x + 1, y, grid_size_x), C);
          }
        }
      }
    }

    const vertex_desc v_source = boost::add_vertex(G);
    const vertex_desc v_sink = boost::add_vertex(G);

    // Connects all the vertices at the border to the sink, vertex 0,0
    auto sink_vertex = 0;
    for (int x = 0; x < grid_size_x; x++) {
      for (int y = 0; y < grid_size_y; y++) {
        // Somewhere on the border
        if (x == 0 || x == grid_size_x - 1 || y == 0 || y == grid_size_y - 1) {
          add_bi_edge(v_source, FROM_2D_TO_1D(x, y, grid_size_x), k + 1);
        }
      }
    }

    // Connects all the knights to the source, vertex 0,1
    auto source_vertex = FROM_2D_TO_1D(0, 1, grid_size_x);

    for (auto knight : knight_vec) {
      add_bi_edge(FROM_2D_TO_1D(knight.first * 3 + 2, knight.first * 3 + 2,
                                   grid_size_x),
                     v_sink, k + 1);
    }

    // Run max flow
    auto max_flow = 0;

    long flow = boost::push_relabel_max_flow(G, v_source, v_sink);
    //std::cout << "The total flow is " << flow << "\n";

    // Retrieve the capacity map and reverse capacity map
    const auto c_map = boost::get(boost::edge_capacity, G);
    const auto rc_map = boost::get(boost::edge_residual_capacity, G);

    // Iterate over all the edges to print the flow along them
    auto edge_iters = boost::edges(G);
    for (auto edge_it = edge_iters.first; edge_it != edge_iters.second;
         ++edge_it) {
      const edge_desc edge = *edge_it;
      const long flow_through_edge = c_map[edge] - rc_map[edge];

      if (flow_through_edge > max_flow) {
        max_flow = flow_through_edge;
      }
    }
    //std::cout << "k : " << k<< std::endl;
    std::cout << std::min(k, max_flow)<< std::endl;
  }

  return 0;
}
