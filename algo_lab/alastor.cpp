#include <vector>
#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
    boost::no_property,
    boost::property<boost::edge_weight_t, long>> Graph;
typedef boost::graph_traits<Graph>::vertex_descriptor vertex_desc;
typedef boost::graph_traits<Graph>::edge_descriptor edge_desc;

typedef boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS> traits;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, boost::no_property,
    boost::property<boost::edge_capacity_t, long,
        boost::property<boost::edge_residual_capacity_t, long,
            boost::property<boost::edge_reverse_t, traits::edge_descriptor>>>> FlowGraph;

class edge_adder {
    FlowGraph &G;
public:
    explicit edge_adder(FlowGraph &G) : G(G) {}
    void add_edge(int from, int to, long capacity) {
        auto c_map = boost::get(boost::edge_capacity, G);
        auto r_map = boost::get(boost::edge_reverse, G);
        const auto e = boost::add_edge(from, to, G).first;
        const auto rev_e = boost::add_edge(to, from, G).first;
        c_map[e] = capacity;
        c_map[rev_e] = 0;
        r_map[e] = rev_e;
        r_map[rev_e] = e;
    }
};

void solve() {
    int n, m, s, p;
    std::cin >> n >> m >> s >> p;

    std::vector<std::tuple<int, int, long, long>> edges; // u, v, c, d
    for (int i = 0; i < m; ++i) {
        int u, v, c, d;
        std::cin >> u >> v >> c >> d;
        edges.emplace_back(u, v, c, d);
        edges.emplace_back(v, u, c, d);
    }

    // Build the original graph for Dijkstra
    Graph g(n);
    auto weight_map = boost::get(boost::edge_weight, g);
    for (const auto &e : edges) {
        int u = std::get<0>(e);
        int v = std::get<1>(e);
        long d = std::get<3>(e);
        auto edge = boost::add_edge(u, v, g).first;
        weight_map[edge] = d;
    }

    // Compute Harry's cost (shortest path from s to p)
    std::vector<long> forward_dist(n, std::numeric_limits<long>::max());
    forward_dist[s] = 0;
    boost::dijkstra_shortest_paths(g, s, boost::distance_map(boost::make_iterator_property_map(
        forward_dist.begin(), boost::get(boost::vertex_index, g))));

    long harry_cost = forward_dist[p];
    if (harry_cost == std::numeric_limits<long>::max()) {
        std::cerr << "No path found!" << std::endl;
        return;
    }

    // Compute reverse distances (distance from each node to p)
    // Build reversed graph
    Graph reversed_g(n);
    auto reversed_weight_map = boost::get(boost::edge_weight, reversed_g);
    for (const auto &e : edges) {
        
        int u = std::get<0>(e);
        int v = std::get<1>(e);
        
        //int u = std::get<1>(e); // reverse the edge direction
        //int v = std::get<0>(e);
        long d = std::get<3>(e);
        auto edge = boost::add_edge(u, v, reversed_g).first;
        reversed_weight_map[edge] = d;
    }

    std::vector<long> reverse_dist(n, std::numeric_limits<long>::max());
    reverse_dist[p] = 0;
    boost::dijkstra_shortest_paths(reversed_g, p, boost::distance_map(boost::make_iterator_property_map(
        reverse_dist.begin(), boost::get(boost::vertex_index, reversed_g))));

    // Build the subgraph for max flow
    FlowGraph sub(n);
    edge_adder adder_sub(sub);

    for (const auto &e : edges) {
        int u = std::get<0>(e);
        int v = std::get<1>(e);
        long c = std::get<2>(e);
        long d = std::get<3>(e);

        if (forward_dist[u] != std::numeric_limits<long>::max() &&
            reverse_dist[v] != std::numeric_limits<long>::max() &&
            (forward_dist[u] + d + reverse_dist[v] == harry_cost)) {
            adder_sub.add_edge(u, v, c);
        }
    }

    // Compute max flow
    long max_flow = boost::push_relabel_max_flow(sub, s, p);
    std::cout << max_flow << std::endl;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    int t;
    std::cin >> t;
    while (t--) solve();
    return 0;
}
