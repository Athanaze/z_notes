///1
#include <iostream>
// BGL includes
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cycle_canceling.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>
#include <boost/graph/successive_shortest_path_nonnegative_weights.hpp>
#include <boost/graph/find_flow_cost.hpp>

// Graph Type with nested interior edge properties for Cost Flow Algorithms
typedef boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS> traits;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, boost::no_property,
    boost::property<boost::edge_capacity_t, long,
        boost::property<boost::edge_residual_capacity_t, long,
            boost::property<boost::edge_reverse_t, traits::edge_descriptor,
                boost::property <boost::edge_weight_t, long> > > > > graph; // new! weightmap corresponds to costs

typedef boost::graph_traits<graph>::edge_descriptor             edge_desc;
typedef boost::graph_traits<graph>::out_edge_iterator           out_edge_it; // Iterator
typedef traits::vertex_descriptor vertex_desc;

// Custom edge adder class
class edge_adder {
 graph &G;

 public:
  explicit edge_adder(graph &G) : G(G) {}
  void add_edge(int from, int to, long capacity, long cost) {
    auto c_map = boost::get(boost::edge_capacity, G);
    auto r_map = boost::get(boost::edge_reverse, G);
    auto w_map = boost::get(boost::edge_weight, G); // new!
    const edge_desc e = boost::add_edge(from, to, G).first;
    const edge_desc rev_e = boost::add_edge(to, from, G).first;
    c_map[e] = capacity;
    c_map[rev_e] = 0; // reverse edge has no capacity!
    r_map[e] = rev_e;
    r_map[rev_e] = e;
    w_map[e] = cost;   // new assign cost
    w_map[rev_e] = -cost;   // new negative cost
  }
};


void solve(){
  int n, m, s, p;
  std::cin >> n >> m >> s >> p;
  
  // n : the total number of location
  graph G(n); 
  edge_adder adder(G);
  
  auto c_map = boost::get(boost::edge_capacity, G);
  auto r_map = boost::get(boost::edge_reverse, G);
  auto rc_map = boost::get(boost::edge_residual_capacity, G);
  
  // m : the total number of flyways
  // s : location denoting 4PD <- THE ENTRY POINT
  // p : location denoting the Portkey <- THE SINK
  // Following m lines describe the flyways
  
  int max_capacity_c = 0;
  for(int flyway = 0; flyway < m; flyway++){
    int u, v, c, d;
    // Flyway between u and v, capacity c, cost d    
    std::cin>> u>> v>> c>>d;
    if(c > max_capacity_c){
      max_capacity_c = c;
    }
    adder.add_edge(u, v, c, d);
    
  }
  int team_size = m*max_capacity_c;
  const vertex_desc harry_source = boost::add_vertex(G);
  const vertex_desc all_source = boost::add_vertex(G);
  
  // 4PD is the entry point
  adder.add_edge(harry_source, s, 1, 0);
  adder.add_edge(all_source, s, team_size, 0);
  
  boost::successive_shortest_path_nonnegative_weights(G, harry_source, p);
  int harry_cost = boost::find_flow_cost(G);
  
  boost::successive_shortest_path_nonnegative_weights(G, all_source, p);
  int cost2 = boost::find_flow_cost(G);
  
  // Iterate over all edges leaving the source to sum up the flow values.
  int s_flow = 0;
  bool found_m = false;
  
  out_edge_it e, eend;
  edge_desc modifiable_edge;
  
  for(boost::tie(e, eend) = boost::out_edges(boost::vertex(all_source,G), G); e != eend; ++e) {
      s_flow += c_map[*e] - rc_map[*e];  
      if(boost::source(*e, G) == all_source && boost::target(*e, G) == s){
        modifiable_edge = *e;
        found_m = true;
      } 
  }
   if(!found_m){
    std::cout << "not found" << std::endl;
  }
  
  c_map[modifiable_edge] = s_flow;
  
  int capa = s_flow;
  std::cout << "Starting with capa : " << capa << std::endl;
  
  
  int left = 1;
  int right = s_flow;
  int mid;
  
  while(left != right){
    
    mid = left + (right -left)/2;
  
    c_map[modifiable_edge] = mid;
    
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // computes with mid
    boost::successive_shortest_path_nonnegative_weights(G, all_source, p);
    long cost2 = boost::find_flow_cost(G) ;
      
    int flow_for_this_iteration = 0;
    for(boost::tie(e, eend) = boost::out_edges(boost::vertex(all_source,G), G); e != eend; ++e) {
      flow_for_this_iteration += c_map[*e] - rc_map[*e];     
    }
    cost2 = cost2 / flow_for_this_iteration;
    ///////////////////////////////////////////////////////////////////////////////////////////////
  
    if(cost2 <= harry_cost){
      // we are too cheap, so the lower bound (left) becomes the current position, and the upper bound stays the same
      left = mid; 
    }
    if(cost2 > harry_cost){
      // we are too expensive, so the lower bound (left) stays the same, and the upper bound is now the current position
      right = mid -1;
    }
  
  }
  std::cout << left << std::endl;
}

int main() {
    int n_tests;
    std::cin >> n_tests;
    for(int t_n = 0; t_n < n_tests; t_n++){
      solve();
    }
    return 0;
}
