///1
// STL includes
#include <iostream>
#include <vector>
#include <set>
#include <tuple>

// BGL includes
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>

// Graph Type with nested interior edge properties for flow algorithms
typedef boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS> traits;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, boost::no_property,
    boost::property<boost::edge_capacity_t, long,
        boost::property<boost::edge_residual_capacity_t, long,
            boost::property<boost::edge_reverse_t, traits::edge_descriptor>>>> graph;

typedef traits::vertex_descriptor vertex_desc;
typedef traits::edge_descriptor edge_desc;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
  boost::no_property, boost::property<boost::edge_weight_t, int> >      weighted_graph;
typedef boost::property_map<weighted_graph, boost::edge_weight_t>::type weight_map;
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

void solve_test(){
  int n, m, b, p, d;
  
  /*
  n : # intersections
  m : # roads
  b : # barracks
  p : # plazas
  d : travalable distance 
  */
  std::cin >> n >> m >> b >> p >> d;
  
  std::vector<int> barracks_vec(b, 0);
  
  for(int i = 0; i < b; i++){
    std::cin >> barracks_vec[i];
  }
  
  std::set<int> piazzas_set;
  if(p > 0){
    for(int i = 0; i < p; i++){
      int q_value;
      std::cin >> q_value;
      piazzas_set.insert(q_value);
    }
  }
  
  weighted_graph G(n);
  weight_map weights = boost::get(boost::edge_weight, G);
  edge_desc e;
  std::vector<std::tuple<int, int>> xy_vec(m, {0,0});
  // following m lines describe the roads
  for(int i = 0; i < m; i++){
    int x, y, l;
    
    std::cin >> x >> y >> l;
    xy_vec[i] = {x, y};
    e = boost::add_edge(x, y, G).first; weights[e]=l;
    
  }
  
  // Will contains all vertices that should receive flow
  std::set<int> within_distance;
  
  // For each barrack, run dijkstra
  
  for(int barrack_source: barracks_vec){
    std::vector<int> dist_map(n);

    boost::dijkstra_shortest_paths(G, barrack_source,
      boost::distance_map(boost::make_iterator_property_map(
        dist_map.begin(), boost::get(boost::vertex_index, G))));
    
    for(int i = 0; i < n; i++){
      if(dist_map[i] <= d){
        within_distance.insert(i);
      }
    }
  }
  
  //std::cout << "Size of within_distance set : " << within_distance.size() << std::endl;
  
  // Graph for max flow
  // # nodes = # roads + # intersections + (2 (super sink and source) added later)
  graph G2(n+m+2);
  // < all the intersections vertices >< one node per road>
  edge_adder adder(G2);
  // Add special vertices source and sink
  const vertex_desc v_source = boost::add_vertex(G2);
  const vertex_desc v_sink = boost::add_vertex(G2);
  
  // Create three node per road : starting intersection (x), node for the road, ending intersection (y)
  auto road_index = n;
  for(auto xy_tuple: xy_vec){
    
    adder.add_edge(std::get<0>(xy_tuple) ,road_index , 1);
    adder.add_edge(std::get<1>(xy_tuple) ,road_index , 1);
    
    // While we are are here, connect the road node to the super sink with capa 2
    adder.add_edge(road_index, v_sink, 2);
    road_index++;
  }
  int counter_2 = 0;
  int counter_1 = 0;
  for(int fillable_vertex: within_distance){
    if(piazzas_set.find(fillable_vertex) == piazzas_set.end()){
      // The fillable vertex is not a piazzas, so capacity 1
      adder.add_edge(v_source, fillable_vertex, 1);
      counter_1++;
    }else{
       adder.add_edge(v_source, fillable_vertex, 2);
       counter_2++;
    }
  }
  
  //std::cout << "Counter 2 : " << counter_2 << " p : " << p << " Counter 1 : " << counter_1 << " n-p:" << n-p << std::endl;
  
  // Calculate flow from source to sink
  // The flow algorithm uses the interior properties (managed in the edge adder)
  // - edge_capacity, edge_reverse (read access),
  // - edge_residual_capacity (read and write access).
  long flow = boost::push_relabel_max_flow(G2, v_source, v_sink);
  //std::cout << "The total flow is " << flow << "\n";

  // Retrieve the capacity map and reverse capacity map
  const auto c_map = boost::get(boost::edge_capacity, G2);
  const auto rc_map = boost::get(boost::edge_residual_capacity, G2);

  // Iterate over all the edges to print the flow along them
  auto edge_iters = boost::edges(G2);
  int n_safe_roads = 0;
  
  for (auto edge_it = edge_iters.first; edge_it != edge_iters.second; ++edge_it) {
    const edge_desc edge = *edge_it;
    const long flow_through_edge = c_map[edge] - rc_map[edge];
    
    if(boost::target(edge, G2) == v_sink && flow_through_edge == 2){
      n_safe_roads++;
    }
    /*
    std::cout << "edge from " << boost::source(edge, G) << " to " << boost::target(edge, G)
              << " runs " << flow_through_edge
              << " units of flow (negative for reverse direction). \n";
    */
  }
  
  std::cout << n_safe_roads <<std::endl;
  
}

int main()
{
  
  int n_tests;
  std::cin >> n_tests;
  
  for(int t = 0; t < n_tests; t++){
    solve_test();
  }
  return 0;
}
