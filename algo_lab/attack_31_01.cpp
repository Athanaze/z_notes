///4
// STL includes
#include <iostream>
#include <vector>
#include <set>
#include <tuple>
#include <queue>
#include <limits>
#include <map>

// BGL includes
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/max_cardinality_matching.hpp>

// Typedefs for road network
typedef boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::undirectedS> RoadTraits;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
    boost::no_property, boost::property<boost::edge_weight_t, long> > road_graph;
typedef boost::property_map<road_graph, boost::edge_weight_t>::type weight_map;
typedef RoadTraits::vertex_descriptor road_vertex_desc;
typedef RoadTraits::edge_descriptor road_edge_desc;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_mm;
typedef boost::graph_traits<graph_mm>::vertex_descriptor                       vertex_desc;

int maximum_matching(const graph_mm &G, std::set<int> &p_set, const std::vector<std::pair<int, int> > &roads_vec) {
  
  int n = boost::num_vertices(G);
  std::vector<vertex_desc> mate_map(n);  // exterior property map
  const vertex_desc NULL_VERTEX = boost::graph_traits<graph_mm>::null_vertex();

  boost::edmonds_maximum_cardinality_matching(G,
    boost::make_iterator_property_map(mate_map.begin(), boost::get(boost::vertex_index, G)));
  int matching_size = boost::matching_size(G,
    boost::make_iterator_property_map(mate_map.begin(), boost::get(boost::vertex_index, G)));
  
  
  std::set<int> free_vertices;
  
  for (int i = 0; i < n; ++i) {
    // mate_map[i] != NULL_VERTEX: the vertex is matched
    if(mate_map[i] == NULL_VERTEX){
       free_vertices.insert(i);
    }
   
  }
  
  // Loop over all roads and try to add more when we can
 auto n_degen_matches = 0;
  for(auto xy_elem : roads_vec){
    if(xy_elem.first != xy_elem.second){
          auto it_first = p_set.find(xy_elem.first);
    auto it_second = p_set.find(xy_elem.second);
    
    auto f_first = free_vertices.find(xy_elem.first);
    auto f_second = free_vertices.find(xy_elem.second);
    
    
    if( f_first != free_vertices.end() && f_second != free_vertices.end()){

      n_degen_matches++;
       //matching_size++;
    }
    
    // First is plaza : if other is free, we can connect
    if(it_first != p_set.end() && f_second != free_vertices.end() ){
      p_set.erase(it_first); 
      free_vertices.erase(f_second); 
      matching_size++;
    }
    
    if(it_second != p_set.end() && f_first != free_vertices.end() ){
      p_set.erase(it_second); 
      free_vertices.erase(f_first); 
      matching_size++;
    }
    }

    
  }
  std::cout << "should n_degen_matches happen lol" << n_degen_matches<< std::endl;
  /*
  for (int i = 0; i < n; ++i) {
    // mate_map[i] != NULL_VERTEX: the vertex is matched
    
    mate_map[i] 
    
    // i < mate_map[i]: visit each edge in the matching only once
    if (mate_map[i] != NULL_VERTEX && i < mate_map[i]) std::cout << i << " " << mate_map[i] << "\n";
  }*/
  
  return matching_size;
  /*for (int i = 0; i < n; ++i) {
    // mate_map[i] != NULL_VERTEX: the vertex is matched
    // i < mate_map[i]: visit each edge in the matching only once
    if (mate_map[i] != NULL_VERTEX && i < mate_map[i]) std::cout << i << " " << mate_map[i] << "\n";
  }*/
}


void solve_test() {
    int n, m, b, p;
    long d;

    std::cin >> n >> m >> b >> p >> d;

    std::vector<int> barracks_vec(b, 0);
    for (int i = 0; i < b; i++) {
        std::cin >> barracks_vec[i];
    }
    std::set<int> plaza_set;
    if(p > 0){
      for(int p_counter =0; p_counter < p; p_counter++){
        int p_index;
        std::cin >> p_index;
      
        plaza_set.insert(p_index);
      }
      
    }
    // Build the road network graph 'G_road', undirected
    road_graph G_road(n + 1);
    weight_map weights = boost::get(boost::edge_weight, G_road);
    road_edge_desc e_road;
    std::vector<std::pair<int, int> > xy_vec;
    // Following m lines describe the roads
    for (int i = 0; i < m; i++) {
        int x, y;
        long l;
        std::cin >> x >> y >> l;
        xy_vec.push_back(std::make_pair(x, y));
        e_road = boost::add_edge(x, y, G_road).first; weights[e_road] = l;
    }
    
    

    // Will contain all vertices that should receive flow
    std::set<int> within_distance;

    // Add a super source vertex connected to all barracks with zero-weight edges
    int S = n; // Index of the super source
    for (int barrack_source : barracks_vec) {
        e_road = boost::add_edge(S, barrack_source, G_road).first; weights[e_road] = 0;
    }

    // Run Dijkstra's algorithm from the super source
    std::vector<long> dist_map(n + 1, std::numeric_limits<long>::max());
    boost::dijkstra_shortest_paths(G_road, S,
        boost::distance_map(boost::make_iterator_property_map(
            dist_map.begin(), boost::get(boost::vertex_index, G_road))));

    // Build within_distance set
    for (int i = 0; i < n; i++) {
        if (dist_map[i] <= d) {
            within_distance.insert(i);
        }
    }
    
    int n_removed = 0;
    std::set<int> clean_set;
    
    // only keep the plazas that are within distance
    for(int sacha_pit: plaza_set){
      if(within_distance.find(sacha_pit) != within_distance.end()){
        clean_set.insert(sacha_pit);
      }
      
        
    }
    
    
    
    // Build a new graph with all the routes that have both endpoints
    graph_mm G_mm(n);
    
    for(auto pa: xy_vec){
      auto from_v = pa.first;
      auto to_v = pa.second;
      auto inm = within_distance.end();
      if(within_distance.find(from_v) != inm && within_distance.find(to_v) != inm){
        boost::add_edge(from_v, to_v, G_mm);
      }
    }
    std::cout << maximum_matching(G_mm, clean_set, xy_vec) << std::endl;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    int n_tests;
    std::cin >> n_tests;

    for (int t = 0; t < n_tests; t++) {
        
          solve_test();
        
        
    }
    return 0;
}
