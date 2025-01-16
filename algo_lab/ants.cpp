///1
// STL includes
#include <iostream>
#include <vector>

// BGL includes
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
  boost::no_property, boost::property<boost::edge_weight_t, int> >      weighted_graph;
typedef boost::property_map<weighted_graph, boost::edge_weight_t>::type weight_map;
typedef boost::graph_traits<weighted_graph>::edge_descriptor            edge_desc;
typedef boost::graph_traits<weighted_graph>::vertex_descriptor          vertex_desc;

int dijkstra_dist(const weighted_graph &G, int s, int t) {
  int n = boost::num_vertices(G);
  std::vector<int> dist_map(n);

  boost::dijkstra_shortest_paths(G, s,
    boost::distance_map(boost::make_iterator_property_map(
      dist_map.begin(), boost::get(boost::vertex_index, G))));

  return dist_map[t];
}

int dijkstra_path(const weighted_graph &G, int s, int t, std::vector<vertex_desc> &path) {
  int n = boost::num_vertices(G);
  std::vector<int>         dist_map(n);
  std::vector<vertex_desc> pred_map(n);

  boost::dijkstra_shortest_paths(G, s,
    boost::distance_map(boost::make_iterator_property_map(
      dist_map.begin(), boost::get(boost::vertex_index, G)))
    .predecessor_map(boost::make_iterator_property_map(
      pred_map.begin(), boost::get(boost::vertex_index, G))));

  int cur = t;
  path.clear(); path.push_back(cur);
  while (s != cur) {
    cur = pred_map[cur];
    path.push_back(cur);
  }
  std::reverse(path.begin(), path.end());
  return dist_map[t];
}

void solve(){
  int n, e, s, a, b;
  
  std::cin >> n >> e >> s >> a >> b;
  
  // n : number of node in the forest
  // e : number of edges in the forest
  // s : number of species
  // a,b : start and finish node
  
  // The next e lines describe one edge : t1 t2 w0 w1 ... ws-1
  
  std::vector<weighted_graph> vec_graphs(s);
  std::vector<weight_map> vec_weight_maps;
  
  for(int i =0; i < s; i++){
    vec_weight_maps.push_back(
      boost::get(boost::edge_weight, vec_graphs[i])
    );
  }
  
  for(int line_counter = 0; line_counter < e; line_counter ++){
    int t1, t2;
    
    std::cin >> t1 >> t2;
    
    for(int s_counter = 0; s_counter < s; s_counter++){
      int w_value;
      std::cin >> w_value;
      auto added_edge = boost::add_edge(t1, t2, vec_graphs[s_counter]).first;
      vec_weight_maps[s_counter][added_edge]=w_value;
    }
  }
  std::cout << "Just before reading the hives \n" << std::endl;
  // Compute the private network for each species
  
  weighted_graph all_species_priv_network_graph(n);
  weight_map all_species_priv_network_weights = boost::get(boost::edge_weight, all_species_priv_network_graph);
  
  for(int s_i = 0; s_i < s; s_i++){
    int hive_vertex;
    
    std::cin >> hive_vertex;
    
    std::set<int> explored_vertices;
    explored_vertices.insert(hive_vertex);
    vertex_desc starting_vertex = hive_vertex;
    
    while(explored_vertices.size() < n){
      bool first = true;
      int cheapest_weight;
      vertex_desc cheapest_v;
      auto out_edges = boost::out_edges(starting_vertex, vec_graphs[s_i]);
      
      for(auto it = out_edges.first; it != out_edges.second; ++it){
          
        vertex_desc target = boost::target(*it, vec_graphs[s_i]);
        int edge_w = vec_weight_maps[s_i][*it];
        std::cout << "Edge weight : " << edge_w << std::endl;
        if(first){
          cheapest_v = target;
          cheapest_weight = edge_w;
          first = false;
        }else{
          auto set_it = explored_vertices.find(target);
          if(edge_w < cheapest_weight && set_it == explored_vertices.end()){
            cheapest_v = target;
            cheapest_weight = edge_w;
          }
        }
      }
      
      // The cheapest outgoing edge has been found, we go to the corresponding node
      explored_vertices.insert(cheapest_v);
      
      // Add that edge to the MST
      auto added_edge = boost::add_edge(
        starting_vertex,
        cheapest_v,
        all_species_priv_network_graph
      ).first;
      
      all_species_priv_network_weights[added_edge] = cheapest_weight;
 
      starting_vertex = cheapest_v;
    }
  }
  std::cout << "Dijkstra distance : " << a << " , b: " << b << std::endl;
  std::cout <<  dijkstra_dist(all_species_priv_network_graph, a, b) << std::endl;
  
}


int main()
{
  std::ios_base::sync_with_stdio(false); // Always!
  int t;
  std::cin >> t;
  for(int t_counter = 0; t_counter < t; t_counter++){
    solve();
  }

  return 0;
}
