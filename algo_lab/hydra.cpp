// STL includes
#include <iostream>
#include <vector>

// BGL includes
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
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

void solve_test(){
  int n, m;
  
  std::cin >> n >> m;
  
  // n : number of hydra's head
  // m : number of eradication patterns
  
  int k,d;
  
  std::cin >> k >> d;
  // k : length of eradiction pattern
  // d: max number of times a head may appear in the list of eradication patterns
  // For each head, list of 
  
  std::vector<std::vector<std::vector<int>>> heads_pattern(n, std::vector<std::vector<int>>(d, std::vector<int>(k, -1)));

  // The following m lines define erdication patterns
  std::vector<int> pattern_index(n, 0);
  
  for(int m_counter = 0; m_counter < m; m_counter++){
    
    std::vector<int> pattern_vec(k, 0);
    for(int k_counter = 0; k_counter < k; k_counter++){
      std::cin >> pattern_vec[k_counter];
    }
    
    // last element
    auto head_number = pattern_vec[k-1];
    
    heads_pattern[head_number][pattern_index[head_number]] = pattern_vec;
    
    pattern_index[head_number]++;
    
  }
  
  weighted_graph G((n*d) + 2);
  weight_map weights = boost::get(boost::edge_weight, G);

  edge_desc e;
  
  // Add all the initial edges, from node 0 to all the possible patterns to eradicate head 0
  
  int super_source = n*d;
  int super_sink = (n*d)+1;
  
  for(int s_d = 0; s_d < d; s_d++){
    e = boost::add_edge(super_source, s_d, G).first; weights[e]=k;
  }
  
  for(int s_d = 0; s_d < d; s_d++){
    auto current_pattern_index = (n-1)*d + s_d;
    e = boost::add_edge(current_pattern_index, super_sink, G).first; weights[e]=0;
  }
  
  //std::cout << "After filling3" << std::endl;
  
  for(int nh =1; nh < n; nh++){
    
    for(int s_d = 0; s_d < d; s_d++){
        // Connect each element of the previous layer to that pattern
        auto current_pattern_index = (nh)*d + s_d;
        // Loop over each pattern of the previous layer
        
        for(int prev_d = 0; prev_d < d; prev_d++){
          auto prev_pattern = heads_pattern[nh-1][prev_d];
          //std::cout << "After prev_pattern, sd:"<<s_d << std::endl;
          
          if(prev_pattern[0] != -1 && heads_pattern[nh][s_d][0] != -1){
            // Each layer has d vertex
            auto vertex_index = (nh-1)*d + prev_d;
            
            // Compute weight
            int sliding_amount = 0;
            
            bool valid_overlap = true;
            bool repeat = true;
            while(repeat){
              for(int sk =0; sk < k; sk++){
                if(sk+sliding_amount < k){
                  if(prev_pattern[sk+sliding_amount] != heads_pattern[nh][s_d][sk]){
                    valid_overlap = false;
                  }
                }
              }
              if(!valid_overlap){
                valid_overlap = true;
                sliding_amount++;
              }else{
                repeat = false;
              }
              
            }

            
            bool ok_to_add = true;
            // Before adding, check that the current pattern does not need to cut again a head that has been eradicated
            
            for(int sk = sliding_amount; sk < k; sk++){
              if(heads_pattern[nh][s_d][sk] < nh-1){
                ok_to_add = false;
              }
            }
            if(sliding_amount == k){
              for(int sk =0; sk < k; sk++){
                if(heads_pattern[nh][s_d][sk] < nh){
                  ok_to_add = false;
                }
              }
            }
            if(ok_to_add){
               e = boost::add_edge(vertex_index, current_pattern_index, G).first; weights[e]=sliding_amount;
            }
           
          }
        }
    }
  }
  
  std::vector<vertex_desc> path;
  int r = dijkstra_dist(G, super_source, super_sink);
  if(r == 2147483647 ){
    std::cout << "Impossible!"  << "\n";
  }else{
    
    std::cout << r << "\n";
  }
}

int main()
{
  int n_tests;
  
  std::cin >> n_tests;
  for(int ti = 0; ti < n_tests; ti++){
    solve_test();
  }
  

  return 0;
}
