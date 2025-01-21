///1

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_face_base_2.h>
#include <boost/pending/disjoint_sets.hpp>
#include <vector>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <limits>

// we want to store an index with each vertex
typedef std::size_t                                            Index;
typedef std::tuple<Index,Index,int, Index> Edge;
typedef std::vector<Edge> EdgeV;

void test_case(){
  Index n, tatooine_index;
  
  std::cin >> n >> tatooine_index;
  tatooine_index -= 1; // we do 0 based indexing
  EdgeV edges;
  
  int edge_counter = 0;
  // The following n − 1 lines describe the costs for building transmission channels
  for(Index j = 1; j <= n-1; j++){
    // he j-th line, for j ∈ {1, . . . , n − 1}, contains n − j integer numbers between 1 and 220
    for(Index k  = 1; k <= n-j; k++){
      int cost_for_this_link;
      std::cin >> cost_for_this_link; // k-th number : the cost for building a transmission channel between planets j and j + k
      
      // FOR SIMPLIFICATION WE USE 0 BASED INDEXING WHEN STORING
      edges.push_back({j-1, j+k-1, cost_for_this_link, edge_counter});
      edge_counter++;
    }
  }
  
  std::sort(edges.begin(), edges.end(),
      [](const Edge& e1, const Edge& e2) -> bool {
        auto e1_2 = std::get<2>(e1);
        auto e2_2 = std::get<2>(e2); 
        if(e1_2 == e2_2){
          return std::get<3>(e1) < std::get<3>(e2);
        }
        return e1_2 < e2_2;
  });
       
  // Compute EMST using Kruskal's algorithm. This step takes O(n alpha(n)) time
  // in theory; for all practical purposes alpha(n) is constant, so linear time.
  
  EdgeV picked_edges;
  
  // setup and initialize union-find data structure
  boost::disjoint_sets_with_storage<> uf(n);
  Index n_components = n;
  
  bool processed_t = false;
  for (EdgeV::const_iterator e = edges.begin(); e != edges.end(); ++e) {
      
      if(!processed_t){
        auto g0e = std::get<0>(*e);
        auto g1e = std::get<1>(*e);
        if(g0e == tatooine_index || g1e == tatooine_index){
          processed_t = true;
          Index c1 = uf.find_set(g0e);
          Index c2 = uf.find_set(g1e);
          if (c1 != c2 ) {
            // this edge connects two different components => part of the emst
            uf.link(c1, c2);
            picked_edges.push_back(*e);
            
            if (--n_components == 1) break;
          }
        }
      }
  }
  

  for (EdgeV::const_iterator e = edges.begin(); e != edges.end(); ++e) {

      auto g0e = std::get<0>(*e);
      auto g1e = std::get<1>(*e);
      
      Index c1 = uf.find_set(g0e);
      Index c2 = uf.find_set(g1e);
      if (c1 != c2 ) {
        // this edge connects two different components => part of the emst
        uf.link(c1, c2);
        picked_edges.push_back(*e);
        
        if (--n_components == 1) break;
      }
  }

  
  // Now we build alternative mst each time excluding one edge
  
  int min_cost_mst = std::numeric_limits<int>::max();
  
  for(auto edge_to_exclude : picked_edges){
     int cost_mst = 0;
     EdgeV run_edges;
    Index i1 = std::get<0>(edge_to_exclude);
    Index i2 = std::get<1>(edge_to_exclude);
    
    // setup and initialize union-find data structure
    boost::disjoint_sets_with_storage<> uf2(n);
    Index n_components2 = n;
    // ... and process edges in order of increasing length
    
    processed_t = false;
    for (EdgeV::const_iterator e = edges.begin(); e != edges.end(); ++e) {
        
        if(!processed_t){
          auto g0e = std::get<0>(*e);
          auto g1e = std::get<1>(*e);
          if(g0e == tatooine_index || g1e == tatooine_index){
            processed_t = true;
            Index c1 = uf2.find_set(g0e);
            Index c2 = uf2.find_set(g1e);
            if (c1 != c2 ) {
              // this edge connects two different components => part of the emst
              uf2.link(c1, c2);
              cost_mst += std::get<2>(*e);
              if (--n_components2 == 1) break;
            }
          }
        }
    }
    
    
    
    for (EdgeV::const_iterator e = edges.begin(); e != edges.end(); ++e) {
      // determine components of endpoints
      Index c1 = uf2.find_set(std::get<0>(*e));
      Index c2 = uf2.find_set(std::get<1>(*e));
      
      bool process_this_edge = true;
      if(i1 == c1){
        if(i2 == c2){
          process_this_edge = false;
        }
      }
      if(i1 == c2){
        if(i2 == c1){
          process_this_edge = false;
        }
      }
      
      if (c1 != c2 && process_this_edge) {
        // this edge connects two different components => part of the emst
        uf2.link(c1, c2);
        cost_mst += std::get<2>(*e);
        
        if (--n_components2 == 1) break;
      }
    }
    if (cost_mst < min_cost_mst){
      min_cost_mst = cost_mst;
    }
  }
  
  
  std::cout << min_cost_mst << std::endl;
  
}

int main() 
{
  std::ios_base::sync_with_stdio(false);
  int t;
  std::cin >> t;
  for(int i = 0; i < t; i++){
    test_case();
  }
  return 0;
}
