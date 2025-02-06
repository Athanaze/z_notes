#include <iostream>
#include <vector>
#include <limits>
#include <cstdint>
#include <algorithm>

// Boost graph includes
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/max_cardinality_matching.hpp>
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
  boost::no_property, boost::property<boost::edge_weight_t, int> >      weighted_graph;

// Data‐structure for storing an input road (and later for building the matching graph)
struct Road {
    int u, v;
    int len;
};
 
void testcase(){
  int n, m, b, p;
  int d;
  std::cin >> n >> m >> b >> p >> d;
  
  // Read barrack positions.
  std::vector<int> barracks(b);
  for (int i = 0; i < b; i++){
      std::cin >> barracks[i];
  }

  // Read plaza positions.
  std::vector<bool> isPlaza(n, false);
  for (int i = 0; i < p; i++){
      int qq;
      std::cin >> qq;
      isPlaza[qq] = true;
  }

  // Read roads. Also, we need to keep a list of roads for the later matching graph.
  std::vector<Road> roadList;
  roadList.reserve(m);

  // Build the city graph for Dijkstra. We use a Boost undirected graph,
  // with vertices 0 ... n-1 corresponding to intersections.
  // To enable multi–source Dijkstra we add an extra “dummy” vertex (index n)
  // connected with zero–weight edges to each barrack.

  typedef boost::adjacency_list<
      boost::vecS, boost::vecS, boost::undirectedS,
      boost::no_property,
      boost::property<boost::edge_weight_t, int> > CityGraph;

  // Create graph with n+1 vertices (the extra vertex is the dummy source)
  CityGraph G(n+1);

  // Add each road (as an undirected edge) to the graph.
  // Also record the road in roadList.
  for (int i = 0; i < m; i++){
      int u, v;
      int len;
      std::cin >> u >> v >> len;
      roadList.push_back({u, v, len});
      boost::add_edge(u, v, len, G);
  }

  // Add an edge from the dummy vertex to every barrack vertex with zero cost.
  int dummy = n; // dummy vertex index is n
  for (int src : barracks) {
      boost::add_edge(dummy, src, 0, G);
  }

  // Compute distances using Boost’s dijkstra_shortest_paths.
  
  std::vector<int> dist(boost::num_vertices(G), std::numeric_limits<int>::max());

  boost::dijkstra_shortest_paths(G, dummy,
      boost::distance_map(
          boost::make_iterator_property_map(dist.begin(),
              boost::get(boost::vertex_index, G)
          )
      ).weight_map(boost::get(boost::edge_weight, G))
  );

  // Mark all intersections (0 .. n-1) that can be reached within distance d.
  std::vector<bool> usable(n, false);
  for (int i = 0; i < n; i++){
      if(dist[i] <= d){
        usable[i] = true;
      }
  }

  // Now reduce the problem to a matching problem.
  // Construct a new graph H as follows:
  //  • Each usable non–plaza intersection becomes one vertex.
  //  • Each usable plaza becomes two vertices.
  // (Roads incident to plazas are added twice so that the same physical road
  // may be “secured” from either copy.)

  std::vector<int> regIndex(n, -1);      // mapping for usable non-plaza intersections
  std::vector<int> plazaIndex1(n, -1);     // first copy for usable plazas
  std::vector<int> plazaIndex2(n, -1);     // second copy for usable plazas
  int vertexCount = 0;

  for (int i = 0; i < n; i++){
      if (!usable[i]) continue;

      if (!isPlaza[i]){
          regIndex[i] = vertexCount++;
      } else {
          plazaIndex1[i] = vertexCount++;
          plazaIndex2[i] = vertexCount++;
      }
  }

  // Build the matching graph H.
  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> MatchingGraph;
  MatchingGraph H(vertexCount);

  // For every road that connects two usable intersections, add edges to H.
  // (By hypothesis, roads connecting two plazas never occur.)
  for (const auto &r : roadList){
      int u = r.u, v = r.v;
      if(!usable[u] || !usable[v])
          continue;

      // Skip any road between two plazas (by hypothesis this never happens).
      if(isPlaza[u] && isPlaza[v])
          continue;

      // Case: both endpoints are regular.
      if(!isPlaza[u] && !isPlaza[v]){
          boost::add_edge(regIndex[u], regIndex[v], H);
      }
      // Case: one endpoint is plaza and the other is regular.
      else if(!isPlaza[u] && isPlaza[v]){
          boost::add_edge(regIndex[u], plazaIndex1[v], H);
          boost::add_edge(regIndex[u], plazaIndex2[v], H);
      }
      else if(isPlaza[u] && !isPlaza[v]){
          boost::add_edge(regIndex[v], plazaIndex1[u], H);
          boost::add_edge(regIndex[v], plazaIndex2[u], H);
      }
  }

  // Compute the maximum matching on H.
  // Each edge in the matching corresponds to one secured road.
  typedef boost::graph_traits<MatchingGraph>::vertex_descriptor Vertex;
  std::vector<Vertex> mate(vertexCount);
  boost::edmonds_maximum_cardinality_matching(H, &mate[0]);
  int matchSize = boost::matching_size(H, &mate[0]);

  std::cout << matchSize << "\n";
} 

// Main program
int main(){
    std::ios::sync_with_stdio(false);
    
    int t;
    std::cin >> t;
    while(t--){
        testcase();
    }
 
    return 0;
}
