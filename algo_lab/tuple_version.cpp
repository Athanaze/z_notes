#include <vector>
#include <iostream>
#include <tuple>
#include <limits>

typedef std::tuple<int, int, int> T3;

T3 explore(int vertex_index, std::vector<int> &vec_costs, std::vector< std::vector<int> > &tree){
  
  int picked = 0, covered = 0, not_covered = 0;
  
  int min_diff_cost_covered = std::numeric_limits<int>::max();
  
  for(int i = 0; i < tree[vertex_index].size(); i++){
    
    auto subtree_result = explore(tree[vertex_index][i], vec_costs, tree);
    
    picked    +=   std::get<0>(subtree_result);
    covered   +=   std::get<1>(subtree_result);
    not_covered += std::get<2>(subtree_result);
    
    min_diff_cost_covered = std::min(
      min_diff_cost_covered,
      std::get<0>(subtree_result) - std::get<1>(subtree_result) 
    );
  }
  
  int picked_result = vec_costs[vertex_index] + not_covered;
  int covered_result = std::min(picked_result, covered+min_diff_cost_covered);
  int not_covered_result = std::min(picked_result, covered);
  
  return {picked_result, covered_result, not_covered_result};
  
}

void solve(){
  int n;
  
  std::cin >> n;
  
  std::vector< std::vector<int> > tree(n);
  
  for(int i = 0; i < n-1; i++){
    
    int from_i;
    int to_j;
    
    std::cin >> from_i >> to_j;
    
    tree[from_i].push_back(to_j);
    
   
  }
  
  std::vector<int> vec_costs(n, 0);
  for(int i = 0; i < n; i++){
    std::cin >> vec_costs[i];
  }
  auto res = explore(0, vec_costs, tree);
  std::cout << std::get<1>(res) << std::endl;
}

int main(){
  std::ios_base::sync_with_stdio(false);
  int t0;
  std::cin >> t0;
  
  for(int t =0; t < t0; t++){
    solve();
  }
  return 0;
}
