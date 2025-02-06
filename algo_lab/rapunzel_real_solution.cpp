#include <vector>
#include <iostream> // We will use C++ input/output via streams
#include <set>


void dfs(int node, int m, int k, std::vector<int> &brightness_vec, std::vector<bool> &valid, std::vector<std::vector<int>> &strands, std::vector<int> &path, std::multiset<int> &window){
  
  path.push_back(node);
  window.insert(brightness_vec[node]);
  
  int L = path.size();
  
  bool el_deleted = false;
  int brightness_to_remove;
  if(L > m){
    brightness_to_remove = brightness_vec[path[(L-1)-m]];
    auto it = window.find(brightness_to_remove); 
    window.erase(it); 
    
    el_deleted = true;

  }
  
 // Check if current window of m nodes is valid
  if(L >= m){
      int currMin = *window.begin();
      int currMax = *window.rbegin();
      if(currMax - currMin <= k){
          valid[path[L - m]] = true;
      }
  }
  
  for(int child : strands[node]){
    dfs(child, m, k, brightness_vec, valid, strands, path, window);
  }
  
  // Cleanup : remove last element of path, if a brightness was removed put it back in
  path.pop_back();
  
  auto it = window.find(brightness_vec[node]); 
  window.erase(it); 
  if(el_deleted){
    window.insert(brightness_to_remove);
  }
  
}

void testcase() {
  int n, m, k; 
  std::cin >> n >> m >> k; // Read the number of integers to follow
  
  std::vector<int> brightness_vec(n);
  std::vector<bool> valid(n, false);
  std::vector<std::vector<int>> strands(n);
  
  for(int i = 0; i < n; i++){
    int b;
    std::cin >> b;
    brightness_vec[i] = b;
  }
  
  for(int i = 0; i < n-1; i++){
    int u, v;
    std::cin >> u >> v;
    
    strands[u].push_back(v);
    
  }
  
  
  std::multiset<int> window;
  std::vector<int> path;
  
  dfs(0, m, k, brightness_vec, valid, strands, path, window);
  
  
  bool at_least_one = false;
  
  for(int i = 0; i < n; i++){
    if(valid[i]){
      at_least_one = true;
      std::cout << i << " ";
    }
  }
  
  if(!at_least_one){
    std::cout << "Abort mission";
  }
  
  std::cout << std::endl; // Output the final result
  
}
int main() {
  std::ios_base::sync_with_stdio(false); // Always!
  int t; std::cin >> t; // Read the number of test cases
  for (int i = 0; i < t; ++i){
    testcase(); // Solve a particular test case
  }
    
}
