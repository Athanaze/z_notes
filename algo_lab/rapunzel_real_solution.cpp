#include <iostream>
#include <vector>
#include <set>

void dfs(int node, 
         std::vector<std::vector<int>>& children,
         std::vector<int>& brightness,
         std::vector<bool>& valid,
         std::vector<int>& path,
         std::multiset<int>& window,
         int m,
         int k) {
    // push current node onto DFS path and add its brightness into the window.
    path.push_back(node);
    window.insert(brightness[node]);

    int L = path.size();
    bool removedFlag = false;
    int removedValue = 0;
    // If the path length is > m then remove the brightness of the (L - m - 1)th node
    if(L > m){
        int removeIndex = L - m - 1;
        removedValue = brightness[path[removeIndex]];
        auto it = window.find(removedValue);
        if(it != window.end()){
            window.erase(it);
        }
        removedFlag = true;
    }

    // Check if current window of m nodes is valid
    if(L >= m){
        int currMin = *window.begin();
        int currMax = *window.rbegin();
        if(currMax - currMin <= k){
            valid[path[L - m]] = true;
        }
    }

    // Recurse on children
    for (int child : children[node]){
      dfs(child, children, brightness, valid, path, window, m, k);
    }
    // Restore window state before backtracking
    if(removedFlag){
      window.insert(removedValue);
    }
    // Remove our own brightness and pop from path
    auto it = window.find(brightness[node]);
    if(it != window.end()){
      window.erase(it);
    }
        
    path.pop_back();
}

void testcase(){
  int n, m, k;
  std::cin >> n >> m >> k;

  std::vector<int> brightness(n);
  for (int i = 0; i < n; i++){
      std::cin >> brightness[i];
  }

  std::vector<std::vector<int>> children(n);
  for (int i = 0; i < n - 1; i++){
      int u, v;
      std::cin >> u >> v;
      children[u].push_back(v);
  }

  std::vector<bool> valid(n, false);
  std::vector<int> path;
  std::multiset<int> window;

  dfs(0, children, brightness, valid, path, window, m, k);

  bool any = false;
  for (int i = 0; i < n; i++){
      if(valid[i]){
          if(any){
            std::cout << " ";
          } 
          std::cout << i;
          any = true;
      }
  }
  if(!any){
      std::cout << "Abort mission";
  }
  std::cout << "\n";
}


int main(){
    std::ios_base::sync_with_stdio(false); 
    
    int t;
    std::cin >> t;
    while(t--){
        testcase();
    }
    return 0;
}
