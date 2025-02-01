///3
#include <iostream> // We will use C++ input/output via streams
#include <vector>
#include <limits>

void explore(
  std::vector<std::vector<int>> &strands,
  int m_counter,
  int from,
  int m,
  int min_hb,
  int max_hb,
  std::vector<int> &hb,
  int k,
  bool* output
  ){

  
  if( m_counter == m-1 && max_hb-min_hb <= k ){
    *output = true;
    return;
    
  }
  if(m_counter >= m || max_hb-min_hb > k){
    return;
  }
  for(int child : strands[from]){
      //std::cout << "child : " << child << std::endl;
      int this_hb = hb[child];
      explore(strands, m_counter+1, child, m, std::min(this_hb, min_hb), std::max(this_hb, max_hb), hb, k, output);
  }
}

void testcase() {
  int n, m , k;
  std::cin >> n >> m >> k; 
  
  std::vector<int> hb(n, 0);
  for (int i = 0; i < n; i++) {
    std::cin >> hb[i]; 
  }
  
  
  std::vector<std::vector<int>> strands(n, std::vector<int>());
  for(int i= 0; i < n-1; i++){
    int u, v;
    std::cin >> u >> v;
    // from u to v
    strands[u].push_back(v);
  }
  
  // We try to start from every hair tie
  
  bool found_one = false;
  for(int start_index = 0; start_index < n; start_index++){
    bool output = false;
    int min_hb = hb[start_index];
    int max_hb = hb[start_index];
    
    explore(strands, 0, start_index, m, min_hb, max_hb, hb, k, &output);
    if(output){
        found_one = true;
        std::cout << start_index << " ";
    }
  }
  if(!found_one) std::cout << "Abort mission";
  
  std::cout << std::endl;
}
int main() {
  std::ios_base::sync_with_stdio(false); // Always!
  int t; std::cin >> t; // Read the number of test cases
  for (int i = 0; i < t; ++i){
    testcase();
  }
}
