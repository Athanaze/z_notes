///2
#include <iostream>
#include <vector>
#include <limits>

int overlap(std::vector<int> &vec_a, std::vector<int> &vec_b, int k, int tillhead){
  for(int r = k; r >=0; r--){
    int shift = k-r;
    int b_i = 0;
    bool overlap_ok = true;
    for(int a_i= shift; a_i < k; a_i++ ){
      if(vec_a[a_i] != vec_b[b_i]){
        overlap_ok = false;
      }
      b_i++;
    }
    if(overlap_ok){
      bool floating_elements_after_tillhead = true;
      while(b_i < k){
        if(vec_b[b_i] < tillhead){
          floating_elements_after_tillhead = false;
        }
        b_i++;
      }
      if(floating_elements_after_tillhead){
         return r;
      }
    }
  }
  return -1;
}

/// Processes one test case of the Hydra eradication problem.
void testcase() {
    int n, m;
    std::cin >> n >> m;
    
    int k, d;
    std::cin >> k >> d; // d is given but not directly used here.
    
    // P[h] will store all eradication patterns (of length k) that end with head h.
    std::vector< std::vector<std::vector<int>> > P(n);
    
    // Read each eradication pattern and group it by the head it eradicates.
    for (int i = 0; i < m; i++){
        std::vector<int> pattern(k);
        for (int j = 0; j < k; j++){
          std::cin >> pattern[j];
        }
        int lastHead = pattern[k - 1];
        if (lastHead >= 0 && lastHead < n) {
          P[lastHead].push_back(pattern);
        }
    }
    // dp[i][u] indicates the minimum number of cuts needed to eradicate heads 0..i
    // when using the u-th pattern in P[i] for head i.
    const int INF = std::numeric_limits<int>::max();
    std::vector< std::vector<int> > dp(n);
    for (int i = 0; i < n; i++){
        dp[i].resize(P[i].size(), INF);
    }
    
    // Base case: To eradicate head 0, any pattern from P[0] requires k cuts.
    for (int u = 0; u < P[0].size(); u++){
        dp[0][u] = k;
    }
    
    // Fill dp for heads 1 .. n-1 using valid overlaps.
    for (int h = 1; h < n; h++){
      for(int v = 0; v < P[h].size(); v++){
        int best = INF;
        for(int u = 0; u < P[h-1].size(); u++){
          if (dp[h - 1][u] == INF){
            continue;
          }
                    
          int ov = overlap(P[h-1][u], P[h][v], k, h);
          if(ov == -1){
            continue;
          }
          // Add extra cuts needed (k - ov) to chain the patterns.
          best = std::min(best, dp[h - 1][u] + (k - ov));
        }
        dp[h][v] = best;
      }
    }
    
    // The answer is the minimum number of cuts among all patterns for the final head.
    int answer = INF;
    for (int u = 0; u < dp[n - 1].size(); u++){
        answer = std::min(answer, dp[n - 1][u]);
    }
    
    if (answer == INF){
       std::cout << "Impossible!\n";
    }
    else {
      std::cout << answer << "\n";
    }
}

/// Main function: reads test cases and solves each.
int main(){
    std::ios_base::sync_with_stdio(false); 
    int T;
    std::cin >> T;
    while (T--) {
        testcase();
    }
    return 0;
}
