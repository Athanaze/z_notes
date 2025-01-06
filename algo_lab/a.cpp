#include <iostream>
#include <vector>
void solve(){
  
  int n_coins;
  std::cin >> n_coins;
  
  std::vector<int> vec_v(n_coins, 0);
  
  for(int v = 0; v < n_coins; v++){
    std::cin >> vec_v[v];
  }
  
  std::vector<std::vector<int>> dp(n_coins, std::vector<int>(n_coins, 0));
  
  for(int x=0; x < n_coins; x++){
    // Fill the diagonal
    dp[x][x] = vec_v[x];
  }

  for(int i=n_coins-1; i >= 0; i--){
    for(int j=i; j < n_coins; j++){
       auto ll = 0;
      if(i+2 < n_coins && i+2 <= j){
         ll = dp[i+2][j] + vec_v[i];
      }
      
      auto lr = 0;
      if(i+1 < n_coins && j-1 >= 0){
         lr = dp[i+1][j-1] + vec_v[i];
      }
      
      auto rl = 0;
      
      if(i+1 < n_coins && j-1 >= 0 && (i+1) <= (j-1)){
        rl = dp[i+1][j-1] + vec_v[j];
      }
      
      auto rr = 0;
      if(j-2 >= i){
        rr = dp[i][j-2] + vec_v[j];
      }
      
      dp[i][j] = std::min(std::max(ll, rl), std::max(lr, rr));
    }
  }
  
  std::cout << dp[0][n_coins-1] << std::endl;
  
}

int main(){
  int n_tests;
  std::cin >> n_tests;
  
  for(int t_run=0; t_run < n_tests; t_run++){
    solve();
  }
}
