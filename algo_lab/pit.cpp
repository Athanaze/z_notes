// top down DP
#include <vector>
#include <iostream> 
#include <tuple>
#include <set>
#include <limits>
#include <cmath>

#define NORTH true
#define SOUTH false

typedef std::vector<int> last3;

int penalty_f(int p, int q){
  return std::pow(2, std::abs((p )-q));
}

// p : number fighters through north entrance, q : number fighters through south entrance
int excitement(int p, int q, last3 lnorth, last3 lsouth, int ftype, bool choice, int m){
 int penalty = 0;
 int e = 0;
 
 std::set<int> last_m_fighters;
 last_m_fighters.insert(ftype);
 if(choice == NORTH){
   
   m = std::min(m, p);
   penalty =penalty_f(p+1, q);
   int i = lnorth.size()-1;
   int counter = 0;
   while(counter < lnorth.size()){
     last_m_fighters.insert(lnorth[i]);
     i--;
     counter ++;
   }
 }
 else{
   m = std::min(m, q);
   penalty = penalty_f(p, q+1); 
   
   int i = lsouth.size()-1;
   int counter = 0;
   while(counter < lsouth.size()){
     last_m_fighters.insert(lsouth[i]);
     i--;
     counter ++;
   }
   //std::cout << "size : " << last_m_fighters.size() <<" c " <<counter << " m " << m<< std::endl;
 }
 
 e = 1000*last_m_fighters.size();
 
 auto score = e-penalty;
 if(score < 0){
   score = std::numeric_limits<int>::min();
 }
 return score;
}


int total_excitement(int round_index, std::vector<int> fighters, last3 lnorth, last3 lsouth, int m, int p, int q, int n){
  
  if(round_index == n) {
    return 0;
  }
  
  last3 new_lnorth;
  
  
  
  // Ensure m-1 is valid and within the size of the array
  if (lnorth.size() >= m - 1) {
      for (int i = lnorth.size() - (m - 1); i < lnorth.size(); ++i) {
          new_lnorth.push_back(lnorth[i]);
      }
  }else{
    if(lnorth.size() == 1){
      new_lnorth.push_back(lnorth[0]);
    }
  }
  new_lnorth.push_back(fighters[round_index]);
   
  last3 new_lsouth;
   if (lsouth.size() >= m - 1) {
      for (int i = lsouth.size() - (m - 1); i < lsouth.size(); ++i) {
          new_lsouth.push_back(lsouth[i]);
      }
  }else{
    if(lsouth.size() == 1){
      new_lsouth.push_back(lsouth[0]);
    }
  }
  new_lsouth.push_back(fighters[round_index]);
  
  int total_next_rounds_north = 0;
  int total_next_rounds_south = 0;
  
  if(round_index < n-1){
    total_next_rounds_north = total_excitement(round_index+1, fighters, new_lnorth, lsouth, m, p+1, q, n);
    total_next_rounds_south = total_excitement(round_index+1, fighters, lnorth, new_lsouth, m, p, q+1, n);
  }
  
  auto choosing_north_for_this_round = excitement(p, q, lnorth, lsouth, fighters[round_index], NORTH, m ) + total_next_rounds_north;
  auto choosing_south_for_this_round = excitement(p, q, lnorth, lsouth, fighters[round_index], SOUTH, m ) + total_next_rounds_south;
  
  return std::max(choosing_north_for_this_round, choosing_south_for_this_round);
  
}

void testcase() {
  int n, k,m; 
  std::cin >> n >> k >> m; 
  
  std::vector<int> fighters(n, 0);
  for (int i = 0; i < n; ++i) {
    std::cin >> fighters[i];
  }
  
  last3 lnorth;
  last3 lsouth;
  
  auto result = total_excitement(0, fighters, lnorth, lsouth, m, 0, 0, n);

  std::cout << result << std::endl; // Output the final result
}
int main() {
  std::ios_base::sync_with_stdio(false); // Always!
  int t; std::cin >> t; // Read the number of test cases
  for (int i = 0; i < t; i++){
    testcase(); // Solve a particular test case
  }
  
}
