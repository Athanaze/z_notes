///1



#include <bits/stdc++.h>
using namespace std;

/*
  getOverlap(p, q, k, tillHead):
    Returns the largest t in [0..k] such that
    the suffix of p of length t equals the prefix of q of length t,
    and the new cuts in q (beyond the overlap) do not involve eradicated heads (heads < tillHead).
    Returns -1 if no valid overlap is possible.
*/
int getOverlap(const vector<int> &p, const vector<int> &q, int k, int tillHead) {
    for(int t = k; t >= 0; t--){
        bool match = true;
        // Check overlapping positions (no need to check for eradicated heads here)
        for(int s = 0; s < t; s++){
            if(p[k - t + s] != q[s]){
                match = false;
                break;
            }
        }
        if(!match) continue;
        // Check new cuts in q (positions beyond the overlap)
        for(int s = t; s < k; s++){
            int h_q = q[s];
            if(h_q < tillHead){
                match = false;
                break;
            }
        }
        if(match) return t;
    }
    return -1; // No valid overlap
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T; cin >> T;
    while(T--){
        int n,m; cin >> n >> m;
        int k,d; cin >> k >> d; // d is not directly needed in this solution

        // P[h] = all patterns of length k whose last head is h.
        vector<vector<vector<int>>> P(n);

        for(int i=0; i<m; i++){
            vector<int> pat(k);
            for(int &x : pat) cin >> x;
            int last = pat[k-1];
            // Add pat to P[last].
            if(last >= 0 && last < n){
                P[last].push_back(pat);
            }
        }

        // If some head h has no pattern, we cannot eradicate it => Impossible.
        bool impossible = false;
        for(int h=0; h<n; h++){
            if(P[h].empty()){
                impossible = true;
                break;
            }
        }
        if(impossible){
            cout << "Impossible!\n";
            continue;
        }

        // Precompute overlaps considering only new cuts (positions beyond the overlap) must not involve eradicated heads.
        vector<vector<vector<int>>> overlapCost(n-1);
        for(int i=0; i+1<n; i++){
            int szA = (int)P[i].size();
            int szB = (int)P[i+1].size();
            overlapCost[i].assign(szA, vector<int>(szB, -1)); // Initialize with -1 (invalid)
            for(int u = 0; u < szA; u++){
                for(int v = 0; v < szB; v++){
                    // Heads 0 to i have been eradicated when moving to head i+1
                    overlapCost[i][u][v] = getOverlap(P[i][u], P[i+1][v], k, i+1);
                }
            }
        }

        // dp[i][u] = minimal number of total cuts to eradicate heads [0..i],
        //            if we use the u-th pattern of P[i] to kill head i.
        vector<vector<int>> dp(n);
        for(int i=0; i<n; i++){
            dp[i].resize(P[i].size(), INT_MAX);
        }

        // Base case for head 0: using any pattern p in P[0]
        for(int u=0; u<(int)P[0].size(); u++){
            bool valid = true;
            // Ensure the pattern does not require cuts on eradicated heads (no heads eradicated yet)
            for(int s = 0; s < k; s++){
                int h = P[0][u][s];
                if(h < 0 || h < 0){ // All heads are >= 0 initially
                    valid = false;
                    break;
                }
            }
            if(valid) dp[0][u] = k;
        }

        // Fill dp for i=1..n-1 using chaining with valid overlaps
        for(int i=1; i<n; i++){
            int szA = (int)P[i-1].size();
            int szB = (int)P[i].size();
            for(int v=0; v<szB; v++){
                int best = INT_MAX;
                for(int u=0; u<szA; u++){
                    if(dp[i-1][u] == INT_MAX) continue;
                    int ov = overlapCost[i-1][u][v];
                    if(ov == -1) continue; // Invalid overlap (new cuts involve eradicated heads)
                    // We add (k - ov) new cuts
                    best = min(best, dp[i-1][u] + (k - ov));
                }
                dp[i][v] = best;
            }
        }

        // Final answer is min of dp[n-1][u]
        int ans = *min_element(dp[n-1].begin(), dp[n-1].end());
        if(ans == INT_MAX) cout << "Impossible!\n";
        else cout << ans << "\n";
    }
    return 0;
}
