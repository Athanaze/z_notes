///3
#include <bits/stdc++.h>
using namespace std;

struct Frame {
    int node;
    bool is_exit;
};

int main(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    int t;
    cin >> t;
    while(t--){
        int n, m;
        long long k;
        cin >> n >> m >> k;
        vector<long long> h(n);
        for(int i=0; i<n; ++i){
            cin >> h[i];
        }
        // Build tree
        vector<vector<int>> children(n, vector<int>());
        for(int i=0; i<n-1; ++i){
            int u, v;
            cin >> u >> v;
            children[u].push_back(v);
        }
        // Initialize result
        vector<bool> result(n, false);
        // Initialize path and deques
        vector<int> path;
        deque<int> max_deque;
        deque<int> min_deque;
        // Initialize stack
        vector<Frame> stack;
        stack.push_back(Frame{0, false});
        while(!stack.empty()){
            Frame frame = stack.back();
            stack.pop_back();
            if(!frame.is_exit){
                // Push exit frame
                stack.push_back(Frame{frame.node, true});
                // Push children in reverse order
                for(auto it = children[frame.node].rbegin(); it != children[frame.node].rend(); ++it){
                    stack.push_back(Frame{*it, false});
                }
                // Entering the node
                path.push_back(frame.node);
                int current_index = path.size()-1;
                // Update max_deque
                while(!max_deque.empty() && h[frame.node] >= h[path[max_deque.back()]]){
                    max_deque.pop_back();
                }
                max_deque.push_back(current_index);
                // Update min_deque
                while(!min_deque.empty() && h[frame.node] <= h[path[min_deque.back()]]){
                    min_deque.pop_back();
                }
                min_deque.push_back(current_index);
                // Remove elements outside the window
                while(!max_deque.empty() && max_deque.front() <= current_index - m){
                    max_deque.pop_front();
                }
                while(!min_deque.empty() && min_deque.front() <= current_index - m){
                    min_deque.pop_front();
                }
                // Check window size
                if(path.size() >= m){
                    long long max_h = h[path[max_deque.front()]];
                    long long min_h = h[path[min_deque.front()]];
                    if(max_h - min_h <= k){
                        int start_node = path[path.size()-m];
                        result[start_node] = true;
                    }
                }
            }
            else{
                // Exiting the node
                // Remove from path
                if(!path.empty()){
                    path.pop_back();
                }
                // current_index is now path.size()
                int current_index = path.size();
                // Remove from deques if necessary
                if(!max_deque.empty() && max_deque.back() == current_index){
                    max_deque.pop_back();
                }
                if(!min_deque.empty() && min_deque.back() == current_index){
                    min_deque.pop_back();
                }
            }
        }
        // Collect results
        vector<int> ans;
        for(int i=0; i<n; ++i){
            if(result[i]){
                ans.push_back(i);
            }
        }
        if(ans.empty()){
            cout << "Abort mission\n";
        }
        else{
            // Sort the ans (they are already in order of traversal, but to ensure increasing order)
            sort(ans.begin(), ans.end());
            // Remove duplicates if any (though there shouldn't be any)
            ans.erase(unique(ans.begin(), ans.end()), ans.end());
            // Print
            for(int i=0; i<ans.size(); ++i){
                if(i >0) cout << ' ';
                cout << ans[i];
            }
            cout << '\n';
        }
    }
}
