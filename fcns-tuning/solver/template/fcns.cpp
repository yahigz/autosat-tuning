#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <utility>
#include <vector>

using namespace std;

namespace {

struct Solver {
    int n;
    int m;
    int k;
    vector<vector<int>> graph;
    vector<int> degree;
    vector<vector<unsigned short>> forbidden;
    vector<int> domain_size;
    vector<int> colouring;
    vector<int> remembered;
    vector<char> in_C;
    vector<int> bad_count;
    mutable mt19937_64 rng;
    mutable bool colour_prefers_different = true;

    Solver(int n_, int m_, const vector<vector<int>>& graph_, int k_, uint64_t seed)
    : n(n_),
      m(m_),
      k(k_),
      graph(graph_),
      degree(n_),
      forbidden(n_, vector<unsigned short>(k_, 0)),
      domain_size(n_, k_),
      colouring(n_, -1),
      remembered(n_, -1),
      in_C(n_, 0),
      bad_count(n_ * k_, 0),
      rng(seed)
      {
        for (int v = 0; v < n; ++v) {
            degree[v] = static_cast<int>(graph[v].size());
        }
        for (int v = 0; v < n; ++v) {
            int single = singleton_color(v);
            if (single != -1) {
                add_blocking(v, -1, single);
            }
        }
    }

    int singleton_color(int v) const {
        if (domain_size[v] != 1) {
            return -1;
        }
        for (int c = 0; c < k; ++c) {
            if (forbidden[v][c] == 0) {
                return c;
            }
        }
        return -1;
    }

    int bad(int v, int c) const {
        return bad_count[v * k + c];
    }

    void add_blocking(int v, int old_single, int new_single) {
        if (old_single == new_single) {
            return;
        }
        if (old_single != -1) {
            for (int to : graph[v]) {
                --bad_count[to * k + old_single];
            }
        }
        if (new_single != -1) {
            for (int to : graph[v]) {
                ++bad_count[to * k + new_single];
            }
        }
    }

    int U_size() const {
        int cnt = 0;
        for (char is_in_C : in_C) {
            if (!is_in_C) {
                ++cnt;
            }
        }
        return cnt;
    }

    int C_size() const {
        return n - U_size();
    }

    void uncolour_vertex_and_update_domains(int v) {
        int c = colouring[v];
        in_C[v] = 0;
        colouring[v] = -1;

        int single = singleton_color(v);
        if (single != -1) {
            add_blocking(v, -1, single);
        }

        for (int to : graph[v]) {
            int old_domain = domain_size[to];
            int old_single = -1;
            if (!in_C[to] && old_domain == 1) {
                old_single = singleton_color(to);
            }

            if (forbidden[to][c] > 0) {
                if (forbidden[to][c] == 1) {
                    ++domain_size[to];
                }
                --forbidden[to][c];
            }

            int new_domain = domain_size[to];
            int new_single = -1;
            if (!in_C[to] && new_domain == 1) {
                new_single = singleton_color(to);
            }
            if (!in_C[to]) {
                add_blocking(to, old_single, new_single);
            }
        }
    }

    void colour_vertex_and_update_domains(int v, int c) {
        if (!in_C[v]) {
            int old_single = singleton_color(v);
            if (old_single != -1) {
                add_blocking(v, old_single, -1);
            }
        }

        in_C[v] = 1;
        colouring[v] = c;
        remembered[v] = c;

        for (int to : graph[v]) {
            int old_domain = domain_size[to];
            int old_single = -1;
            if (!in_C[to] && old_domain == 1) {
                old_single = singleton_color(to);
            }

            if (forbidden[to][c] == 0) {
                --domain_size[to];
            }
            ++forbidden[to][c];

            int new_domain = domain_size[to];
            int new_single = -1;
            if (!in_C[to] && new_domain == 1) {
                new_single = singleton_color(to);
            }
            if (!in_C[to]) {
                add_blocking(to, old_single, new_single);
            }
        }
    }

    int uncolored_degree(int v) const {
        int result = 0;
        for (int to : graph[v]) {
            if (!in_C[to]) {
                ++result;
            }
        }
        return result;
    }

    int UVERTEX() const {
int best_domain = numeric_limits<int>::max();
int best_degree = -1;
vector<int> candidates;
for (int v = 0; v < n; ++v) {
    if (in_C[v]) continue;
    int dom = domain_size[v];
    int deg = degree[v];
    if (dom < best_domain || (dom == best_domain && deg > best_degree)) {
        best_domain = dom;
        best_degree = deg;
        candidates.clear();
        candidates.push_back(v);
    } else if (dom == best_domain && deg == best_degree) {
        candidates.push_back(v);
    }
}
if (candidates.empty()) {
    for (int v = 0; v < n; ++v) {
        if (!in_C[v]) candidates.push_back(v);
    }
}
uniform_int_distribution<int> dist(0, static_cast<int>(candidates.size()) - 1);
return candidates[dist(rng)];
    }

    int CVERTEX() const {
int best_domain = -1;
int best_bad = numeric_limits<int>::max();
vector<int> candidates;
for (int v = 0; v < n; ++v) {
    if (!in_C[v]) continue;
    int dom = domain_size[v];
    int b = 0;
    for (int c = 0; c < k; ++c) {
        if (forbidden[v][c] == 0 && bad(v, c) == 0) b++;
    }
    if (dom > best_domain || (dom == best_domain && b < best_bad)) {
        best_domain = dom;
        best_bad = b;
        candidates.clear();
        candidates.push_back(v);
    } else if (dom == best_domain && b == best_bad) {
        candidates.push_back(v);
    }
}
uniform_int_distribution<int> dist(0, static_cast<int>(candidates.size()) - 1);
return candidates[dist(rng)];
    }

    vector<int> build_D(int u) const {
        vector<int> result;
        for (int c = 0; c < k; ++c) {
            if (forbidden[u][c] == 0 && bad(u, c) == 0) {
                result.push_back(c);
            }
        }
        return result;
    }

    int COLOUR(int u, const vector<int>& D) const {
if (D.empty()) return -1;
int remembered_color = remembered[u];
// Prefer the remembered color if it is available and not causing immediate conflict
for (int c : D) {
    if (c == remembered_color) {
        return c;
    }
}
// Otherwise, pick a color that minimizes the number of neighbors that would become singleton
int best_c = D[0];
int best_impact = numeric_limits<int>::max();
for (int c : D) {
    int impact = 0;
    for (int to : graph[u]) {
        if (!in_C[to] && forbidden[to][c] == 0) {
            if (domain_size[to] == 1) impact += 10;
            else if (domain_size[to] == 2) impact += 1;
        }
    }
    if (impact < best_impact) {
        best_impact = impact;
        best_c = c;
    }
}
return best_c;
    }

    pair<int, vector<int>> FCNS(int B, int max_steps) {
        int steps = 0;
        while (steps < max_steps) {
            if (U_size() == 0) {
                break;
            }

            int u = UVERTEX();
            if (u < 0) {
                break;
            }

            vector<int> D = build_D(u);
            if (!D.empty()) {
                int c = COLOUR(u, D);
                colour_vertex_and_update_domains(u, c);
            } else {
                int limit = min(B, C_size());
                for (int i = 0; i < limit; ++i) {
                    int c_vertex = CVERTEX();
                    if (c_vertex < 0) {
                        break;
                    }
                    uncolour_vertex_and_update_domains(c_vertex);
                    colour_prefers_different = true;
                }
            }

            ++steps;
        }

        vector<int> result = colouring;
        int used_colors = 0;
        vector<int> remap(k, -1);
        for (int v = 0; v < n; ++v) {
            if (result[v] < 0) {
                continue;
            }
            if (remap[result[v]] == -1) {
                remap[result[v]] = used_colors++;
            }
            result[v] = remap[result[v]];
        }
        return {used_colors, result};
    }

    vector<int> greedy_coloring(const vector<vector<int>>& graph) {
int n = graph.size();
vector<int> color(n, -1);
vector<int> order(n);
iota(order.begin(), order.end(), 0);
sort(order.begin(), order.end(), [&](int a, int b) {
    return graph[a].size() > graph[b].size();
});
vector<bool> used(n, false);
for (int v : order) {
    fill(used.begin(), used.end(), false);
    for (int to : graph[v]) {
        if (color[to] != -1) used[color[to]] = true;
    }
    int c = 0;
    while (c < n && used[c]) ++c;
    color[v] = c;
}
return color;
    }
};

int count_used_colors(const vector<int>& colors) {
    int used = 0;
    for (int c : colors) {
        used = max(used, c + 1);
    }
    return used;
}

vector<int> make_best_coloring(const vector<vector<int>>& graph,
                               const chrono::steady_clock::time_point& launch_time) {
    int m = 0;
    for (const auto& neighbors : graph) {
        m += static_cast<int>(neighbors.size());
    }
    int n = static_cast<int>(graph.size());
    uint64_t base_seed = 51;
    vector<int> best = Solver(n, m, graph, n, base_seed).greedy_coloring(graph);
    int best_colors = count_used_colors(best);

    vector<int> b_values = {1, 2, 4, 6, 8, 15, 25, 40};
    sort(b_values.begin(), b_values.end());
    b_values.erase(unique(b_values.begin(), b_values.end()), b_values.end());

    int attempts = 200;
    int target_k = best_colors - 1;
    for (int attempt = 0; attempt < attempts && target_k >= 1; ++attempt) {
        int b = b_values[attempt % static_cast<int>(b_values.size())];
        int max_steps = 1000 * n;
        uint64_t seed = base_seed + static_cast<uint64_t>(attempt) * 0x9e3779b97f4a7c15ULL;
        Solver solver(n, m, graph, target_k, seed);
        auto [used_colors, colors] = solver.FCNS(b, max_steps);
        if (static_cast<int>(colors.size()) != n) {
            // it means that the solver failed to color all vertices, so we skip this attempt
            continue;
        }

        bool complete = true;
        for (int v = 0; v < n; ++v) {
            if (colors[v] < 0) {
                complete = false;
                // it means that the solver failed to color all vertices, so we skip this attempt
                break;
            }
        }
        if (!complete) {
            continue;
        }
        --target_k;
        if (used_colors < best_colors) {
            best = std::move(colors);
            best_colors = used_colors;
            const double elapsed_seconds = chrono::duration<double>(
                chrono::steady_clock::now() - launch_time
            ).count();
            cerr << "{\"colors\": " << best_colors
                 << ", \"time\": " << elapsed_seconds << "}" << endl;
        }
    }

    return best;
}

}  // namespace

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    const auto launch_time = chrono::steady_clock::now();

    int n, m;
    cin >> n >> m;
    vector<vector<int>> graph(n);
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    vector<int> answer = make_best_coloring(graph, launch_time);
    int colors_used = count_used_colors(answer);

    cout << colors_used << '\n';
    for (int v = 0; v < n; ++v) {
        cout << answer[v] + 1 << ' ';
    }
    cout << '\n';
    return 0;
}
