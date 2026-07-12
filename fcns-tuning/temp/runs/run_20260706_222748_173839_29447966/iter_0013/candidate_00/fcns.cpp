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
    mt19937_64 rng;
    bool colour_prefers_different = true;

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

    int UVERTEX() {
vector<int> candidates;
        vector<int> fallback;
        for (int v = 0; v < n; ++v) {
            if (in_C[v]) {
                continue;
            }
            fallback.push_back(v);
            if (domain_size[v] > 1) {
                candidates.push_back(v);
            }
        }
        const vector<int>& pool = candidates.empty() ? fallback : candidates;
        // Prefer vertices with smaller domain size (more constrained) to reduce branching
        int best_domain = numeric_limits<int>::max();
        vector<int> best_pool;
        for (int v : pool) {
            if (domain_size[v] < best_domain) {
                best_domain = domain_size[v];
                best_pool.clear();
                best_pool.push_back(v);
            } else if (domain_size[v] == best_domain) {
                best_pool.push_back(v);
            }
        }
        uniform_int_distribution<int> dist(0, static_cast<int>(best_pool.size()) - 1);
        return best_pool[dist(rng)];
    }

    int CVERTEX() {
int best_domain = -1;
        int best_degree = numeric_limits<int>::max();
        vector<int> candidates;

        for (int v = 0; v < n; ++v) {
            if (!in_C[v]) {
                continue;
            }
            if (domain_size[v] > best_domain) {
                best_domain = domain_size[v];
                best_degree = degree[v];
                candidates.clear();
                candidates.push_back(v);
            } else if (domain_size[v] == best_domain) {
                if (degree[v] < best_degree) {
                    best_degree = degree[v];
                    candidates.clear();
                    candidates.push_back(v);
                } else if (degree[v] == best_degree) {
                    candidates.push_back(v);
                }
            }
        }
        // Among candidates, prefer vertices with higher uncolored degree to free more constraints
        int best_udeg = -1;
        vector<int> final_candidates;
        for (int v : candidates) {
            int udeg = uncolored_degree(v);
            if (udeg > best_udeg) {
                best_udeg = udeg;
                final_candidates.clear();
                final_candidates.push_back(v);
            } else if (udeg == best_udeg) {
                final_candidates.push_back(v);
            }
        }
        uniform_int_distribution<int> dist(0, static_cast<int>(final_candidates.size()) - 1);
        return final_candidates[dist(rng)];
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

    int COLOUR(int u, const vector<int>& D) {
if (D.empty()) {
            return -1;
        }

        // Prefer colors that are least used among neighbors to spread colors
        vector<int> best_colors;
        int min_conflict = numeric_limits<int>::max();
        for (int c : D) {
            int conflict = 0;
            for (int to : graph[u]) {
                if (in_C[to] && colouring[to] == c) {
                    ++conflict;
                }
            }
            if (conflict < min_conflict) {
                min_conflict = conflict;
                best_colors.clear();
                best_colors.push_back(c);
            } else if (conflict == min_conflict) {
                best_colors.push_back(c);
            }
        }
        // Among best, prefer remembered color if not conflicting
        int remembered_color = remembered[u];
        for (int c : best_colors) {
            if (c == remembered_color) {
                return c;
            }
        }
        // Otherwise random among best
        shuffle(best_colors.begin(), best_colors.end(), rng);
        return best_colors.front();
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
};

vector<int> greedy_coloring(const vector<vector<int>>& graph) {
int n = static_cast<int>(graph.size());
    vector<int> color(n, -1);
    // Use DSATUR-like heuristic: pick vertex with largest saturation degree (number of different colors used among neighbors)
    vector<int> sat_degree(n, 0);
    vector<int> used_colors(n, 0); // number of distinct colors used among neighbors
    vector<bool> color_used(n, false);
    int max_color = 0;
    for (int i = 0; i < n; ++i) {
        // Initially, saturation degree is 0
        sat_degree[i] = 0;
    }
    // Use a set to order vertices by (-sat_degree, -degree, index)
    set<tuple<int, int, int>> pq; // (-sat_degree, -degree, index)
    for (int i = 0; i < n; ++i) {
        pq.insert({0, -static_cast<int>(graph[i].size()), i});
    }
    vector<int> color(n, -1);
    while (!pq.empty()) {
        auto it = pq.begin();
        int v = get<2>(*it);
        pq.erase(it);
        // Find smallest color not used by neighbors
        vector<bool> used_colors(n, false);
        for (int to : graph[v]) {
            if (color[to] != -1) {
                used_colors[color[to]] = true;
            }
        }
        int c = 0;
        while (c < n && used_colors[c]) ++c;
        color[v] = c;
        // Update saturation degrees of uncolored neighbors
        for (int to : graph[v]) {
            if (color[to] == -1) {
                // Recompute saturation degree for neighbor
                int old_sat = sat_degree[to];
                // Compute new saturation degree
                vector<bool> neighbor_colors(n, false);
                for (int u : graph[to]) {
                    if (color[u] != -1) {
                        neighbor_colors[color[u]] = true;
                    }
                }
                int new_sat = 0;
                for (int col = 0; col < n; ++col) {
                    if (neighbor_colors[col]) ++new_sat;
                }
                // Update in set
                pq.erase({-old_sat, -static_cast<int>(graph[to].size()), to});
                sat_degree[to] = new_sat;
                pq.insert({-new_sat, -static_cast<int>(graph[to].size()), to});
            }
        }
    }
    // Remap colors to be consecutive
    vector<int> remap(n, -1);
    int next_color = 0;
    for (int v = 0; v < n; ++v) {
        if (remap[color[v]] == -1) {
            remap[color[v]] = next_color++;
        }
        color[v] = remap[color[v]];
    }
    return color;
}

int count_used_colors(const vector<int>& colors) {
    int used = 0;
    for (int c : colors) {
        used = max(used, c + 1);
    }
    return used;
}

vector<int> make_best_coloring(const vector<vector<int>>& graph,
                               const chrono::steady_clock::time_point& launch_time) {
    int n = static_cast<int>(graph.size());
    vector<int> best = greedy_coloring(graph);
    int best_colors = count_used_colors(best);

    uint64_t base_seed = static_cast<uint64_t>(chrono::high_resolution_clock::now().time_since_epoch().count());
    vector<int> b_values = {1, 2, 4, 6, 8, 15, 25, 40};
    sort(b_values.begin(), b_values.end());
    b_values.erase(unique(b_values.begin(), b_values.end()), b_values.end());

    int attempts = 200;
    int target_k = best_colors - 1;
    for (int attempt = 0; attempt < attempts && target_k >= 1; ++attempt) {
        int b = b_values[attempt % static_cast<int>(b_values.size())];
        int max_steps = 1000 * n;
        uint64_t seed = base_seed + static_cast<uint64_t>(attempt) * 0x9e3779b97f4a7c15ULL;
        Solver solver(n, 0, graph, target_k, seed);
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
