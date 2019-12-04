
#include "pch.h"
#include "DirectedGraph.h"


DirectedGraph::DirectedGraph() {
    m_V = 0;
}

DirectedGraph::~DirectedGraph() {
}

int DirectedGraph::NewVertex() {
    m_adj.push_back(std::vector<int>());
    return m_V++;
}

void DirectedGraph::AddEdge(int from, int to) {
    m_adj[from].push_back(to);
}

std::vector<int> DirectedGraph::TopologicalSort(bool *pfSucceeded) {
    std::vector<int> completion_list;

    /* topological sorting via dfs */
    std::vector<int> states;
    states.resize(m_V);
    std::fill(states.begin(), states.end(), 0);
    std::function<void(int)> dfs = [&](int root) {
        if (!states[root]) {
            states[root] = 1;
            for (auto to : m_adj[root]) {
                if (states[to] == 1) { /* back edge; indicates cyclic graph */
                    *pfSucceeded = false;
                    return;
                }
                dfs(to);
            }
            states[root] = 2;
            completion_list.push_back(root);
        }
    };
    for (int i = 0; i < m_V; i++) {
        dfs(i);
    }

    *pfSucceeded = true;
    std::reverse(completion_list.begin(), completion_list.end());
    return completion_list;
}
