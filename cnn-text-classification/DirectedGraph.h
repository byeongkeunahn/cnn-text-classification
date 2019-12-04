
#pragma once


class DirectedGraph
{
public:
    DirectedGraph();
    ~DirectedGraph();

public:
    int NewVertex(); // allocates new vertex and returns its number (* zero-based indexing)
    void AddEdge(int from, int to);

    std::vector<int> TopologicalSort(bool *pfSucceeded); // obtains the topological sort of the current graph

private:
    int m_V;
    std::vector<std::vector<int>> m_adj;
};
