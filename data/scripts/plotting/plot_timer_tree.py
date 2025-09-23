#!/usr/bin/env python3
import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt


def add_nodes_edges(G, node, parent_id=None, metric='avg_time', counter=[0], path="root"):
    """
    Recursively add nodes and edges to G.
    - Each node gets a unique ID based on traversal order.
    - Labels still show just the function name + count/time.
    """
    counter[0] += 1
    node_id = f"{path}_{counter[0]}"  # unique ID for this occurrence
    value = node.get(metric, 0.0)
    count = node.get('count', 0)
    label = f"{node['name']}\n{value:.2f}s / {count}"

    G.add_node(node_id, label=label, metric=value, count=count)
    if parent_id:
        G.add_edge(parent_id, node_id)

    for i, child in enumerate(node.get('children', [])):
        child_path = f"{path}_{i}"
        add_nodes_edges(G, child, node_id, metric, counter, path=child_path)


def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Compute hierarchical positions for a tree (without pygraphviz)
    """
    if root is None:
        root = next(iter(nx.topological_sort(G)))

    def _hierarchy_pos(G, root, left, right, vert_loc, pos=None):
        if pos is None:
            pos = {}
        pos[root] = ((left + right) / 2, vert_loc)
        children = list(G.successors(root))
        if len(children) != 0:
            dx = (right - left) / len(children)
            for i, child in enumerate(children):
                pos = _hierarchy_pos(G, child, left + i * dx, left + (i + 1) * dx, vert_loc - vert_gap, pos)
        return pos

    return _hierarchy_pos(G, root, 0, width, vert_loc)


def plot_tree(G, root_label):
    pos = hierarchy_pos(G)
    node_metrics = [G.nodes[n]['metric'] for n in G.nodes()]
    max_size = 3000
    sizes = [max_size * (m / max(node_metrics) if max(node_metrics) > 0 else 1) for m in node_metrics]
    labels = nx.get_node_attributes(G, 'label')

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, labels=labels,
            node_size=sizes, node_color=node_metrics, cmap=plt.cm.Oranges,
            font_size=8, font_weight='bold', edge_color='gray', ax=ax)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Oranges,
                               norm=plt.Normalize(vmin=min(node_metrics), vmax=max(node_metrics)))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Time')

    # --- Add heading above root node ---
    # Find root node (top of hierarchy)
    root_node = max(pos, key=lambda k: pos[k][1])  # largest y value = top
    x_root, y_root = pos[root_node]
    ax.text(x_root, y_root + 0.05, root_label, fontsize=12, fontweight='bold',
            ha='center', va='top')

    ax.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Tree plot of timing data (no pygraphviz)")
    parser.add_argument("json_file", help="Path to JSON timing tree file")
    parser.add_argument("--metric", choices=['root_time', 'min_time', 'max_time', 'sum_time', 'avg_time'],
                        default='root_time',
                        help="Time metric per MPI process. 'root_time': total time of rank 0, '{min,max,sum,avg}_time':"
                             " {min,max,sum,avg} of total time over MPI processes}")
    args = parser.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)

    G = nx.DiGraph()
    add_nodes_edges(G, data, metric=args.metric)
    plot_tree(G, f"(Accumulated Time (s) / Count) using metric '{args.metric}'")


if __name__ == "__main__":
    main()
