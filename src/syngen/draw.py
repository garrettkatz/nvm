import networkx as nx
import matplotlib.pyplot as plt

def recurse_dendrites(layer, parent, dendrites):
    nodes = []
    edges = []
    for dendrite in dendrites:
        name = layer["name"] + "-" + dendrite["name"]
        nodes.append(name)
        edges.append((name, parent["name"]))
        if "children" in dendrite:
            n,e = recurse_dendrites(layer, dendrite, dendrite["children"])
            nodes += n
            edges += e
    return nodes, edges

def draw_network(structures, connections, do_dendrites=False):
    nodes = []
    edges = []

    # add nodes
    for struct in structures:
        for layer in struct["layers"]:
            name = layer["name"]
            if name is "bias": continue
            if name in ["fef", "tc", "sc"]:
                if "device" not in nodes:
                    name = "device"
                else: continue

            nodes.append(name)
            if do_dendrites and "dendrites" in layer:
                n,e = recurse_dendrites(layer, layer, layer["dendrites"])
                nodes += n
                edges += e

    # add edges
    for conn in connections:
        if conn["from layer"] in ["go", "bias"]: continue
        if conn["from layer"] in ["fef", "sc", "tc"]:
            conn["from layer"] = "device"
        if conn["to layer"] in ["fef", "sc", "tc"]:
            conn["to layer"] = "device"
        if do_dendrites and "dendrite" in conn:
            name = conn["to layer"] + "-" + conn["dendrite"]
            edges.append((conn["from layer"], name))
        else:
            edges.append((conn["from layer"], conn["to layer"]))

    # create networkx graph
    G=nx.DiGraph()

    for node in nodes:
        G.add_node(node)
    for edge in edges:
        G.add_edge(*edge)

    # draw graph
    pos = nx.spring_layout(G)
    #pos = nx.shell_layout(G)
    nx.draw_networkx_labels(G, pos, {n:n for n in nodes})
    nx.draw(G, pos)

    # show graph
    plt.show()
