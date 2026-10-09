import numpy as np


def dataset_reader(file_path):
    num_nodes = 0
    num_arcs = 0
    source = None
    sink = None
    arcs = []

    with open(file_path, "r", encoding="utf-8") as file:
        for l in file:
            if l.startswith("c") or not l.split():
                continue
            tokens = l.split()
            # tokens[0] -> token type
            match tokens[0]:
                case "p":
                    num_nodes = int(tokens[2])
                    num_arcs = int(tokens[3])
                case "n":
                    node_id = int(tokens[1])
                    node_type = tokens[2]  # source or target(sink)
                    if node_type == "s":
                        source = node_id
                    elif node_type == "t":
                        sink = node_id
                case "a":
                    u = int(tokens[1])
                    v = int(tokens[2])
                    c_uv = float(tokens[3])  # capacity
                    arcs.append((u, v, c_uv))
                case _:
                    continue

    b = np.array([c_uv for _, _, c_uv in arcs], dtype=np.float64)
    c = np.zeros(len(arcs), dtype=np.float64)

    return {
        "num_nodes": num_nodes,
        "num_arcs": num_arcs,
        "source": source,
        "sink": sink,
        "arcs": arcs,
        "b": b,
        "c": c,
    }
