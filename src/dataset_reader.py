def dataset_reader(file_path):
    num_nodes = 0
    num_arcs = 0
    sources = {}
    sinks = {}
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
                    if len(tokens) > 3:
                        commodity_id = int(tokens[3])
                    else:
                        commodity_id = 1
                    if node_type == "s":
                        sources[commodity_id] = node_id
                    elif node_type == "t":
                        sinks[commodity_id] = node_id
                case "a":
                    u = int(tokens[1])
                    v = int(tokens[2])
                    c_uv = float(tokens[3])  # capacity
                    arcs.append((u, v, c_uv))
                case _:
                    continue

    return {
        "num_nodes": num_nodes,
        "num_arcs": num_arcs,
        "sources": sources,
        "sinks": sinks,
        "arcs": arcs,
    }
