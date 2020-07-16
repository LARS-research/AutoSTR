import numpy as np

choice = lambda x: x[np.random.randint(len(x))] \
    if isinstance(x, tuple) else choice(tuple(x))

class WarpNode:
    
    def __init__(self, node_str):
        assert isinstance(node_str, str)
        self.node_str = node_str
        self.node_value = [int(v) for v in node_str.split('-')]
    
    def __eq__(self, other_node):
        return self.node_str == other_node.node_str

    def __gt__(self, other_node):
        if self.node_value[-1] != other_node.node_value[-1]:
            return self.node_value[-1] > other_node.node_value[-1]
        elif self.node_value[-2] != other_node.node_value[-2]:
            return self.node_value[-2] > other_node.node_value[-2]
        else:
            return self.node_value[0] > self.node_value[0]


class EAPathFinder:

    def __init__(self, max_layer=15):
        self.max_layer = max_layer
        self._all_paths = []
        self.all_directions = [[0, 0], [1, 0], [1, 1]]
        self._nodes_indegree_map = None
        self._topological_sorted_nodes = None
        self._all_edges = None

    def is_correct_position(self, position):
        x, y = position
        return 0 <= x <= 5 and 0 <= y <= 2

    def find_path(self, path, cur_layer):
        position = path[-1]
        if position == [5, 2] and cur_layer == self.max_layer:
            self._all_paths.append(path)
            return
        if cur_layer >= self.max_layer:
            return
        for dx, dy in self.all_directions:
            next_position = [position[0] + dx, position[1] + dy]
            if self.is_correct_position(next_position):
                self.find_path(path + [next_position], cur_layer + 1)
    
    @property
    def all_paths(self):
        if self._all_paths == []:
            self.find_path([[0, 0]], 0)
        return np.array(self._all_paths)
    
    @property
    def n_paths(self):
        return self.all_paths.shape[0]

    @property
    def nodes_indegree_map(self):
        if self._nodes_indegree_map is None:
            self._nodes_indegree_map = {}
            for path in self.all_paths:
                for i in range(1, len(path)):
                    cur_node = path[i]
                    pre_node = path[i - 1]

                    cur_node_str = '%d-%d-%d' % (cur_node[0], cur_node[1], i)
                    pre_node_str = '%d-%d-%d' % (pre_node[0], pre_node[1], i - 1)
                    if cur_node_str not in self._nodes_indegree_map:
                        self._nodes_indegree_map[cur_node_str] = [pre_node_str]
                    else:
                        self._nodes_indegree_map[cur_node_str].append(pre_node_str)
            for k, node_list in self._nodes_indegree_map.items():
                node_list = [WarpNode(n) for n in node_list]
                node_list.sort()
                self._nodes_indegree_map[k] = [n.node_str for n in node_list]
        return self._nodes_indegree_map

    @property
    def topological_sorted_nodes(self):
        if self._topological_sorted_nodes is None:
            all_nodes = [WarpNode(n) for n in self.nodes_indegree_map.keys()]
            all_nodes.sort()
            self._topological_sorted_nodes = [n.node_str for n in all_nodes]
        return self._topological_sorted_nodes
    
    @property
    def final_node(self):
        return self.topological_sorted_nodes[-1]

    @property
    def n_nodes(self):
        return len(self.topological_sorted_nodes)

    @property
    def all_edges(self):
        if self._all_edges is None:
            self._all_edges = set()
            for path in self.all_paths:
                for i in range(len(path) - 1):
                    cur_node  = path[i]
                    next_node = path[i + 1]
                    cur_node_str  = '%d-%d-%d' % (cur_node[0], cur_node[1], i)
                    next_node_str = '%d-%d-%d' % (next_node[0], next_node[1], i + 1)
                    edge_key = '%s$%s' % (cur_node_str, next_node_str)
                    self._all_edges.add(edge_key)
        return self._all_edges

    def choice_random_path(self, seed):
        np.random.seed(seed)
        return self.all_paths[np.random.randint(self.n_paths)]

    @property
    def default_path(self):
        return np.array([
            [0, 0],
            [0, 0],
            [1, 1],
            [1, 1],
            [1, 1],
            [2, 2],
            [2, 2],
            [2, 2],
            [3, 2],
            [3, 2],
            [3, 2],
            [4, 2],
            [4, 2],
            [4, 2],
            [5, 2],
            [5, 2],
        ])


if __name__ == "__main__":
    ea_pathfinder = EAPathFinder()
    n_paths = ea_pathfinder.n_paths
    n_nodes = ea_pathfinder.n_nodes
    print(n_paths)
    print(n_nodes)
    edges = ea_pathfinder.all_edges
    for path in ea_pathfinder.all_paths:
        if (ea_pathfinder.default_path == path).all():
            print('default path in ')
    import ipdb; ipdb.set_trace()
