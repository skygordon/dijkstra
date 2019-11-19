"""6.009 Lab 8: Graphs, Paths, Matrices."""

from abc import ABC, abstractmethod
# from datetime import datetime
# NO ADDITIONAL IMPORTS ALLOWED!

# DIJKSTRA Impementation!
class PriorityQueue:
    def __init__(self):
        self.A = {}

    def insert(self, label, key):
        self.A[label] = key
    
    def extract_min(self):
        min_label = None
        for label in self.A:
            if (min_label is None) or (self.A[label] < self.A[min_label]): # .key?
                min_label = label
        del self.A[min_label]
        return min_label

    def decrease_key(self, label, key):
        if key < self.A[label]:
            self.A[label] = key

def relax(Adj, d, parent, u, v):
    if d[v[0]] > v[1] + d[u]:
        d[v[0]] = v[1] + d[u]
        parent[v[0]] = u

def dijkstra(Adj, s):
    not_used_nodes = set()
    d = {}
    parent = {} 
    for v in Adj:
        d[v] = float('inf')
        parent[v] = None
        not_used_nodes.add(v)
    d[s], parent[s] = 0, s # float(‘inf’), s
    Q = PriorityQueue()
    for v in Adj: #or i in range(len(Adj)): 
        Q.insert(v, d[v])  # all nodes negative values
    for _ in range(len(Adj)): 
        u = Q.extract_min() 
        not_used_nodes.remove(u)
        for v in Adj[u]:    #  v = (city, weight)
            if v[0] in not_used_nodes:
                relax(Adj, d, parent, u, v)
                Q.decrease_key(v[0], d[v[0]])
    for i in d: # if not reachable makes distance = None instead of infinity
        if d[i]== float('inf'):
            d[i] = None
    return d, parent


class Graph(ABC):
    """Interface for a mutable directed, weighted graph."""

    @abstractmethod
    def add_node(self, node):
        """Add a node to the graph.

        Arguments:
            node (str): the node to add

        Raises:
            ValueError: if the node already exists.

        """

    @abstractmethod
    def add_edge(self, start, end, weight):
        """Add a directed edge to the graph.

        If the edge already exists, then set its weight to `weight`.

        Arguments:
            start (str): the node where the edge starts
            end (str): the node where the edge ends
            weight (int or float): the weight of the edge, assumed to be a nonnegative number

        Raises:
            LookupError: if either of these nodes doesn't exist

        """

    # @abstractmethod
    def nodes(self):
        """Return the nodes in the graph.

        Returns:
            set: all of the nodes in the graph

        """
        return self.nodess
        
    @abstractmethod
    def neighbors(self, node):
        """Return the neighbors of a node.

        Arguments:
            node (str): a node name

        Returns:
            set: all tuples (`neighbor`, `weight`) for which `node` has an
                 edge to `neighbor` with weight `weight`

        Raises:
            LookupError: if `node` is not in the graph

        """
        

    # @abstractmethod
    def get_path_length(self, start, end):
        """Return the length of the shortest path from `start` to `end`.

        Arguments:
            start (str): a node name
            end (str): a node name

        Returns:
            float or int: the length (sum of edge weights) of the path or
                          `None` if there is no such path.

        Raises:
            LookupError: if either `start` or `end` is not in the graph

        """
        if start not in self.nodess or end not in self.nodess:
            raise LookupError  # LookupError: if `node` is not in the graph
        Adj = self.make_dijkstra_ready()
        distances, parents = dijkstra(Adj, start)
        return distances[end]

    # @abstractmethod
    def get_path(self, start, end):
        """Return the shortest path from `start` to `end`.

        Arguments:
            start (str): a node name
            end (str): a node name

        Returns:
            list: nodes, starting with `start` and, ending with `end`, which
                  comprise the shortest path from `start` to `end` or `None`
                  if there is no such path

        Raises:
            LookupError: if either `start` or `end` is not in the graph

        """
        if start not in self.nodess or end not in self.nodess:
            raise LookupError  # LookupError: if `node` is not in the graph
        Adj = self.make_dijkstra_ready()
        distances, parents = dijkstra(Adj, start)
        if parents[end] == None:
            return None # return `None` if there is no such path
        path = [end]
        node = end
        while True:
            if node == start: # reached the start of the path, path is complete
                path.reverse()
                break
            else:
                node = parents[node]
                path.append(node)
        return path
            

    # @abstractmethod
    def get_all_path_lengths(self):
        """Return lengths of shortest paths between all pairs of nodes.

        Returns:
            dict: map from tuples `(u, v)` to the length of the shortest path
                  from `u` to `v`

        """
        Adj = self.make_dijkstra_ready() # gets it in adjacency format for dijkstra's, runs once at the beginning instead of every time in get path
        path_lengths = {}
        for node in self.nodess:
            distances, parents = dijkstra(Adj, node)
            for other_node in distances:
                if distances[other_node] != None: # if it is reachable
                    path_lengths[(node, other_node)] = distances[other_node]
        return path_lengths

    # @abstractmethod
    def get_all_paths(self):
        """Return shortest paths between all pairs of nodes.

        Returns:
            dict: map from tuples `(u, v)` to a list of nodes (starting with
                  `u` and ending with `v`) which is a shortest path from `u`
                  to `v`

        """
        Adj = self.make_dijkstra_ready() # gets it in adjacency format for dijkstra's, runs once at the beginning instead of every time in get path
        paths = {}
        for node in self.nodess:
            distances, parents = dijkstra(Adj, node)
            for other_node in distances:
                path = [other_node]
                n = other_node
                while True:
                    if n == node: # reached the start of the path, path is complete
                        path.reverse()
                        break
                    else:
                        n = parents[n]
                        if n == None:
                            break
                        path.append(n)
                if n != None:
                    paths[(node, other_node)] = path
        return paths

        
class AdjacencyDictGraph(Graph):
    """A graph represented by an adjacency dictionary."""

    def __init__(self):
        """Create an empty graph."""
        self.graph = {}
        self.nodess = set()

    def add_node(self, node):
        if node in self.nodess:
            raise ValueError  # ValueError: if the node already exists.
        else:
            self.nodess.add(node)
            self.graph[node] = []

    def add_edge(self, start, end, weight):
        if start in self.graph and end in self.graph:
            for edge in self.graph[start]:
                if edge[0]==end:
                    self.graph[start].remove(edge)
                    break
            self.graph[start].append((end, weight))
        else:
            raise LookupError #if either of these nodes doesn't exist

    def neighbors(self, node):
        if node not in self.nodess:
            raise LookupError  # LookupError: if `node` is not in the graph
        return set(self.graph[node])

    def make_dijkstra_ready(self):
        return self.graph


class AdjacencyMatrixGraph(Graph):
    """A graph represented by an adjacency matrix."""

    def __init__(self):
        """Create an empty graph."""
        self.graph = []
        self.node_to_indice = {} # has all nodes 
        self.indice_to_node = {}
        self.nodess = set()

    def add_node(self, node):
        if node in self.nodess:
            raise ValueError  # ValueError: if the node already exists.
        else:
            self.nodess.add(node)
            index = len(self.graph)
            self.node_to_indice[node] = index
            self.indice_to_node[index] = node
            new_row = [float('inf') for n in range(index)] # adds node's list if there isn't one already
            new_row.append(0) # distance to itself for node is always 0
            self.graph.append(new_row)
            for row in range(index):
                self.graph[row].append(float('inf')) # adds value to other nodes arrays 
    
    def get_node(self, node):
        if node in self.node_to_indice:
            index = self.node_to_indice[node] # already exists, got its index in the matrix
            return index # returns node's index in the matrix
        else: 
            raise LookupError #if either of these nodes doesn't exist

    def add_edge(self, start, end, weight):
        start_index = self.get_node(start)
        end_index = self.get_node(end)
        self.graph[start_index][end_index] = weight

    def neighbors(self, node):
        if node not in self.nodess:
            raise LookupError  # LookupError: if `node` is not in the graph
        node_index = self.get_node(node)
        neighbors = set()
        for i in range(len(self.graph)): # goes through given node's array
            if self.graph[node_index][i] != float('inf') and node_index != i: # if it finds a node in the array that isn't itself and it reachable via a edge it adds it to neighbors set
                neighbors.add((self.indice_to_node[i], self.graph[node_index][i]))    # (`neighbor`, `weight`)
        return neighbors
    
    def make_dijkstra_ready(self): # gets it in adjacency format for dijkstra's
        Adj = {}
        for node in self.nodess:
            Adj[node] = list(self.neighbors(node))
        return Adj
       
        
  

class GraphFactory:
    """Factory for creating instances of `Graph`."""

    def __init__(self, cutoff=0.5):
        """Create a new factory that creates instances of `Graph`.

        Arguments:
            cutoff (float): the maximum density (as defined in the lab handout)
                            for which the an `AdjacencyDictGraph` should be
                            instantiated instead of an `AdjacencyMatrixGraph`

        """
        self.cutoff = cutoff # saves cutoff value to use to compare later when factory is used to make graphs

    def from_edges_and_nodes(self, weighted_edges, nodes):
        """Create a new graph instance.

        Arguments:
            weighted_edges (list): the edges in the graph given as
                                   (start, end, weight) tuples
            nodes (list): nodes in the graph

        Returns:
            Graph: a graph containing the given edges

        """
        density = len(weighted_edges)/(len(nodes)*(len(nodes)-1)) # We define density to be the number of edges in the graph, divided by the maximum number of edges which could occur in a graph with the same number of nodes 
        if density <= self.cutoff: # compares against given cutoff
            graph = AdjacencyDictGraph()
        else:
            graph = AdjacencyMatrixGraph()
        for node in nodes:
            graph.add_node(node)
        for s, t, w in weighted_edges: # gets the start, end, weight for each edge
            graph.add_edge(s, t, w)   # adds to graph
        return graph

def get_most_central_node(graph):
    """Return the most central node in the graph.

    "Most central" is defined as having the shortest average round trip to any
    other node.

    Arguments:
        graph (Graph): a graph with at least one node from which round trips
                       to all other nodes are possible

    Returns:
        node (str): the most central node in the graph; round trips to all
                    other nodes must be possible from this node

    """
    APSP = graph.get_all_path_lengths() # APSP = All Pairs Shortest Path-lengths
    amount_of_round_trips = {node:0 for node in graph.nodes()} # dictionary of nodes mapped to how many paths they have
    total_length_round_trips = {node:0 for node in graph.nodes()} # keyed with nodes mapped to value of total length of all round trip paths it has
    for path in APSP:
        amount_of_round_trips[path[0]] += 1 # increases amount of trips by one
        total_length_round_trips[path[0]] += APSP[path]  # increases total length by the given path length
        amount_of_round_trips[path[1]] += 1  # increases amount of trips by one
        total_length_round_trips[path[1]] += APSP[path]  # increases total length by the given path length
    possible_trips = max(dict([(v, k) for k, v in amount_of_round_trips.items()]))
    for n in amount_of_round_trips:
        if amount_of_round_trips[n] != possible_trips:
            del total_length_round_trips[n]
    return dict([(v, k) for k, v in total_length_round_trips.items()])[min(dict([(v, k) for k, v in total_length_round_trips.items()]))] # finds smallest sum of path lengths and return this node, the most central node
    






if __name__ == "__main__":
    # You can place code (like custom test cases) here that will only be
    # executed when running this file from the terminal.
    pass

    # factory1 = GraphFactory(.2)
    # factory2 = GraphFactory(1)
    # edges = [
    #             ('Daphne', 'Fred', 1),
    #             ('Daphne', 'Scooby', 5),
    #             ('Daphne', 'Shaggy', 7),
    #             ('Daphne', 'Velma', 3),
    #             ('Fred', 'Daphne', 1),
    #             ('Fred', 'Scooby', 5),
    #             ('Fred', 'Shaggy', 7),
    #             ('Fred', 'Velma', 9),
    #             ('Scooby', 'Daphne', 5),
    #             ('Scooby', 'Fred', 5),
    #             ('Scooby', 'Shaggy', 0),
    #             ('Scooby', 'Velma', 3),
    #             ('Shaggy', 'Daphne', 7),
    #             ('Shaggy', 'Fred', 7),
    #             ('Shaggy', 'Scooby', 0),
    #             ('Shaggy', 'Velma', 6),
    #             ('Velma', 'Daphne', 3),
    #             ('Velma', 'Fred', 9),
    #             ('Velma', 'Scooby', 3),
    #             ('Velma', 'Shaggy', 6),
    #         ]
    # nodes = ["Shaggy", "Scooby", "Daphne", "Velma", "Fred"]
    # graph1 = factory1.from_edges_and_nodes(edges, nodes) # AdjacencyMatrixGraph object
    # graph2 = factory2.from_edges_and_nodes(edges, nodes) # AdjacencyDictGraph object
    # start1 = datetime.now()
    # print(get_most_central_node(graph1))
    # end1 = datetime.now()
    # print('time for matrix:')
    # print(end1 - start1)
    # start2 = datetime.now()
    # print(get_most_central_node(graph2))
    # end2 = datetime.now()
    # print('time for dict:')
    # print(end2 - start2)

    
