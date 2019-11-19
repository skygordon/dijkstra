import unittest
import lab


class Graphs:
    """Graphs used for testing."""

    class Simple:
        """Most basic example."""

        def __new__(cls, graph_class):
            graph = graph_class()
            [graph.add_node(n) for n in 'abcx']
            graph.add_edge('a', 'b', 3)
            graph.add_edge('b', 'c', 1)
            graph.add_edge('c', 'a', 2.2)
            return graph

        path_lengths = {
            ('a', 'a'): 0, ('a', 'b'): 3, ('a', 'c'): 4,
            ('b', 'a'): 3.2, ('b', 'c'): 1, ('b', 'b'): 0,
            ('c', 'a'): 2.2, ('c', 'b'): 5.2, ('c', 'c'): 0,
            ('x', 'x'): 0,
        }

    class Triangle:
        """Simple triangle."""

        def __new__(cls, graph_class):
            graph = graph_class()
            [graph.add_node(n) for n in 'abc']
            graph.add_edge('a', 'b', 3)
            graph.add_edge('b', 'c', 1)
            graph.add_edge('c', 'a', 2.2)
            return graph

    class WeightedTriangle:
        """Bidirectional triangle with weights."""

        def __new__(cls, graph_class):
            graph = graph_class()
            [graph.add_node(n) for n in 'abc']
            graph.add_edge('a', 'b', 9999)
            graph.add_edge('b', 'a', 0)
            graph.add_edge('b', 'c', 1)
            graph.add_edge('c', 'b', 2.2)
            graph.add_edge('a', 'c', 2.2)
            graph.add_edge('c', 'a', 2.2)
            return graph

        path_lengths = {
            ('a', 'a'): 0, ('a', 'b'): 4.4, ('a', 'c'): 2.2,
            ('b', 'a'): 0, ('b', 'b'): 0, ('b', 'c'): 1,
            ('c', 'a'): 2.2, ('c', 'b'): 2.2, ('c', 'c'): 0,
        }

    class Line:
        """Linear graph."""

        def __new__(cls, graph_class):
            graph = graph_class()
            [graph.add_node(n) for n in 'abc']
            graph.add_edge('a', 'b', 1)
            graph.add_edge('b', 'a', 1)
            graph.add_edge('b', 'c', 1)
            graph.add_edge('c', 'b', 1)
            return graph

    class Star:
        """Bidirectional graph with one node in the center."""

        def __new__(cls, graph_class):
            graph = graph_class()
            [graph.add_node(n) for n in 'abcd']
            graph.add_edge('a', 'b', 1)
            graph.add_edge('b', 'a', 1)
            graph.add_edge('a', 'c', 1)
            graph.add_edge('c', 'a', 1)
            graph.add_edge('a', 'd', 1)
            graph.add_edge('d', 'a', 1)
            return graph

        path_lengths = {
            ('a', 'a'): 0, ('a', 'b'): 1, ('a', 'c'): 1,
            ('a', 'd'): 1, ('b', 'a'): 1, ('b', 'b'): 0,
            ('b', 'c'): 2, ('b', 'd'): 2, ('c', 'a'): 1,
            ('c', 'b'): 2, ('c', 'c'): 0, ('c', 'd'): 2,
            ('d', 'a'): 1, ('d', 'b'): 2, ('d', 'c'): 2,
            ('d', 'd'): 0,
        }

    class Dumbbell:
        """Non-symmetric, weighted, dumbbell-shaped graph."""

        def __new__(cls, graph_class):
            graph = graph_class()
            [graph.add_node(n) for n in 'abcdefg']
            graph.add_edge('a', 'b', 1)
            graph.add_edge('b', 'a', 1)
            graph.add_edge('b', 'c', 1)
            graph.add_edge('c', 'b', 1)
            graph.add_edge('b', 'd', 1)
            graph.add_edge('d', 'b', 1)
            graph.add_edge('a', 'e', 1)
            graph.add_edge('e', 'a', 1)
            graph.add_edge('a', 'f', 4)
            graph.add_edge('f', 'a', 4)
            graph.add_edge('g', 'f', 5)
            graph.add_edge('f', 'g', 3)
            return graph

    class Diamond:
        """Diamond-shaped graph with some extra edges."""

        def __new__(cls, graph_class):
            graph = graph_class()
            [graph.add_node(n) for n in 'abcde']
            graph.add_edge('a', 'b', 1)
            graph.add_edge('b', 'a', 1)
            graph.add_edge('a', 'd', 1)
            graph.add_edge('d', 'a', 1)
            graph.add_edge('d', 'b', 1)
            graph.add_edge('b', 'd', 1)
            graph.add_edge('d', 'c', 1)
            graph.add_edge('c', 'd', 1)
            graph.add_edge('b', 'c', 1)
            graph.add_edge('c', 'b', 1)
            graph.add_edge('b', 'e', 1)
            graph.add_edge('e', 'b', 1)
            return graph

        path_lengths = {
            ('a', 'b'): 1, ('a', 'c'): 2, ('a', 'd'): 1, ('a', 'e'): 2,
            ('b', 'a'): 1, ('b', 'c'): 1, ('b', 'd'): 1, ('b', 'e'): 1,
            ('c', 'a'): 2, ('c', 'b'): 1, ('c', 'd'): 1, ('c', 'e'): 2,
            ('d', 'a'): 1, ('d', 'b'): 1, ('d', 'c'): 1, ('d', 'e'): 2,
            ('e', 'a'): 2, ('e', 'b'): 1, ('e', 'c'): 2, ('e', 'd'): 2,
            ('a', 'a'): 0, ('b', 'b'): 0, ('c', 'c'): 0, ('d', 'd'): 0,
            ('e', 'e'): 0,
        }

    class WeightedDiamond:
        """Diamond-shaped weighted graph with some extra edges."""

        def __new__(cls, graph_class):
            graph = graph_class()
            [graph.add_node(n) for n in 'abcde']
            graph.add_edge('a', 'b', 2)
            graph.add_edge('b', 'a', 1.5)
            graph.add_edge('a', 'd', 1.5)
            graph.add_edge('d', 'a', 1.5)
            graph.add_edge('d', 'b', 4)
            graph.add_edge('b', 'd', 4)
            graph.add_edge('d', 'c', 5)
            graph.add_edge('c', 'd', 1)
            graph.add_edge('b', 'c', 99)
            graph.add_edge('c', 'b', 95)
            graph.add_edge('b', 'e', 1)
            graph.add_edge('e', 'b', 4)
            return graph

        path_lengths = {
            ('a', 'b'): 2, ('a', 'c'): 6.5, ('a', 'd'): 1.5, ('a', 'e'): 3,
            ('b', 'a'): 1.5, ('b', 'c'): 8, ('b', 'd'): 3, ('b', 'e'): 1,
            ('c', 'a'): 2.5, ('c', 'b'): 4.5, ('c', 'd'): 1, ('c', 'e'): 5.5,
            ('d', 'a'): 1.5, ('d', 'b'): 3.5, ('d', 'c'): 5, ('d', 'e'): 4.5,
            ('e', 'a'): 5.5, ('e', 'b'): 4, ('e', 'c'): 12, ('e', 'd'): 7,
            ('a', 'a'): 0, ('b', 'b'): 0, ('c', 'c'): 0, ('d', 'd'): 0,
            ('e', 'e'): 0,
        }

    class ZeroEdged:
        """Graph containing edges of length zero."""

        def __new__(cls, graph_class):
            graph = graph_class()
            [graph.add_node(f'node{i}') for i in range(1, 5)]
            graph.add_edge('node1', 'node2', 0)
            graph.add_edge('node2', 'node3', 0)
            graph.add_edge('node3', 'node4', 0)
            graph.add_edge('node3', 'node1', 1)
            return graph

        path_lengths = {
            ('node1', 'node1'): 0,
            ('node1', 'node2'): 0,
            ('node1', 'node3'): 0,
            ('node1', 'node4'): 0,
            ('node2', 'node1'): 1,
            ('node2', 'node2'): 0,
            ('node2', 'node3'): 0,
            ('node2', 'node4'): 0,
            ('node3', 'node1'): 1,
            ('node3', 'node2'): 1,
            ('node3', 'node3'): 0,
            ('node3', 'node4'): 0,
            ('node4', 'node4'): 0,
        }

    class Acyclic:
        """Weighted acyclic graph from the write-up (Figure 1)."""

        def __new__(cls, graph_class):
            graph = graph_class()
            [graph.add_node(n) for n in 'abcdefg']
            graph.add_edge('a', 'b', 3.1)
            graph.add_edge('b', 'c', 1)
            graph.add_edge('b', 'd', 0.5)
            graph.add_edge('g', 'd', 5)
            graph.add_edge('b', 'e', 4)
            graph.add_edge('c', 'e', 0.5)
            graph.add_edge('d', 'e', 2.2)
            graph.add_edge('e', 'f', 6)
            return graph

        path_lengths = {
            ('a', 'b'): 3.1, ('a', 'c'): 4.1, ('a', 'd'): 3.6, ('a', 'e'): 4.6,
            ('a', 'f'): 10.6, ('b', 'c'): 1, ('b', 'd'): 0.5, ('b', 'e'): 1.5,
            ('b', 'f'): 7.5, ('c', 'e'): 0.5, ('c', 'f'): 6.5, ('d', 'e'): 2.2,
            ('d', 'f'): 8.2, ('e', 'f'): 6, ('g', 'd'): 5, ('g', 'e'): 7.2,
            ('g', 'f'): 13.2, ('a', 'a'): 0, ('b', 'b'): 0, ('c', 'c'): 0,
            ('d', 'd'): 0, ('e', 'e'): 0, ('f', 'f'): 0, ('g', 'g'): 0,
        }

    class Cyclic:
        """Graph containing a cycle."""

        def __new__(cls, graph_class):
            graph = graph_class()
            nodes = "ABCD"
            for node in nodes:
                graph.add_node(node)
            for i in range(len(nodes)):
                graph.add_edge(nodes[i], nodes[(i + 1) % len(nodes)], i + 1)
            return graph

    class Complete:
        """Complete graph."""

        def __new__(cls, graph_class):
            graph = graph_class()
            nodes = "abcdefghijkl"
            for node in nodes:
                graph.add_node(node)
            for node1 in nodes:
                for node2 in set(nodes) - {node1}:
                    graph.add_edge(node1, node2, 1)
            return graph

    class Large:
        """A larger, relatively dense graph."""

        def __new__(cls, graph_class):
            graph = graph_class()
            for i in range(100):
                graph.add_node(str(i))
            for i in range(99):
                for j in range(i + 1, 99):
                    graph.add_edge(str(i), str(j), 1)
            return graph


class GraphTests:
    class TestBasic(unittest.TestCase):
        def test_01(self):
            graph = self.graph_class()
            self.assertEqual(graph.nodes(), set())
            graph.add_node("A")
            with self.assertRaises(ValueError):
                graph.add_node("A")
            graph.add_node("B")
            self.assertEqual(graph.nodes(), {"A", "B"})
            graph.add_node("")
            self.assertEqual(graph.nodes(), {"A", "B", ""})

        def test_02(self):
            graph = self.graph_class()
            graph.add_node("A")
            with self.assertRaises(LookupError):
                graph.add_edge("A", "B", 3)
            graph.add_node("B")
            graph.add_edge("A", "B", 3)
            self.assertEqual(graph.neighbors("A"), {("B", 3)})
            self.assertEqual(graph.neighbors("B"), set())
            graph.add_edge("A", "B", 5)
            self.assertEqual(graph.neighbors("A"), {("B", 5)})
            graph.add_node("C")
            self.assertEqual(graph.nodes(), {"A", "B", "C"})
            graph.add_edge("A", "C", 10)
            self.assertEqual(graph.neighbors("A"), {("B", 5), ("C", 10)})

        def test_03(self):
            graph = Graphs.Complete(self.graph_class)
            expected_nodes = set("abcdefghijkl")
            nodes = graph.nodes()
            self.assertEqual(nodes, expected_nodes)
            for node in nodes:
                expected_neighbors = {(neighbor, 1) for neighbor in nodes
                                      if neighbor != node}
                self.assertEqual(graph.neighbors(node), expected_neighbors)

        def test_04(self):
            graph = Graphs.Cyclic(self.graph_class)
            self.assertEqual(graph.nodes(), {"A", "B", "C", "D", "A"})
            self.assertEqual(graph.neighbors("A"), {("B", 1)})
            self.assertEqual(graph.neighbors("B"), {("C", 2)})
            self.assertEqual(graph.neighbors("C"), {("D", 3)})
            self.assertEqual(graph.neighbors("D"), {("A", 4)})

    class TestPath(unittest.TestCase):
        def _validate_all_lengths(self, producer):
            graph = producer(self.graph_class)
            result = graph.get_all_path_lengths()
            self.assertEqual(producer.path_lengths, result)

        def _validate_path(self, path, graph, target_length):
            self.assertIsNotNone(path)
            length = 0
            current_index = 0
            while current_index < len(path) - 1:
                current_node = path[current_index]
                for next_node, edge_weight in graph.neighbors(current_node):
                    if next_node == path[current_index + 1]:
                        length += edge_weight
                        break
                else:
                    self.fail(f"Wrong path, node {path[current_index + 1]} is "
                              f"not connected to node {current_node}")
                current_index += 1
            if length != target_length:
                self.fail(f"Expected shortest path with length {target_length}"
                          f", got length {length} instead")

        def _validate_all_paths(self, producer):
            graph = producer(self.graph_class)
            result = graph.get_all_paths()
            self.assertEqual(len(producer.path_lengths), len(result))
            for pair, length in producer.path_lengths.items():
                self.assertIn(pair, result)
                self._validate_path(result[pair], graph, length)

        def test_01_get_path_length(self):
            graph = Graphs.Simple(self.graph_class)
            # 0 edges
            result = graph.get_path_length('a', 'a')
            expected = 0
            self.assertEqual(expected, result, "Wrong path length")
            # 1 edge
            result = graph.get_path_length('b', 'c')
            expected = 1
            self.assertEqual(expected, result, "Wrong path length")
            # 2 edges
            result = graph.get_path_length('a', 'c')
            expected = 4
            self.assertEqual(expected, result, "Wrong path length")

        def test_02_get_path_length(self):
            graph = Graphs.ZeroEdged(self.graph_class)
            for i, j in [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]:
                self.assertEqual(
                    0, graph.get_path_length(f"node{i}", f"node{j}"),
                    "Wrong path length for 0-weight path")

        def test_03_get_path_length(self):
            graph = Graphs.Simple(self.graph_class)
            for node in ["a", "b", "c"]:
                result = graph.get_path_length(node, "x")
                self.assertIsNone(result, "Expected no path")

        def test_04_get_path_length(self):
            graph = Graphs.Simple(self.graph_class)
            with self.assertRaises(LookupError):
                graph.get_path_length("1", "a")

            with self.assertRaises(LookupError):
                graph.get_path_length("b", "node1")

            with self.assertRaises(LookupError):
                graph.get_path_length("A", "B")

        def test_05_get_path_length(self):
            graph = Graphs.Acyclic(self.graph_class)
            msg = "Wrong path length"
            self.assertEqual(0.5, graph.get_path_length("c", "e"), msg)
            self.assertEqual(1.5, graph.get_path_length("b", "e"), msg)
            self.assertEqual(10.6, graph.get_path_length("a", "f"), msg)

        def test_06_get_path(self):
            graph = Graphs.Simple(self.graph_class)
            # 1 edge
            result = graph.get_path('c', 'a')
            expected = ['c', 'a']
            self.assertEqual(expected, result)
            # 2 edges
            result = graph.get_path('a', 'c')
            expected = ['a', 'b', 'c']
            self.assertEqual(expected, result)

        def test_07_get_path(self):
            graph = Graphs.ZeroEdged(self.graph_class)
            for node in ["node1", "node2", "node3"]:
                self.assertIsNone(graph.get_path("node4", node),
                                  "Expected no path")

        def test_08_get_path(self):
            graph = Graphs.Star(self.graph_class)
            with self.assertRaises(LookupError):
                graph.get_path("a", "1")

            with self.assertRaises(LookupError):
                graph.get_path("a", "1")

        def test_09_get_path(self):
            graph = Graphs.ZeroEdged(self.graph_class)
            self.assertEqual(["node1", "node2", "node3", "node4"],
                             graph.get_path("node1", "node4"))

        def test_10_get_path(self):
            graph = Graphs.Acyclic(self.graph_class)
            # 2 edges
            self.assertEqual(['b', 'c', 'e'], graph.get_path('b', 'e'))
            # 3 edges
            self.assertEqual(['g', 'd', 'e', 'f'], graph.get_path('g', 'f'))
            # 4 edges
            self.assertEqual(['a', 'b', 'c', 'e', 'f'],
                             graph.get_path('a', 'f'))

        def test_11_get_path(self):
            graph = Graphs.Diamond(self.graph_class)
            pairs = [('a', 'c'), ('c', 'a'), ('e', 'd'), ('d', 'e')]
            for source, target in pairs:
                result = graph.get_path(source, target)
                self._validate_path(result, graph, 2)

        def test_12_get_path(self):
            graph = Graphs.WeightedDiamond(self.graph_class)
            pairs = [('a', 'c'), ('c', 'a'), ('e', 'd'), ('d', 'e')]
            for source, target in pairs:
                result = graph.get_path(source, target)
                self._validate_path(
                    result, graph,
                    Graphs.WeightedDiamond.path_lengths[source, target])

        def test_13_get_all_path_lengths(self):
            self._validate_all_lengths(Graphs.ZeroEdged)

        def test_14_get_all_path_lengths(self):
            self._validate_all_lengths(Graphs.WeightedTriangle)

        def test_15_get_all_path_lengths(self):
            self._validate_all_lengths(Graphs.Diamond)

        def test_16_get_all_path_lengths(self):
            self._validate_all_lengths(Graphs.WeightedDiamond)

        def test_17_get_all_path_lengths(self):
            self._validate_all_lengths(Graphs.Acyclic)

        def test_18_get_all_paths(self):
            self._validate_all_paths(Graphs.Simple)

        def test_19_get_all_paths(self):
            self._validate_all_paths(Graphs.ZeroEdged)

        def test_20_get_all_paths(self):
            self._validate_all_paths(Graphs.Star)

        def test_21_get_all_paths(self):
            self._validate_all_paths(Graphs.WeightedTriangle)

        def test_22_get_all_paths(self):
            self._validate_all_paths(Graphs.Diamond)

        def test_23_get_all_paths(self):
            self._validate_all_paths(Graphs.WeightedDiamond)

        def test_24_get_all_paths(self):
            self._validate_all_paths(Graphs.Acyclic)

        def test_25_get_all_paths(self):
            graph = Graphs.Complete(self.graph_class)
            nodes = "abcdefghijkl"
            result = graph.get_all_paths()
            self.assertEqual(len(nodes) ** 2, len(result))
            for start in nodes:
                for end in nodes:
                    self.assertIn((start, end), result)
                    if start == end:
                        self.assertEqual([start], result[(start, end)])
                    else:
                        self.assertEqual([start, end], result[(start, end)])

        def test_26_get_all_paths(self):
            graph = Graphs.Large(self.graph_class)
            expected = {(str(i), str(i)): 0 for i in range(100)}
            for i in range(99):
                for j in range(i, 99):
                    if i != j:
                        expected[(str(i), str(j))] = 1
            self.assertEqual(expected, graph.get_all_path_lengths())

    class TestCentralNode(unittest.TestCase):
        def test_01(self):
            graph = Graphs.Star(self.graph_class)
            result = lab.get_most_central_node(graph)
            expected = 'a'
            self.assertEqual(expected, result)

        def test_02(self):
            graph = Graphs.Diamond(self.graph_class)
            result = lab.get_most_central_node(graph)
            expected = 'b'
            self.assertEqual(expected, result)

        def test_03(self):
            graph = Graphs.Line(self.graph_class)
            result = lab.get_most_central_node(graph)
            expected = 'b'
            self.assertEqual(expected, result)

        def test_04(self):
            graph = Graphs.WeightedDiamond(self.graph_class)
            result = lab.get_most_central_node(graph)
            expected = 'a'
            self.assertEqual(expected, result)

        def test_05(self):
            graph = Graphs.Dumbbell(self.graph_class)
            result = lab.get_most_central_node(graph)
            expected = 'a'
            self.assertEqual(expected, result)

        def test_06(self):
            graph = Graphs.WeightedTriangle(self.graph_class)
            result = lab.get_most_central_node(graph)
            expected_possibilities = {'b', 'c'}
            self.assertIn(result, expected_possibilities)


class Test_1_AdjacencyDictBasic(GraphTests.TestBasic):
    graph_class = lab.AdjacencyDictGraph


class Test_2_AdjacencyMatrixBasic(GraphTests.TestBasic):
    graph_class = lab.AdjacencyMatrixGraph


class Test_3_AdjacencyDictPaths(GraphTests.TestPath):
    graph_class = lab.AdjacencyDictGraph


class Test_4_AdjacencyMatrixPaths(GraphTests.TestPath):
    graph_class = lab.AdjacencyMatrixGraph


class Test_5_Factory(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix_class = lab.AdjacencyMatrixGraph
        self.dict_class = lab.AdjacencyDictGraph

    def test_01(self):
        cutoffs = {0.1: self.matrix_class,
                   0.25: self.dict_class,
                   0.4: self.dict_class}

        for cutoff, expected_class in cutoffs.items():
            factory = lab.GraphFactory(cutoff)
            edges = [("A", "B", 2), ("A", "C", 0), ("A", "D", 4)]
            nodes = ["A", "B", "C", "D"]
            graph = factory.from_edges_and_nodes(edges, nodes)
            self.assertIsInstance(graph, expected_class)
            self.assertEqual({"A", "B", "C", "D"}, graph.nodes(),
                             msg=f"failed on {expected_class}")
            self.assertEqual({("B", 2), ("C", 0), ("D", 4)},
                             graph.neighbors("A"),
                             msg=f"failed on {expected_class}")

    def test_02(self):
        cutoffs = {0: self.dict_class,
                   1.0: self.dict_class}
        for cutoff, expected_class in cutoffs.items():
            factory = lab.GraphFactory(cutoff)
            edges = []
            nodes = ["123", "456", "789"]
            graph = factory.from_edges_and_nodes(edges, nodes)
            self.assertIsInstance(graph, expected_class)
            self.assertEqual({"123", "456", "789"}, graph.nodes())
            self.assertEqual(set(), graph.neighbors("123"))

    def test_03(self):
        cutoffs = {0: self.matrix_class,
                   0.8: self.matrix_class,
                   1.0: self.dict_class}
        for cutoff, expected_class in cutoffs.items():
            factory = lab.GraphFactory(cutoff)
            edges = [
                ('Daphne', 'Fred', 1),
                ('Daphne', 'Scooby', 5),
                ('Daphne', 'Shaggy', 7),
                ('Daphne', 'Velma', 3),
                ('Fred', 'Daphne', 1),
                ('Fred', 'Scooby', 5),
                ('Fred', 'Shaggy', 7),
                ('Fred', 'Velma', 9),
                ('Scooby', 'Daphne', 5),
                ('Scooby', 'Fred', 5),
                ('Scooby', 'Shaggy', 0),
                ('Scooby', 'Velma', 3),
                ('Shaggy', 'Daphne', 7),
                ('Shaggy', 'Fred', 7),
                ('Shaggy', 'Scooby', 0),
                ('Shaggy', 'Velma', 6),
                ('Velma', 'Daphne', 3),
                ('Velma', 'Fred', 9),
                ('Velma', 'Scooby', 3),
                ('Velma', 'Shaggy', 6),
            ]
            nodes = ["Shaggy", "Scooby", "Daphne", "Velma", "Fred"]
            graph = factory.from_edges_and_nodes(edges, nodes)
            expected_nodes = {"Shaggy", "Scooby", "Daphne", "Velma", "Fred"}
            expected_neighbors = {("Scooby", 0), ("Velma", 6), ("Daphne", 7),
                                  ("Fred", 7)}
            self.assertIsInstance(graph, expected_class)
            self.assertEqual(expected_nodes, graph.nodes())
            self.assertEqual(expected_neighbors, graph.neighbors("Shaggy"))

    def test_04(self):
        cutoffs = {0: self.matrix_class,
                   0.99: self.matrix_class,
                   1.0: self.dict_class}
        for cutoff, expected_class in cutoffs.items():
            factory = lab.GraphFactory(cutoff)
            edges = [
                ('boston', 'chicago', 91238.21),
                ('chicago', 'boston', 91238.21),
            ]
            nodes = ["boston", "chicago"]
            graph = factory.from_edges_and_nodes(edges, nodes)
            self.assertIsInstance(graph, expected_class)
            self.assertEqual({"boston", "chicago"}, graph.nodes())


class Test_6_CentralNodeDict(GraphTests.TestCentralNode):
    graph_class = lab.AdjacencyDictGraph


class Test_7_CentralNodeMatrix(GraphTests.TestCentralNode):
    graph_class = lab.AdjacencyMatrixGraph


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
