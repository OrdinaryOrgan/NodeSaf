from deception.nodeSaf import NodeSaf
from igraph import Graph


INPUT_SETTINGS = {
    'path': 'datasets/pol.gml',  # dol.gml les.gml pol.gml mad.edgelist
}

if __name__ == '__main__':
    g = Graph.Read_GML(INPUT_SETTINGS['path'])
    # g = Graph.Read_Edgelist(INPUT_SETTINGS['path'])
    # g = Graph.Famous('Thomassen')  # Coxeter Zachary Meredith Thomassen
    example1 = NodeSaf(graph = g, target_partitions_index = [0], budget = 100, rounds = 5)
    example1.run()
