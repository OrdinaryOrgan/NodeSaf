import igraph
import time


class NodeSaf(object):
    def __init__(self, graph: igraph.Graph, target_partitions_index: list, budget: int, rounds: int):
        self.graph = graph
        self.true_partitions = None  # Community Structure: actually a list
        self.target_partitions_index = target_partitions_index  # list of int: list of communities index
        self.budget = budget
        self.rounds = rounds
        self.partitions_num = 0  # Number of communities
        self.target_comm_list = []  # list of target communities: list of list
        self.target_node_index_list = []  # list of int
        self.count = 0
        self.deception_score = 0
        self.nmi_list = []
        self.start_time = time.process_time()

    # To check the targets: all or some
    def preprocess(self):
        print("Start NodeSBM: Based on Safeness")
        print("Method: preprocess")
        self.true_partitions = self.reCalCommStructure(self.graph)
        self.partitions_num = len(self.true_partitions)
        if len(self.target_partitions_index) == self.partitions_num:
            # Decept all communities
            self.target_node_index_list = list(range(self.graph.vcount()))
            self.target_comm_list = list(self.true_partitions)
        else:
            # Decept some communities
            for index in self.target_partitions_index:
                self.target_node_index_list.extend(self.true_partitions[index])
                self.target_comm_list.append(self.true_partitions[index])
        # print(f"Target Node List: {self.target_node_index_list}")
        # print(f"Target Community List: {self.target_comm_list}")
        count = 0
        for v in self.graph.vs:
            v["original_index"] = count
            count += 1
        print(f"Graph: {self.graph}")
        print(f"Partitions: {self.true_partitions}")
        init_partitions_num = len(self.true_partitions)
        print(f"Initial partitions num: {init_partitions_num}\n")
        # deception_score = self.getDeceptionScore(after_updates = False)
        # self.deception_score = deception_score
        # print(f"Deception Score: {deception_score}")

    # Get the number of reachable vertices
    def getReachableNum(self, node_index: int, comm: list, graph: igraph.Graph):
        # print("Method: getReachableNum")
        # print(f"Node: {node_index}, Comm: {comm}, Graph: {graph}")
        comm_graph = graph.induced_subgraph(comm)  # comm_graph: igraph.Graph
        new_node_index = 0
        for index in range(comm_graph.vcount()):
            node = comm_graph.vs[index]
            if node["original_index"] == node_index:
                new_node_index = index
                break
        diameter = comm_graph.diameter()
        neighbors_num = comm_graph.neighborhood_size(new_node_index, order = diameter)
        return neighbors_num

    # Get inter and intra edge list
    def getEdgelist(self, node_index: int, comm: list, graph: igraph.Graph):
        # print("Method: getEdgelist")
        node = graph.vs[node_index]
        node_edges = node.incident()  # node_edges: list of igraph.Edge
        # print(f"Input Arguments:")
        # print(f"Node index: {node_index}")
        # print(f"Community: {comm}")
        # node_edges = graph.incident(node_index, mode = 'all')
        intraEdgeList = []
        interEdgeList = []
        for edge in node_edges:
            # print(f"{edge.source, edge.target}")
            if (edge.target in comm) and (edge.source in comm):  # Intra edges
                intraEdgeList.append(edge)
            else:  # Inter Edges
                interEdgeList.append(edge)
        return intraEdgeList, interEdgeList

    # Get node safeness
    def getNodeSafeness(self, node_index: int, comm: list, graph: igraph.Graph, tau = 0.5):
        # print("Method: getNodeSafeness")
        if graph.degree(node_index, mode = 'all') == 0 or len(comm) == 1:
            return 0
        # print(f"Node: {node_index}, Comm: {comm}, Graph: {graph}")
        intraEdgeList, interEdgeList = self.getEdgelist(node_index, comm, graph)
        # print(f"Num of intraEdgeList: {len(intraEdgeList)}, num of interEdgeList: {len(interEdgeList)}")
        result = float(tau * (self.getReachableNum(node_index, comm, graph) - len(intraEdgeList)) / (len(comm) - 1))
        # print(result)
        result = result + float((1 - tau) * len(interEdgeList) / graph.degree(node_index, mode = 'all'))
        return result
    # Get list of node safeness

    def getNodeSafenessList(self, comm: list, graph: igraph.Graph):
        print("Method: getNodeSafenessList")
        node_safeness_list = []
        for node in comm:
            node_safeness_list.append(self.getNodeSafeness(node, comm, graph))
        print(f"Node safeness list: {node_safeness_list}")
        return node_safeness_list

    # Get community safeness
    def getCommSafeness(self, comm: list, graph: igraph.Graph):
        print("Method: getCommSafeness")
        res = 0
        for node in comm:
            res += self.getNodeSafeness(node, comm, graph, tau = 0.2)
        res = float(res / len(comm))
        print(f"Community safeness: {res}")
        return res

    # Get the delta of deletion
    def getDeltaDeletion(self, node_to_del: int, comm: list, graph: igraph.Graph):
        print("Method: getDeltaDeletion")
        if len(comm) <= 1:
            return -1
        comm_after_del = comm.copy()
        comm_after_del.remove(node_to_del)
        graph_after_del = graph.copy()
        # Cost = number of edges to delete
        node = graph.vs[node_to_del]
        node_edges = node.incident()  # node_edges: list of igraph.Edge
        # node_edges = graph.incident(node_to_del, mode = 'all')
        cost = len(node_edges)
        graph_after_del.delete_vertices(node_to_del)
        new_comm = []
        for original_index in comm_after_del:
            for index in range(graph_after_del.vcount()):
                node = graph_after_del.vs[index]
                if node["original_index"] == original_index:
                    new_comm.append(index)
        delta = self.getCommSafeness(new_comm, graph_after_del) - self.getCommSafeness(comm, graph)
        return delta, cost

    def getDeltaAdditionWithEdges(self, comm: list, graph: igraph.Graph, edgelist: list):
        print("Method: getDeltaAdditionWithEdges")
        graph_after_add = graph.copy()
        graph_after_add.add_vertex()
        # Get the index of added node
        node_index = len(graph_after_add.vs) - 1
        graph_after_add.add_edges(edgelist)
        comm_after_add = comm.copy()
        comm_after_add.append(node_index)
        print(f"Comm after add: {comm_after_add}")
        # print(f"Graph after add: {graph_after_add}")
        delta = self.getCommSafeness(comm_after_add, graph_after_add) - self.getCommSafeness(comm, graph)
        cost = len(edgelist)
        return delta, cost

    def chooseNpNode(self, comm: list, graph: igraph.Graph):
        print("Method: chooseNpNode")
        ratiolist = []
        for node in comm:
            intraEdgeList, interEdgeList = self.getEdgelist(node, comm, graph)
            # intraEdgeCount = len(intraEdgeList)
            interEdgeCount = len(interEdgeList)
            if graph.degree(node) == 0:
                ratio = 0
            else:
                ratio = interEdgeCount / graph.degree(node)
            # print(f"Node: {node}, InterEdgeCount: {interEdgeCount}, IntraEdgeCount: {intraEdgeCount}, Ratio: {ratio}")
            ratiolist.append(ratio)
        print(f"Ratio list: {ratiolist}")
        np_index = comm[ratiolist.index(max(ratiolist))]
        return np_index

    # Get the index of community which has most inter edge with given vertex
    def getInterEdgeCommIndex(self, node_index: int, comm: list, comm_list: list, graph: igraph.Graph):
        print("Method: getInterEdgeCommIndex")
        node = graph.vs[node_index]
        node_edges = node.incident()  # node_edges: list of igraph.Edge
        # node_edges = graph.incident(node_index, mode = 'all')
        interEdgeCountList = [0] * self.partitions_num  # List of count
        for edge in node_edges:
            if (edge.target not in comm) or (edge.source not in comm):  # Inter Edge
                if edge.target not in comm:
                    for index in range(self.partitions_num):  # Go through all communities
                        if edge.target in comm_list[index]:  # Target Community
                            interEdgeCountList[index] += 1
                elif edge.source not in comm:
                    for index in range(self.partitions_num):
                        if edge.source in comm_list[index]:
                            interEdgeCountList[index] += 1
        # print(interEdgeCountList)
        max_index = interEdgeCountList.index(max(interEdgeCountList))
        print(f"Index of comm to add edges: {max_index}")
        return max_index

    def getDeceptionScore(self, after_updates):
        comm_num = len(self.true_partitions)
        if after_updates is False:
            community = self.target_comm_list
        else:
            community = self.target_comm_list
            community[:] = (value for value in community if value != -1)
        # print(f"Comm list: {community}")
        member_for_community = []
        for member in community:
            current_community_member = [1 if member == community else 0 for community in self.true_partitions]
            # print(f"True partitions: {self.true_partitions}")
            member_for_community.append(current_community_member)
        # print(f"Member for community: {member_for_community}")
        member_for_community = [sum(x) for x in zip(*member_for_community)]
        # print(f"Member for community: {member_for_community}")
        ratio_community_members = [members_for_c / len(com) for (members_for_c, com) in
                                   zip(member_for_community, self.true_partitions)]
        # print(f"Ratio_Comm: {ratio_community_members}")
        spread_members = sum([1 if value_per_com > 0 else 0 for value_per_com in ratio_community_members])
        # print(f"Spread members: {spread_members}")
        second_part = 1 / 2 * ((spread_members - 1) / comm_num) + 1 / 2 * (
                1 - sum(ratio_community_members) / spread_members)
        num_components = 0
        for node in self.target_node_index_list:
            # print(f"Node: {node}, Comm: {self.target_node_index_list}")
            num_components += self.getReachableNum(node, self.target_node_index_list, self.graph)
        first_part = 1 - ((num_components - 1) / (len(self.target_node_index_list) - 1))
        dec_score = first_part * second_part
        return dec_score * -1

    def reCalCommStructure(self, graph: igraph.Graph):
        return igraph.Graph.community_multilevel(graph)

    def nodeSafenessAlgo(self, comm: list, comm_list: list, graph: igraph.Graph):
        print("Method: nodeSafenessAlgo")
        # Node Addition
        np = self.chooseNpNode(comm, graph)
        print(f"Np node index: {np}")
        budget = self.budget
        new_graph = graph.copy()
        new_graph.add_vertex()
        new_graph_add = new_graph.copy()
        node_to_add = len(new_graph_add.vs) - 1
        print(f"Index of node to add: {node_to_add}")
        new_comm_add = comm.copy()
        new_comm_add.append(node_to_add)
        target_add_edge_comm = self.getInterEdgeCommIndex(np, comm, comm_list, graph)
        target_node_list = []
        for node in self.true_partitions[target_add_edge_comm]:
            target_node_list.append(node)
            budget -= 1
            if budget == 1:
                break
        target_node_list.append(np)
        # target_node_list.sort()
        print(f"Target node list to add edges: {target_node_list}")
        target_edge_list = []
        for node in target_node_list:
            target_edge_list.append((node_to_add, node))
        new_graph_add.add_edges(target_edge_list)
        delta_add, add_cost = self.getDeltaAdditionWithEdges(comm, graph, target_edge_list)
        print(f"Delta add: {delta_add}, Add cost: {add_cost}")
        add_budget_remain = budget - 1
        print(f"Add budget remain: {add_budget_remain}")
        # Node Deletion
        node_safeness_list = self.getNodeSafenessList(comm, graph)
        node_to_del = comm[node_safeness_list.index(min(node_safeness_list))]
        edge_to_del = graph.incident(node_to_del)
        print(f"Node to delete: {node_to_del}")
        # print(f"Edge to delete: {edge_to_del}")
        if len(comm) == 1:
            delta_del = 0
            del_cost = 0
        else:
            delta_del, del_cost = self.getDeltaDeletion(node_to_del, comm, graph)
        print(f"Delta delete: {delta_del}, Delete cost: {del_cost}")
        del_budget_remain = self.budget - del_cost - 1
        print(f"Delete budget remain: {del_budget_remain}")
        if delta_del > delta_add and delta_del > 0 and del_budget_remain >= 0:
            print("Choose node deletion")
            new_graph = graph.copy()
            new_graph.delete_edges(edge_to_del)
            formal_partitions = self.true_partitions
            print(f"Formal partitions: {formal_partitions}")
            new_partitions = self.reCalCommStructure(new_graph)
            print(f"New partitions: {new_partitions}")
            nmi = igraph.compare_communities(formal_partitions, new_partitions, method = 'nmi')
            print(f"NMI: {nmi}")
            self.nmi_list.append(nmi)
            # print(f"NMI list: {self.nmi_list}")
            new_graph_del = graph.copy()
            new_graph_del.delete_vertices(node_to_del)
            self.graph = new_graph_del
            self.budget = del_budget_remain
        else:
            if delta_add > 0 and add_budget_remain >= 0:
                print(f"Choose node addition")
                formal_partitions = self.reCalCommStructure(new_graph)
                print(formal_partitions)
                new_partitions = self.reCalCommStructure(new_graph_add)
                nmi = igraph.compare_communities(formal_partitions, new_partitions, method = 'nmi')
                print(f"NMI: {nmi}")
                self.nmi_list.append(nmi)
                # print(f"NMI list: {self.nmi_list}")
                self.graph = new_graph_add
                self.budget = add_budget_remain
            else:
                self.budget = -1

    def refresh(self):
        print("Method: refresh")
        self.true_partitions = self.reCalCommStructure(self.graph)
        self.partitions_num = len(self.true_partitions)
        self.count += 1
        if self.count >= self.partitions_num:
            self.count = 0
        self.target_comm_list.clear()
        print(f"Count: {self.count}, Num: {self.partitions_num}")
        # self.target_comm_list.append(self.true_partitions[self.target_partitions_index[self.count]])
        # self.target_node_index_list = self.true_partitions[self.target_partitions_index[self.count]]
        self.target_comm_list.append(self.true_partitions[self.count])
        self.target_node_index_list = self.true_partitions[self.count]

    def run(self):
        self.preprocess()
        while self.budget > 0:
            self.nodeSafenessAlgo(self.target_node_index_list, list(self.true_partitions), self.graph)
            self.refresh()
            self.printInfo()
        end_time = time.process_time()
        print(f"Algo time cost: {end_time - self.start_time}")

    def printInfo(self):
        print(f"Graph: {self.graph.summary()}")
        print(f"Partitions: {self.true_partitions}")
        print(f"Num of partitions: {len(self.true_partitions)}")
        # deception_score = self.getDeceptionScore(after_updates = False)
        # print(f"Deception Score: {deception_score}")
        # print(f"Delta of deception score: {deception_score - self.deception_score}")
        print(f"NMI list: {self.nmi_list}")
        print("End of print")
        print("--------------------------------------------------")
