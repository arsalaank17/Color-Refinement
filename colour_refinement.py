import copy
from graphs import *
from graph_io import load_graph, write_dot, save_graph
from graph import *
import os
import time
from collections import defaultdict

class ColourRefinement:
    def __init__(self, graph_set, indices=None, mode="GI", extra_aut = None):
        if mode == "GI":
            with open(graph_set) as f:
                self.graph_set = load_graph(f, read_list=True)  # graph set we are working with
        elif mode == "AUT":
            with open(graph_set) as f:
                graph = load_graph(f, read_list=False)  # graph set we are working with
            self.graph_set = ([graph, copy.deepcopy(graph)], [])
        elif mode == "GIAUT":
            with open(graph_set) as f:
                self.graph_set = load_graph(f, read_list=True)  # graph set we are working with
        elif mode == "EXTRA_AUT":
            with open(graph_set) as f:
                graph = load_graph(f, read_list=True)  # graph set we are working with
            # print (graph[0][extra_aut])
            self.graph_set = ([graph[0][extra_aut], copy.deepcopy(graph[0][extra_aut])], [])

        self.disjoint_graph = None  # disjoint union that we are working with
        self.coloring = None  # coloring of the disjoint union
        self.next_color = 2  # next color to be assigned
        self.vertex_count = len(self.graph_set[0][0])  # vertex count in a single graph
        self.indices = indices  # used if we are coloring specific vertices in graph set
        self.vertices_by_color = defaultdict(list)  # form: {color: [vertices of color]
        self.check_for_vertex_issues()

    def generate_dot(self, graph):
        """
        Generates .dot file of a given graph. Used for visualization.
        """
        with open('src/graphs/visualised/graph.dot', 'w') as f:
            write_dot(graph, f)

    def view_graph(self, graph, output_name):
        """
        Creates a .jpg file of a given graph. Used for visualisation.
        """
        self.generate_dot(graph)
        os.system("dot -Tjpg src/graphs/visualised/graph.dot -o src/graphs/visualised/" + output_name + ".jpg")

    def create_disjoint_union(self):
        """
        Used to combine graphs from the graph set into a disjoint union.
        Default parameter values will create a disjoint union of all graphs in the graph set file.
        If you want to make a disjoint union of specific graphs, set all_graphs to False, and indicate the indices of graphs
        to combine.
        """
        if not self.indices:
            disjoint_graph = self.graph_set[0][0]
            for i in range(1, len(self.graph_set[0])):
                disjoint_graph = disjoint_graph + self.graph_set[0][i]
            self.disjoint_graph = disjoint_graph
        else:
            disjoint_graph = self.graph_set[0][self.indices[0]]
            for i in range(1, len(self.indices)):
                disjoint_graph = disjoint_graph + self.graph_set[0][self.indices[i]]
            self.disjoint_graph = disjoint_graph

    def create_initial_colouring(self):
        """
        Generates initial coloring of the (disjoint) graph. (C(v) = 1 for all v in V).
        """
        for vertex in self.disjoint_graph.vertices:
            vertex.colornum = 1

        self.coloring = {v: 1 for v in self.disjoint_graph.vertices}
    
    def check_for_vertex_issues(self):
        """
        Fix of the BasicGIAut1 issue
        """
        # Check that all vertices are the same
        vertex_count_equal = True
        biggest_vertex_count = self.vertex_count

        for graph in self.graph_set[0]:
            if len(graph.vertices) != self.vertex_count:
                vertex_count_equal = False
                # print('here')
                if len(graph.vertices) > self.vertex_count:
                    biggest_vertex_count = len(graph.vertices)

        # Add disconnected vertices to smaller graphs (will not affect isomorphism)
        if not vertex_count_equal:
            # print('here now')
            if biggest_vertex_count != self.vertex_count:
                self.vertex_count = biggest_vertex_count
            for graph in self.graph_set[0]:
                if len(graph.vertices) != self.vertex_count:
                    for i in range(self.vertex_count - len(graph.vertices)):
                        graph.add_vertex(Vertex(graph))

        self.create_disjoint_union()

    def color_refinement(self, visuals=False, initialize=True):
        """
        Creates the stable coloring of the (disjoint) graph. The visuals option is disabled by default, if enabled, 
        prints out a bunch of debug messages as well as creates images of the coloring for each iteration.
        """
        if initialize:
            # Create initial coloring
            self.create_initial_colouring()
        if visuals:
            self.view_graph(self.disjoint_graph, "initialcoloring")
        iteration = 1
        old_coloring = {}
        # continue until we obtain stable coloring
        while self.coloring != old_coloring:
            # form: {color: [vertices of color]
            self.vertices_by_color = defaultdict(list)  # dictionary to keep track of vertices of each color easier
            for vertex, color in self.coloring.items():
                self.vertices_by_color[color].append(vertex)
            if visuals:
                print('Iteration ' + str(iteration) + '. Vertices by color: ')
                print('Coloring in the beginning of iteration is: ')
                print(self.coloring)
                for color, vertices in self.vertices_by_color.items():
                    print(str(color) + ": " + str(vertices))
            # for each color in our current coloring..
            for color in self.vertices_by_color:
                if visuals:
                    print('Working on color: ' + str(color) + '\n')
                neighbourhoods = defaultdict(dict)  # form: {vertex: vertex_neighbourhood_info}
                # ({vertex: {color: num_of_vertices_of_that_color}})
                for vertex in self.vertices_by_color[color]:
                    vertex_neighbours = vertex.neighbours  # get vertex's neighbours
                    # form: {Neighbour_of_v: color_of_neighbour}
                    neighbourhoods_coloring = {neighbour: self.coloring[neighbour] for neighbour in vertex_neighbours}
                    # form: {color: num_of_neighbours_of_vertex_of_color}
                    neighbourhood = defaultdict(int)
                    for neighbour, color_of_neighbour in neighbourhoods_coloring.items():
                        neighbourhood[color_of_neighbour] += 1
                    neighbourhoods[vertex] = neighbourhood
                for key in neighbourhoods:
                    neighbourhoods[key] = {k: neighbourhoods[key][k] for k in sorted(neighbourhoods[key].keys())}
                if visuals:
                    print('neighbourhoods: ')
                    print(neighbourhoods)

                encodings = defaultdict(int)
                i = 1
                for key, value in neighbourhoods.items():
                    if str(value) not in encodings:
                        encodings[str(value)] = i
                        i += 1

                grouped_vertices = defaultdict(list)
                for key, value in neighbourhoods.items():
                    grouped_vertices[encodings[str(value)]].append(key)
                # Now we have a dictionary of the form:
                # neighbourhood: vertices with identical neighbourhoods.
                # Now we can assign each vertex in a group the same color

                if visuals:
                    print('encodings:')
                    print(encodings)
                    print('The vertices grouped by neighbourhoods: ')
                    for e in grouped_vertices:
                        print(e, ":", grouped_vertices[e])
                # Biggest group will keep the same color, all else will assign a new one
                lens_of_partitions = [len(partition) for partition in list(grouped_vertices.values())]
                biggest_partition = list(grouped_vertices.values())[lens_of_partitions.index(max(lens_of_partitions))]

                for group in list(grouped_vertices.values()):
                    if group == biggest_partition:
                        if visuals:
                            print('Group with ' + str(group[0]) + ' keeps color: ' + str(group[0].colornum))
                        continue
                    else:
                        if visuals:
                            print('Group with ' + str(group[0]) + ' GETS color: ' + str(self.next_color))
                        for vertex in group:
                            vertex.colornum = self.next_color
                        self.next_color += 1
            old_coloring = self.coloring
            self.coloring = {v: v.colornum for v in self.disjoint_graph.vertices}
            if visuals:
                self.view_graph(self.disjoint_graph, "iteration" + str(iteration))
            iteration += 1
            if visuals:
                print('\n')

    def check_isomorphisms(self):
        """
        Checks which graphs can be possibly isomorphic using the current coloring.
        (To be used after color_refinement() has been called).
        """
        coloring_values = list(self.coloring.values())
        graphs_coloring = {}
        if not self.indices:
            for i in range(0, len(coloring_values), self.vertex_count):
                graphs_coloring[i // self.vertex_count] = tuple(sorted(coloring_values[i:(i + self.vertex_count)]))
        else:
            j = 0
            for i in self.indices:
                graphs_coloring[i] = tuple(sorted(coloring_values[j:(j + self.vertex_count)]))
                j += self.vertex_count
        isomorphic_graphs = {graphs_coloring[key]: [graph for graph in graphs_coloring if
                                                    graphs_coloring[graph] == graphs_coloring[key]]
                             for key in graphs_coloring}
        possibly_isomorphic = []  # list of tuples (Graphs, is_discrete: bool)
        for key in isomorphic_graphs:
            if len(set(key)) == len(key):
                possibly_isomorphic.append((isomorphic_graphs[key], True))
            else:
                possibly_isomorphic.append((isomorphic_graphs[key], False))
        return possibly_isomorphic

    def print_isomorphisms(self):
        """
        Prints out the (possible) isomorphic graphs, in the given format of this assignment.
        """
        print('Sets of possibly isomorphic graphs:')
        possibly_isomorphic = self.check_isomorphisms()
        for (graphs, is_discrete) in possibly_isomorphic:
            if is_discrete:
                print(str(graphs) + ' discrete')
            else:
                print(graphs)

        return possibly_isomorphic


class GraphIsomorphism:
    def __init__(self, graph_set, mode, extra_aut=0):
        self.graph_set = graph_set
        self.mode = mode        # The mode (GI, AUT or GIAUT) to run the application with 
        self.cr = ColourRefinement(graph_set, mode=mode, extra_aut = extra_aut)
        self.isomorphic = []    # Pairs of isomorphic graphs

    def get_isomorphic_graphs(self, timed=True):
        """
        Runs color refinement
        :param timed: True if you want the color refinement process to be timed
        """
        if timed:
            start_time = time.time()

        self.cr.color_refinement()
        # self.cr.print_isomorphisms()

        if timed:
            exec_time = (time.time() - start_time)
            # print('Execution time in seconds: ' + str(round(exec_time, 2)))
            return round(exec_time, 2)

    def get_individualized_graphs(self, find_one=False):
        """
        Runs graphs individualization by ways of couting isomorphisms or automorphisms
        :param find_one: True if you want to find only one isomorphism
        """
        #checking for which graphs are isomorphic and discrete to know what to count isomorphisms for
        possibly_isomorphic = self.cr.check_isomorphisms()
        
        start_time = time.time()
        
        if self.mode == "GI":
            print("Looking for isomorphisms...")
            for tuple in possibly_isomorphic:
                # possible isomorphism found for 2 graphs
                if len(tuple[0]) == 2:
                    if not tuple[1]:  # if graph is not discrete
                        refinement = ColourRefinement(self.graph_set, indices=tuple[0])
                        refinement.create_initial_colouring()
                        num_isomorphisms = self.count_isomorphisms(refinement, [], find_one=find_one)
                        if num_isomorphisms > 0:
                            self.isomorphic.append(tuple[0])
                # possible isomorphism found for >2 graphs
                if len(tuple[0]) > 2:
                    if not tuple[1]:  # if graph is not discrete
                        for i in range(0, len(tuple[0])):
                            for j in range(i + 1, len(tuple[0])):
                                indices = [tuple[0][i], tuple[0][j]]
                                # print("working on "+str(indices))
                                refinement = ColourRefinement(self.graph_set, indices=indices)
                                refinement.create_initial_colouring()
                                num_isomorphisms = self.count_isomorphisms(refinement, [], find_one=find_one)
                                if num_isomorphisms > 0:
                                    self.isomorphic.append(indices)

        elif self.mode == "AUT":
            
            print("Looking for automorphisms...")
            self.cr.create_initial_colouring()
            num_isomorphisms = self.count_automorphisms(self.cr, [], find_one=find_one)
            if num_isomorphisms > 0:
                print("Number of automorphisms found " + str(num_isomorphisms))

        elif self.mode == "GIAUT":
            automorphisms = {}
            print("Looking for isomorphisms...")
            find_one = False
            for tuple in possibly_isomorphic:
                # possible automorphism found for 2 graphs
                if len(tuple[0]) == 2:
                    if not tuple[1]:  # if graph is not discrete
                        refinement = ColourRefinement(self.graph_set, indices=tuple[0])
                        refinement.create_initial_colouring()
                        num_isomorphisms = self.count_isomorphisms(refinement, [], find_one=find_one)
                        if num_isomorphisms > 0:
                            self.isomorphic.append(tuple[0])
                            automorphisms[indices[0]] = num_isomorphisms

                # possible automorphism found for >2 graphs
                if len(tuple[0]) > 2:
                    if not tuple[1]:  # if graph is not discrete
                        for i in range(0, len(tuple[0])):
                            for j in range(i + 1, len(tuple[0])):
                                indices = [tuple[0][i], tuple[0][j]]
                                refinement = ColourRefinement(self.graph_set, indices=indices)
                                refinement.create_initial_colouring()
                                num_isomorphisms = self.count_isomorphisms(refinement, [], find_one=find_one)
                                if num_isomorphisms > 0:
                                    self.isomorphic.append(indices)
                                    automorphisms[indices[0]] = num_isomorphisms

        exec_time = (time.time() - start_time)
        
        print('Execution time in seconds: ' + str(round(exec_time, 2)))
        if self.mode == "GI":
            equivalance_classes = self.join_output(self.isomorphic)
            print("Equivalence classes: ")
            print(equivalance_classes)

        elif self.mode == "GIAUT":
            equivalance_classes = self.join_output(self.isomorphic)
            for entry in equivalance_classes:
                #if a graph is not isomorphic with any other graph, we still need its automorphisms so they get counted here
                if len(entry) == 1:
                    start_time = time.time()
                    
                    solution = GraphIsomorphism(self.graph_set, "EXTRA_AUT",extra_aut=entry[0])
                    solution.cr.create_initial_colouring()
                    num_auto = solution.count_automorphisms(solution.cr, [], find_one=find_one)
                    automorphisms[entry[0]] = num_auto
                    
                    exec_time += (time.time() - start_time)

            print("Equivalence classes: ")
            for i in range(0, len(equivalance_classes)):
                print(str(equivalance_classes[i]) + " " + str(automorphisms[equivalance_classes[i][0]]))

        return round(exec_time, 2)

    def join_output(self, data):
        """
        Joins seperate tuples of indices, f.e. [0,1], [0,2] and [1,2] joins to [0,1,2]
        :param data: List of lists of indices that are isomorphic
        """
        # create a dictionary with each element as the key
        temp_dict = dict.fromkeys(range(0,len(self.cr.graph_set[0])))
        for lists in data:  # iterating through the lists
            for elements in lists:  # iterating through the elements in the lists
                if temp_dict[elements] == None:
                    temp_dict[elements] = []
                temp_dict[elements].append(lists)  # each element as the key

        result = []
        visited = set()
        for key, value in temp_dict.items():
            if value != None:
                if key not in visited:
                    visited.add(key)
                    common = []
                    for item in value:
                        common += item
                    for elements in common:
                        visited.add(elements)
                    result.append(list(set(common)))  # using a set to remove duplicates
            else:
                result.append([key])
        return result

    def count_isomorphisms(self, refinement: "ColourRefinement", bookkeeping, find_one=False):
        """
        Main method for counting isomorphisms
        :param refinement: ColourRefinemene object that contains the graph for isomorphism counting
        :param bookkeeping: A list of lists of vertices that should be kept in the same color
        :param find_one: True if you want to find only one isomorphism
        """
        refinement.color_refinement(initialize=False)
        possibly_isomorphic = refinement.check_isomorphisms()

        if len(possibly_isomorphic) != 1:  # unbalanced
            return 0
        elif possibly_isomorphic[0][1]:  # bijection if discrete
            return 1

        # list of all key value pairs of the coloring dictionary
        enumereated = list(refinement.coloring.items())

        for color, vertices in refinement.vertices_by_color.items():

            if len(vertices) >= 4:  # picking a color with 4 or more vertices
                picked_color = color
                break

        i = refinement.vertices_by_color[picked_color][0].label

        num = 0
        vertex1 = enumereated[i][0]  # for debugging

        for j in range(refinement.vertex_count, len(enumereated)):
            if enumereated[j][1] == color:  # picking a vertex from the second graph of color "color"

                vertex2 = enumereated[j][0]  # for debugging

                # the vertices that should get a new color in the next recursion
                new_bookkeeping = copy.deepcopy(bookkeeping)
                new_bookkeeping.append([i, j])

                refinement.create_initial_colouring()

                enumereated2 = list(refinement.coloring.items())

                # coloring the vertices we want to keep together
                for entry in new_bookkeeping:
                    refinement.coloring[enumereated2[entry[0]][0]] = refinement.next_color
                    enumereated2[entry[0]][0].colornum = refinement.next_color
                    refinement.coloring[enumereated2[entry[1]][0]] = refinement.next_color
                    enumereated2[entry[1]][0].colornum = refinement.next_color
                    refinement.next_color += 1

                num = num + self.count_isomorphisms(refinement, new_bookkeeping, find_one=find_one)
                if num > 0 and find_one:
                    return 1
        return num

    def count_automorphisms(self, refinement: "ColourRefinement", bookkeeping, find_one=False):
        """
        Main method for counting automorphisms. A slight variation of count_isomorphisms
        :param refinement: ColourRefinemene object that contains the graph for automorphism counting
        :param bookkeeping: A list of lists of vertices that should be kept in the same color
        :param find_one: True if you want to find only one automorphism
        """
        refinement.color_refinement(initialize=False)
        possibly_isomorphic = refinement.check_isomorphisms()

        if len(possibly_isomorphic) != 1:  # unbalanced
            return 0
        elif possibly_isomorphic[0][1]:  # bijection if discrete
            return 1

        # list of all key value pairs of the coloring dictionary
        enumereated = list(refinement.coloring.items())

        for color, vertices in refinement.vertices_by_color.items():

            if len(vertices) >= 4:  # picking a color with 4 or more vertices
                picked_color = color
                break

        i = refinement.vertices_by_color[picked_color][0].label

        num = 0
        vertex1 = enumereated[i][0]  # for debugging

        for j in range(refinement.vertex_count, len(enumereated)):

            if find_one and num >= 1:
                break

            if enumereated[j][1] == color:  # picking a vertex from the second graph of color "color"

                vertex2 = enumereated[j][0]  # for debugging

                # the vertices that should get a new color in the next recursion
                new_bookkeeping = copy.deepcopy(bookkeeping)
                new_bookkeeping.append([i, j])

                refinement.create_initial_colouring()

                enumereated2 = list(refinement.coloring.items())

                # coloring the vertices we want to keep together
                for entry in new_bookkeeping:
                    refinement.coloring[enumereated2[entry[0]][0]] = refinement.next_color
                    enumereated2[entry[0]][0].colornum = refinement.next_color
                    refinement.coloring[enumereated2[entry[1]][0]] = refinement.next_color
                    enumereated2[entry[1]][0].colornum = refinement.next_color
                    refinement.next_color += 1

                num = num + self.count_automorphisms(refinement, new_bookkeeping, find_one=find_one)

        return num

def run(filename, mode, find_one):
    """
    Runs a specific mode of the application on the selected file
    :param filename: The name of the file in src/graphs/delivery to run the application on
    :param mode: The mode (GI, AUT or GIAUT) to run the application with
    :param find_one: True if you want to find only one isomorphism
    """
    local_timer = 0
    if mode == "AUT" and filename[-1] == 'l':
        print("File:", filename)
        print("-------------------------------------------")
        with open('src/graphs/' + filename) as f:
            loaded_graph = load_graph(f, read_list=True)
        for i in range(len(loaded_graph[0])):
            name = filename[:-4] + '_' + str(i) + '.gr'
            with open('src/graphs/' + name, 'w') as f:
                save_graph(loaded_graph[0][i], f)
            print("Solution of graph of index: " + str(i))
            solution = GraphIsomorphism('src/graphs/' + name, mode)
            local_timer += solution.get_isomorphic_graphs()
            print('[' + str(i) + ']')
            local_timer += solution.get_individualized_graphs()
            print("-------------------------------------------")
    else:
        print("File:", filename)
        solution = GraphIsomorphism('src/graphs/' + filename, mode)
        local_timer += solution.get_isomorphic_graphs()
        local_timer += solution.get_individualized_graphs(find_one=find_one)
        print("-------------------------------------------")
    return local_timer

def main():
    timer = 0
    
    timer += run("Basic1GI.grl", "GI", True)

    timer += run("Basic2GI.grl", "GI", True)

    timer += run("Basic3GI.grl", "GI", True)

    timer += run("Basic4Aut.gr", "AUT", False)

    timer += run("Basic5Aut.grl", "AUT", False)

    timer += run("Basic6GIAut.grl", "GIAUT", False)
    minutes = int(round(timer / 60, 0))
    seconds = int(round(timer - minutes * 60, 2))
    print("===========================================")
    print("TOTAL TIME " + str(minutes) + "min " + str(seconds) + "s")
    print("===========================================")

if __name__ == '__main__':
    main()
