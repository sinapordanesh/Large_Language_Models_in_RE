import os
import argparse
from tree_sitter import Parser, Language
import networkx as nx
import keyword
import pickle
import pymongo

    # ------------------------------
    # Function : convertTree
    # ------------------------------

def convertTree(node, parentNode):
    
    global nodePosition
    nodePosition += 1
    
    # if node.type.strip():
    #     nodeName = nodePosition
    
    # else:
    #     nodeName = "separator"
    
    nodeName = nodePosition
    if node.type.strip():
        graph.add_node(nodeName, nodeType = node.type)
        
    else:
        graph.add_node(nodeName, nodeType = "separator")
        
    # graph.add_node(nodeName, nodeType = node.type)
   
    if not (parentNode is None):
        graph.add_edge(parentNode, nodeName, between = str(parentNode) + ' - ' + str(nodeName))
        
    for child in node.children:
        convertTree(child, nodeName)

    pass


    # ------------------------------
    # Function : branchingFactor
    # ------------------------------

def branchingFactor(graph):

    number_of_non_root_nodes = graph.number_of_nodes() - 1

    # print(number_of_non_root_nodes)

    children_exist_node = 0 

    for node in graph.nodes():
        if graph.out_degree(node) > 0:
            children_exist_node += 1

    # print(children_exist_node)

    avgBranchingFactor = number_of_non_root_nodes / children_exist_node

    return(avgBranchingFactor)



    # ------------------------------
    # Function : maxDepthASTNode
    # ------------------------------

def maxDepthASTNode(graph):

    leaf_nodes = []

    # for node in graph.nodes():
    #     print(node)
    
    leaf_nodes = [node for node in graph.nodes() if graph.degree(node) == 1]

    # print(leaf_nodes)

    all_depths = []

    for leaf_node in leaf_nodes:
        all_depths.append(nx.shortest_path_length(graph, source=1, target=leaf_node))

    return(max(all_depths))

    # ------------------------------
    # Function : tfPythonKeywords
    # ------------------------------

def tfPythonKeywords(graph):

    list_of_keyword = keyword.kwlist

    # print(list_of_keyword)

    # print(len(list_of_keyword))

    dictionary_of_keywords = dict.fromkeys(list_of_keyword, 0.0)

    # print(dictionary_of_keywords)

    node_attributes = list(nx.get_node_attributes(graph, "nodeType").values())

    # print(node_attributes)

    for node in node_attributes:
        if node in dictionary_of_keywords.keys():
            dictionary_of_keywords[node] += 1
    
    return(dictionary_of_keywords)

    # for key, value in dictionary_of_keywords.items():
    #     if key in node_attributes:
    #         dictionary_of_keywords[key] +=1


    # print(dictionary_of_keywords)

    # sum_total = 0.0

    # for value in dictionary_of_keywords.values():
    #     sum_total += value

    # print(sum_total)

    # TF_dictionary = dictionary_of_keywords

    # for key, value in TF_dictionary.items():
    #     new_value = value/sum_total
    #     TF_dictionary.update({key:new_value})

    # print(TF_dictionary)


    # ------------------------------
    # Function : ASTNodeTypesTF
    # ------------------------------

def ASTNodeTypesTF(graph):

    leaf_nodes = [node for node in graph.nodes() if graph.degree(node) == 1]

    # print(leaf_nodes)

    with open("/Users/sbukhari/Sandbox/versionpyV1/version-py/Processed/DictOfNodes/DictofNodes.pickle", 'rb') as pickle_load:
        dictionary_of_generalizedNodes = pickle.load(pickle_load)

    # print(dictionary_of_generalizedNodes)

    list_of_generalizedNodes = [value for value in dictionary_of_generalizedNodes.values()]
    list_of_key_generalizedNodes = [key for key in dictionary_of_generalizedNodes.keys()]

    # print(list_of_generalizedNodes)

    dict_with_only_generalNodes = dict.fromkeys(list_of_generalizedNodes)

    for key, value in dict_with_only_generalNodes.items():
        dict_with_only_generalNodes.update({key : 0.0})

    # print(dict_with_only_generalNodes)

    # node_attributes = list(nx.get_node_attributes(graph, "nodeType").values())

    # print(node_attributes)

    list_of_whole_graph = list(graph.nodes(data=True))

    # print(list_of_whole_graph)

    # list_of_all_nodes = list(dictionary_of_generalizedNodes.keys())

    # print(list_of_all_nodes)

    generalNodeCategory = []

    for nodePosition in range(1,len(list_of_whole_graph)+1):
        if nodePosition not in leaf_nodes:
            graph_node = list(list_of_whole_graph[nodePosition][1].values())[0]  
            if graph_node == "module":
                continue
            elif graph_node in list_of_key_generalizedNodes:
                generalNodeCategory.append(dictionary_of_generalizedNodes[graph_node]) 
        else:
            continue

    # print(generalNodeCategory)
    # print(len(generalNodeCategory))
    # print(len(list_of_whole_graph) - len(leaf_nodes))

    dictionary_to_keep_count = {}

    # print(generalNodeCategory)

    for entry in generalNodeCategory:
        count_of_entry = generalNodeCategory.count(entry)
        # print(entry, count_of_entry)
        if entry not in dictionary_to_keep_count.keys():
            dictionary_to_keep_count.update({entry : count_of_entry})
        

    # print(dictionary_to_keep_count)

    sumofAllValues = 0

    for each_value in dictionary_to_keep_count.values():
        # print(each_value)
        sumofAllValues += each_value

    # print(sumofAllValues)

    for each_node, each_value in dictionary_to_keep_count.items():
        new_value = each_value / sumofAllValues
        dictionary_to_keep_count.update({each_node : new_value})


    # print(dictionary_to_keep_count)

    for each_node, each_value in dictionary_to_keep_count.items():
        if each_node in dict_with_only_generalNodes.keys():
            dict_with_only_generalNodes.update({each_node : each_value})
        else:
            print("exception raised")


    return(dict_with_only_generalNodes)


def saveOutputVector(file_name, file_folder, file_author, file_path, output_vector):

    # -------------------------------------------------------
    # Mongo db initialization commands starts here
    # -------------------------------------------------------
        
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")

    mydb = myclient["VersionPy"]

    mycol = mydb["VectorSyntactic"]

    insertDocument = {"File_Name" : file_name, "File_Folder" : file_folder, "Coded_By" : file_author, "File_Path" : file_path, "Output_Vector" : output_vector}

    x = mycol.insert_one(insertDocument)

    # print(x)
    pass




nodeCount = 0
nodePosition = 0 
extractTree = []

def main():

    # Parse the input commmand which contains the filename to extract the lexical features from
    
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument('file', help='Add absolute path to the file that needs to be parsed')
    args = cmd_parser.parse_args()
    
    file_path = args.file
    
    splitted_file_path = file_path.split("/")
    
    # Extracting data about the file
        
    file_name = os.path.splitext(splitted_file_path[-1])[0]
    
    file_folder = splitted_file_path[-2]
    
    file_author = splitted_file_path[-3]

    # Parse the code file in to AST 

    PY_LANGUAGE = Language('build/my-languages.so', 'python')
    parser = Parser()
    parser.set_language(PY_LANGUAGE)

    with open(file_path, 'rb') as f:
        fcontent = f.read(-1)
        tree = parser.parse(fcontent)

    global graph 
    graph = nx.DiGraph()
    
    global newgraph
    newgraph = nx.DiGraph()


    # ------------------------------------------------------------
    # Convert the tree-sitter tree into a networkx graph
    # ------------------------------------------------------------

    convertTree(tree.root_node, None)

    # ------------------------------
    # LEXICAL FEATURE - branchingFactor
    # 
    # branching factor of the tree formed by converting code blocks of files into nodes
    #
    # The average branching factor can be quickly calculated as 
    # the number of non-root nodes (the size of the tree, minus one;
    # divided by the number of non-leaf nodes (the number of nodes with children). WIKIPEDIA
    #
    # ------------------------------

    avgBranchingFactor = branchingFactor(graph)
    # print(avgBranchingFactor)

    # ------------------------------
    # SYNTACTIC FEATURE - MaxDepthASTNode
    # 
    # Maximum depth of an AST node
    # ------------------------------

    max_depth = maxDepthASTNode(graph)
    # print(max_depth)

    # ------------------------------
    # SYNTACTIC FEATURE - pythonKeywords
    # 
    # Term frequency of 36 Python keywords
    # ------------------------------

    dict_TF_pythonKeywords = tfPythonKeywords(graph)
    # print(dict_TF_pythonKeywords)

    # ------------------------------
    # SYNTACTIC FEATURE - ASTNodeTypesTF
    # 
    # Term frequency of 58 possible AST node type excluding leaves
    # ------------------------------

    dict_with_only_generalNodes = ASTNodeTypesTF(graph)
    # print(dict_with_only_generalNodes)

    # ------------------------------
    #
    # Create Output Vector
    #
    # ------------------------------

    # Make dictionaries to list
    list_TF_pythonKeywords = []
    for value in dict_TF_pythonKeywords.values():
        list_TF_pythonKeywords.append(value)

    list_with_only_generalNodes = []
    for value in dict_with_only_generalNodes.values():
        list_with_only_generalNodes.append(value)


    output_vector = [avgBranchingFactor, max_depth] + list_TF_pythonKeywords + list_with_only_generalNodes
    
    # print(output_vector)

    # ------------------------------
    #
    #  Save Output Vector
    #
    # ------------------------------

    saveOutputVector(file_name, file_folder, file_author, file_path, output_vector)




if __name__ == '__main__':
    main() 