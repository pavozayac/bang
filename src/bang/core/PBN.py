from typing import List
from libsbml import *
import itertools
#from parseSBML import parseSBMLDocument

class PBN:

    def __init__(self, n : int,
                nf : List[int], 
                nv : List[int],
                F : List[List[bool]],
                varFInt : List[List[int]],
                cij : List[List[float]],
                perturbation : float,
                npNode : List[int]):
        
        self.n = n  #the number of nodes
        self.nf = nf    #the size is n
        self.nv = nv    #the sizef = cum(nf)
        self.F = F  #each element of F stores a column of the truth table "F"， e.g., F.get(0)=[true false], the length of the element is 2^nv(boolean function index)
        self.varFInt = varFInt  
        self.cij = cij  #the size=n, each element represents the selection probability of a node, and therefore the size of each element equals to the number of functions of each node
        self.perturbation = perturbation    #perturbation rate
        self.npNode = npNode    #index of those nodes without perturbation. To make simulation easy, the last element of npNode will always be n, which also indicate the end of the list. If there is only one element, it means there is no disabled node.


    def getN(self):
        return self.n

    def getNf(self):
        return self.nf
    
    def getNv(self):
        return self.nv
    
    def getF(self):
        return self.F
    
    def getVarFInt(self):
        return self.varFInt
    
    def getCij(self):
        return self.cij
    
    def getPerturbation(self):
        return self.perturbation
    
    def getNpNode(self):
        return self.npNode
    



def load_sbml(path: str) -> PBN:
    #return PBN(*parseSBMLDocument(path))
    pass

def load_assa(path):
    n = 0
    nf = []
    nv = []
    F = []
    varFInt = []
    cij = []
    perturbation = 0.0
    np = []
    type = ''
    i = 0

    # forbidden characters in the variable names
    forbidden_chars = [' ', '\t', '\n', '\r', '\v', '\f', '&', '*', '!', '^', '/', '|', ':', '(', ')']

    # for python eval function
    logical_replacements = {
        '|': ' or ',
        '&': ' and ',
        '!': ' not '
    }
 
    def delete_comments_and_empty_lines(lines):
        new_lines = []
        for line in lines:
            line = line.strip()
            if line == "" or line.startswith("//"):
                continue
            new_lines.append(line)
        return new_lines

    def get_vars(fun):
        vars = []
        cur = ''
        started = False
        for i in range(len(fun)):
            if fun[i] in forbidden_chars:
                if started:
                    if cur not in vars:
                        vars.append(cur)
                    cur = ''
                    started = False
                else:
                    continue
            else:
                cur += fun[i]
                started = True
        if started:
            if cur not in vars:
                vars.append(cur)
        return vars
    
    def get_n(line):
        nonlocal n
        line = line.strip()
        if line.startswith("n=") == False:
            raise ValueError("Invalid file format")
        n = line.split("=")[1]
        if not n.isnumeric():
            raise ValueError("Invalid file format")
        n = int(n)
        nonlocal i
        i += 1

    def get_type(line):
        nonlocal type
        line = line.strip()
        if line.startswith("type=") == False:
            raise ValueError("Invalid file format")
        type = line.split("=")[1]
        if type not in ['synchronous', 'rog', 'rmg', 'rmgrm', 'rmgro', 'rmgrorm', 'aro']:
            raise ValueError("Invalid file format")
        nonlocal i
        i += 1
    
    def get_perturbation(line):
        nonlocal perturbation
        line = line.strip()
        if line.startswith("perturbation=") == False:
            raise ValueError("Invalid file format")
        perturbation = line.split("=")[1]
        try:
            perturbation = float(perturbation)
        except ValueError:
            raise ValueError("Invalid file format")
        nonlocal i
        i += 1
    

    with open(path, 'r') as f:
        lines = f.readlines()
        lines = delete_comments_and_empty_lines(lines)

        # Read the type of the PBN
        get_type(lines[i])

        # Read the number of nodes
        get_n(lines[i])

        # Read the perturbation rate
        get_perturbation(lines[i])
        
        names_dict = {}
        index_dict = {}
        forbidden_names = ['or', 'not', 'and']
        # Read the names of the nodes
        names_line = lines[i]
        names_line = names_line.strip()
        if names_line != "nodeNames":
            raise ValueError("Invalid file format")
        i += 1
        for j in range(n):
            name = lines[i].strip()
            print('name:', name)
            if name in names_dict:
                raise ValueError("Duplicate node name")
            if name in forbidden_names:
                raise ValueError("Invalid node name")
            for char in forbidden_chars:
                if char in name:
                    raise ValueError("Invalid node name")

            names_dict[name] = j
            index_dict[j] = name
            i += 1
        end_names_line = lines[i]
        end_names_line = end_names_line.strip()
        if end_names_line != "endNodeNames":
            raise ValueError("Invalid file format")
        i += 1
        for j in range(n):
            function_count = 0
            node_line = lines[i]
            node_line = node_line.strip()
            if node_line != "node " + index_dict[j]:
                raise ValueError("Invalid file format " + node_line)
            i += 1
            probs = []
            while lines[i].strip() != "endNode":
                function_count += 1
                function_line = lines[i]
                function_line = function_line.strip()
                function_line = function_line.split(":")
                aux_f_line = []
                for f in function_line:
                    aux_f_line.append(f.strip())
                function_line = aux_f_line
                # get probability of the function
                fun_prob = function_line[0]
                try:
                    fun_prob = float(fun_prob)
                    probs.append(fun_prob)
                except ValueError:
                    raise ValueError("Invalid file format")
                # extract variable names
                fun = function_line[1]
                vars = get_vars(fun)
                # check if extracted variables are valid
                for var in vars:
                    if var not in names_dict:
                        raise ValueError("Invalid file format")
                unsorted_var_indices = []
                for var in vars:
                    unsorted_var_indices.append(names_dict[var])

                updated_fun = fun
                for rep in logical_replacements:
                    updated_fun = updated_fun.replace(rep, logical_replacements[rep])
                sorted_var_indices = sorted(unsorted_var_indices)
                nv.append(len(sorted_var_indices))

                # add variable indices to the list varFInt
                varFInt.append(sorted_var_indices)
                sorted_vars = [index_dict[var] for var in sorted_var_indices]

                # generate every possible variable evaluation
                truth_combinations = list(itertools.product([False, True], repeat=len(vars)))
                truth_table = []
                # evaluate the function for every possible combination of variables
                # and store the result in the truth table
                for combination in truth_combinations:
                    values = dict(zip(sorted_vars, combination))
                    evaluated = eval(updated_fun, {}, values)
                    truth_table.append(evaluated)
                F.append(truth_table)
                i += 1
            nf.append(function_count)
            cij.append(probs)
            i += 1

        assert len(F) == sum(nf)
        np_line = lines[i].strip()
        if np_line != "npNode":
            raise ValueError("Invalid file format")
        i += 1
        while lines[i].strip() != "endNpNode":
            node_name = lines[i].strip()
            if node_name not in names_dict:
                raise ValueError("Invalid file format")
            np.append(names_dict[node_name])
            i += 1
        np = sorted(np)
        np.append(n)
    model = PBN(n, nf, nv, F, varFInt, cij, perturbation, np)
    return model



def load_from_file(path, format='sbml'):
    match format:
        case 'sbml':
            return load_sbml(path)
        case 'assa':
            return load_assa(path)
        case _:
            raise ValueError("Invalid format")
        







