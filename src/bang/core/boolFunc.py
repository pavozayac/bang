from libsbml import *
from collections.abc import Callable

# Returns a function evaluating expression in mathExpression and a set of relevant nodes
def getFunc(mathExpression :ASTNode, nodes: dict[str, int]) -> tuple[Callable[[dict[int, bool]], bool], set[int]]:
	
	def merge(func :Callable[[bool, bool], bool]) -> tuple[Callable[[list[bool]], bool], set[int]]:
		func1, set1 = getFunc(mathExpression.getRightChild(), nodes)
		func2, set2 = getFunc(mathExpression.getLeftChild(), nodes)
		return (lambda node_vals: func(func1(node_vals), func2(node_vals))), (set1 | set2)
		
	if mathExpression.getType() == AST_LOGICAL_AND:
		return merge(lambda x1, x2: x1 and x2)
	elif mathExpression.getType() == AST_LOGICAL_OR:
		return merge(lambda x1, x2: x1 or x2)
	elif mathExpression.getType() in {AST_LOGICAL_XOR, AST_RELATIONAL_NEQ}:
		return merge(lambda x1, x2: x1 != x2)
	elif mathExpression.getType() == AST_RELATIONAL_EQ:
		return merge(lambda x1, x2: x1 == x2)
	elif mathExpression.getType() == AST_LOGICAL_NOT:
		func, relevant_nodes = getFunc(mathExpression.getChild(0), nodes)
		return (lambda node_vals: not func(node_vals)), relevant_nodes  # TODO: check correctness and num children
	elif mathExpression.getType() == AST_NAME:
		return (lambda node_vals: node_vals[nodes[mathExpression.getName()]]), {nodes[mathExpression.getName()]}
	elif mathExpression.getType() == AST_INTEGER and (mathExpression.getValue() in {0.0, 1.0}):
		return (lambda node_vals: True if mathExpression.getValue() == 1.0 else False), set() 
	elif mathExpression.getType() == AST_CONSTANT_TRUE:
		return (lambda node_vals: True), set()
	elif mathExpression.getType() == AST_CONSTANT_FALSE:
		return (lambda node_vals: False), set()

def incrementNodes(bits: dict[int, bool], nodes: list[int]) ->  bool:
    carry = True  
    
    for i in range(len(nodes)):
        if bits[nodes[i]] == True:  
            bits[nodes[i]] = False   
        else:                 
            bits[nodes[i]] = True    
            carry = False     
            break
    
    return carry

def parseFunction(mathExpression :ASTNode, nodes: dict[str, int]) -> tuple[list[bool], list[int]]:
	func, tmp = getFunc(mathExpression, nodes)
	relevant_nodes = list(tmp)
	relevant_nodes.sort()
	truth_table :list[bool] = []
	node_dictionary = {node: False for node in relevant_nodes}
	loop: bool
	while True:
		truth_table.append(func(node_dictionary))
		if incrementNodes(node_dictionary, relevant_nodes):
			break
	return truth_table, relevant_nodes