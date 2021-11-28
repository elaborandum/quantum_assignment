#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 14:42:03 2021

@author: Stefan Rombouts
"""

#=============================================================================

import numpy as np
#np_random = np.random.default_rng(20211124)
np_random = np.random.RandomState(20211128) # Legacy random state generator.
# It is now recommended to use np.random.default_rn instead, 
# but we stick with RandomState to maintain compability with networkx.spring_layout


#=============================================================================

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library.standard_gates import RXGate, PhaseGate
from qiskit.providers.aer import QasmSimulator as Simulator

#=============================================================================

import networkx as nx

class WeightedGraph(nx.Graph):
    
    def weighted_edges(self, key='weight'):
        """
        Returns a dictionary that maps edges onto weights.
        """
        return {(i, j): data[key] for i, j, data in self.edges(data=True)}

    def weights(self, key='weight'):
        """
        Returns an iterator over edge weights.
        """
        return self.weighted_edges(key=key).values()

#------------------------------------------------------------------------------

def random_connected_graph(nodes: int, edges: int, max_weight: int = 20, 
                           max_degree=None):
    """
    Creates a random connected graph with a given number of nodes and edges,
    and random integer weights for the edges uniformely drawn 
    from the interval [1:max_weight] (both included).
    The path 0 -> 1 -> 2 -> ... -> nodes-1 -> nodes is generated first,
    so the graph is guaranteed to be fully connected.
    
    """
    assert nodes-1  <= edges <= (max_degree or (nodes-1))*nodes/2
    weights = np_random.randint(1, max_weight+1, size=edges)
    graph = WeightedGraph()
    graph.add_weighted_edges_from(
        [(i, i+1, w) for i, w in enumerate(weights[:nodes-1])])
    while graph.number_of_edges() < edges:
        i, j = np_random.randint((nodes, nodes))
        if (i != j) and (not graph.has_edge(i, j)) and not (
                max_degree and ((graph.degree[i] == max_degree) 
                or (graph.degree[j] == max_degree))):
            graph.add_edge(i, j, weight=weights[graph.number_of_edges()])
    return graph    

graph = random_connected_graph(6, 8, max_degree=3)


#=============================================================================

def pair(iterable):
    """
    Returns the first to elements of iterable, supplemented with None values
    if the iterable has less than 2 elements.
    """
    l = list(iterable)[:2]
    return l if len(l) > 1 else ((l + [None, ]) if l else [None, None])


weights = list(graph.weights())
penalty = sum(weights)
nqubits = len(graph.edges)
theta = Parameter('theta')
backend = Simulator()


def ZoneMixer(i, lefts, rights):
    """
    Returns an X rotation on edge i conditioned
    on at least one of the edges on the left being spin-up 
    and all right edges being spin down or vice versa.
    """
    l0, l1 = pair(lefts)
    r0, r1 = pair(rights)
    qc = QuantumCircuit(nqubits)
    RXCC = RXGate(theta).control(2)
    RXCCM = RXGate(-theta).control(2)
    RXCCC = RXGate(theta).control(3)
    RXCCCM = RXGate(-theta).control(3)
    RXCCCCM = RXGate(-theta).control(4)

    def Mixer01(r): # o-o-
        # Flip edge i only if edge r is active
        qc.crx(theta, r, i)

    def Mixer02(r0, r1): # o-o=
        # Flip edge i if r0 or r1 are active
        qc.crx(theta, r0, i)
        qc.append(RXCCM, (r0, r1, i))
        qc.crx(theta, r1, i)

    def Mixer11(l, r): # -o-o-
        # Flip edge i if left edge or right edge is active but not both
        qc.x(l)
        qc.append(RXCC, (l, r, i))
        qc.x((l, r))
        qc.append(RXCC, (l, r, i))
        qc.x(r)

    def Mixer12(l, r0, r1): # -o-o=
        # Flip edge i if left edge is inactive and any right edge is active
        qc.x(l)
        qc.append(RXCC, (l, r0, i))
        qc.append(RXCCCM, (l, r0, r1, i))
        qc.append(RXCC, (l, r1, i))
        qc.x(l)
        # Flip edge i if left edge is active and both right edges are inactive
        qc.x((r0, r1))
        qc.append(RXCCC, (l, r0, r1, i))
        qc.x((r0, r1))

    def Mixer22(l0, l1, r0, r1): # =o-o=
        # Flip edge i if both left edges are inactive and any right edge is active
        qc.x(l0)
        qc.x(l1)
        qc.append(RXCCC, (l0, l1, r0, i))
        qc.append(RXCCCCM, (l0, l1, r0, r1, i))
        qc.append(RXCCC, (l0, l1, r1, i))
        qc.x(l1)
        qc.x(l0)
                               
    if l0 is None:
        if r1 is None: #  o-o-
            Mixer01(r0)
        else: #  o-o=
            Mixer02(r0, r1)
    elif l1 is None:
        if r0 is None: # -o-o
            # Flip edge i only if left edge is active
            Mixer01(l0)
        elif r1 is None: # -o-o-
            Mixer11(l0, r0)
        else: # -o-o=
            Mixer12(l0, r0, r1)
    else: # l0 and l1 are not None
        if r0 is None: # =o-o
            Mixer02(l0, l1)
        elif r1 is None: # =o-o-
            Mixer12(r0, l0, l1)
        else: # =o-o=
            Mixer22(l0, l1, r0, r1)
            # Flip edge i if any left edges are active but both right edges are inactive
            Mixer22(r0, r1, l0, l1)
    return qc

node_edges = {}
for i, (vl, vr) in enumerate(graph.edges()):
    node_edges.setdefault(vl, []).append(i)
    node_edges.setdefault(vr, []).append(i)
zones = [(i,
          [il for il in node_edges[vl] if il != i],
          [ir for ir in node_edges[vr] if ir != i],
         ) for i, (vl, vr) in enumerate(graph.edges)]
mixers = [ZoneMixer(*zone) for zone in zones]

def NodePenalizer(n):
    """
    Applies a phase penalty if all edges are spin down.
    """
    qc = QuantumCircuit(n, name='NodeProjector')
    qc.x(range(n))
    if n == 1:
        qc.p(penalty * theta, 0)
    else:
        qc.append(PhaseGate(penalty * theta).control(n-1), range(n))
    qc.x(range(n))
    return qc

penalizers = {n:NodePenalizer(n) for n in range(1, 4)}

beta = Parameter('beta')
gamma = Parameter('gamma')
qr = QuantumRegister(nqubits)
qslice = QuantumCircuit(qr)
# problem unitary
for edges in node_edges.values():
    penalizer = penalizers[len(edges)].to_instruction({theta: gamma/2})
    qslice.append(penalizer, edges) 
for i, weight in enumerate(weights):
    qslice.rz(weight*gamma/2, i)
# mixer unitary
for mixer in mixers:
    qslice.append(mixer.assign_parameters({theta: beta/2}), qr)
qslice = transpile(qslice, backend)

def make_qaoa(params):
    """
    Sets up a quantum circuit that searches for the minimal spanning tree on 
    the graph using QAOA. 
    
    Args:
        params : a list of pairs of floats (beta, gamma) for each Trotter slice
    """
    qr = QuantumRegister(nqubits)
    qc = QuantumCircuit(qr)
    # Connect a random node to all its neighbours to have an initial path
    random_node = np_random.randint(len(graph.nodes))
    qc.x(node_edges[random_node])
    for mixer in mixers:
        qc.append(mixer.assign_parameters({theta: np.pi/4}), qr)
    param_iter = iter(params)
    for b, g in zip(param_iter, param_iter): # Consumes params pair by pair.
        qc.append(qslice.assign_parameters({beta: b, gamma: g}), qr)
    qc.measure_all()
    return transpile(qc, backend)

def path_length(x):
    """
    Returns the sum of weights for all edges selected in the bitstring x.
    (leftmost bit corresponds to the first edge)           
    """
    length = sum(weight for weight, bit in zip(weights, x[::-1]) if bit == '1')
    path = [e for e, bit in zip(graph.edges, x[::-1]) if bit == '1']
    node_set = {i for i, j in path} | {j for i, j in path}
    path_penalty = penalty*(len(graph.nodes)-len(node_set))
    return length + path_penalty

def evaluate_QAOA(shots=512):
    """
    Returns a function that evaluates a QAOA circuit
    """

    def execute_circuit(params):
        qc = make_qaoa(params)
        seed = np_random.randint(9999)
        counts = backend.run(qc, seed_simulator=seed, shots=shots
                             ).result().get_counts()
        total_paths = sum(count*path_length(bits) 
                          for bits, count in counts.items())
        total_counts = sum(counts.values())
        return total_paths/total_counts
    
    return execute_circuit


#=============================================================================

def evaluate_params(params, verbose: bool = False):
    """
    Runs the QAOA circuit with the given parameters
    and returns the obtained path lenght and the set of edges in the path 
    """
    qaoa = make_qaoa(params)
    seed = np_random.randint(9999)
    counts = backend.run(qaoa, seed_simulator=seed, shots=1024
                         ).result().get_counts()

    if verbose:
        print(counts)
    best_ix = np.fromiter(counts.values(), dtype=np.int32).argmin()
    best_bit = list(counts.keys())[best_ix]
    best_path = [e for e, bit in zip(graph.edges, best_bit[::-1]) if bit == '1']
    best_length = path_length(best_bit)
    return best_length, best_path

    
#=============================================================================

from scipy.optimize import minimize

def find_best_x(params, verbose: bool = False):
    print("Looking for the best beta and gamma starting from ", params)
    evaluator = evaluate_QAOA()
    best_x = minimize(evaluator, 
                      x0=params, 
                      method='COBYLA',
                      # restrict beta and gamma to the interval $]-\pi, \pi]$
                      constraints=(
                          {'type': 'ineq', 'fun': lambda x: x[0] > -np.pi}, 
                          {'type': 'ineq', 'fun': lambda x: x[0] <= np.pi}, 
                          {'type': 'ineq', 'fun': lambda x: x[1] > -np.pi}, 
                          {'type': 'ineq', 'fun': lambda x: x[1] <= np.pi}, 
                          ),
                      options={'maxiter': 40, 'rhobeg': 0.5, 'disp': True},
                      )
    if verbose:
        print(best_x)
    return best_x.x


#=============================================================================

def guess_best_x(params, iterations=40):
    bg = params
    length, path = evaluate_params(bg)
    print(f"step {0:>4}: starting with length {length:8.2f}")
    for it in range(1, iterations+1):
        bg = np_random.uniform(-np.pi, np.pi, 2)
        l, p = evaluate_params(bg)
        if (len(p) > len(path)) or ((len(p)==len(path)) and (l < length)):
            params, length, path = bg, l, p
            print(f"step {it:>4}: found a better result with length {length:8.2f}")
    return params


#=============================================================================
