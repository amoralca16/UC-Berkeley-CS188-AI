# - * - coding: utf_8 - * - #
# Alicia Morales Carrasco search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


# class Node:
#     '''
#     This class is a data structure needed to create the search tree.
#     '''
#
#     def __init__(self, state, action=None, parent=None):
#         self.__state = state
#         self.__parent = parent
#         self.__action = action
#
#     def getState(self):
#         return self.__state
#
#     def getParent(self):
#         return self.__parent
#
#
# def childNode(problem, parent, action):
#     return 0


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** MY CODE HERE ***"
    # My node representation is: [state, (directions from start state to this node)]
    if not problem.isGoalState(problem.getStartState()): # If the start state is the goal... then do nothing
        explored = []  # Empty list of explored nodes
        frontier = util.Stack()  # The fringe it's a Stack because we extract the last node entered in the frontier (LIFO)
        frontier.push((problem.getStartState(), []))  # Init the search graph using the initial state of problem

        while not frontier.isEmpty():  # If there are still candidates on the frontier...
            node = frontier.pop()  # Choose a leaf node for expansion acording to DFS (depth first)
            if problem.isGoalState(node[0]): # Is it the extracted node our goal?
                return node[1]
            if node[1] not in explored: # If not let's expand it
                explored.append(node[0]) # Don't forget marking it as explored (we don't want to expand it again)
                for child in problem.getSuccessors(node[0]): # Appending all the children to the frontier
                    if not child[0] in explored:
                        frontier.push((child[0], node[1] + [child[1]]))

    return []  # failure


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** MY CODE HERE ***"
    # My node representation is: [state, (directions from de Startstate)]
    if not problem.isGoalState(problem.getStartState()):
        explored = []  # Empty initial explored list of problem
        frontier = util.Queue()  # The fringe it's a Queue because we extract the first node entered on the frontier
        frontier.push((problem.getStartState(), []))  # Init the search graph using the initial state of problem
        while not frontier.isEmpty():  # If there are still candidates on the frontier...
            node = frontier.pop()  # Choose a leaf node for expansion acording to BFS (breadth first)
            if not node[0] in explored: # If we haven't explored this node before
                explored.append(node[0])
                if problem.isGoalState(node[0]): # Let's see if it's the goal
                    return node[1]
                # Append the children to the queue in order to explore them later
                for child in problem.getSuccessors(node[0]):
                    frontier.push((child[0], node[1] + [child[1]]))
    return []  # failure


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** MY CODE HERE ***"

    if not problem.isGoalState(problem.getStartState()):  # If only the starting node isn't the goal
        explored = []  # Empty initial state of problem
        frontier = util.PriorityQueue()  # The fringe it's a PriorityQueue because we extract the last node sorted by the heuristic value
        frontier.push((problem.getStartState(), []), heuristic(problem.getStartState(),problem))  # Init the search graph using the initial state of problem my node representation consist on a list with (state, directions from the initial state, heuristic value)

        while not frontier.isEmpty():  # If there are still candidates on the frontier...
            node = frontier.pop()  # Choose a leaf node (the node who has a bigger heuristic value)
            if problem.isGoalState(node[0]): # If we have arrived to the Goal
                return node[1]
            if not node[0] in explored: # If haven't explored the node before
                explored.append(node[0]) # Append it to the explored list
                for child in problem.getSuccessors(node[0]): # Let's push all non explored children to the fringe.
                    if not child[0] in explored:
                        frontier.push((child[0], node[1] + [child[1]]), problem.getCostOfActions(node[1] + [child[1]]) + heuristic(child[0], problem)) # The total cost it's the real cost + the heuristic cost
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
