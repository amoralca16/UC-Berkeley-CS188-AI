# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

AGENT = 0

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # As it's a Reflex Agent it will return a value depending on the game state (in particular
        # distance to food, and distance to ghosts)


        # If we use newFood, food would not be there to give a reward !
        newFoodList = currentGameState.getFood().asList()


        # Order food by proximity to PacMan
        # Notice: A different approach would be using a priority queue.
        newFoodList.sort(key=lambda pos: util.manhattanDistance(newPos, pos))
        minDistToFood = util.manhattanDistance(newPos, newFoodList[0])

        # Sorting enemy agents by distance to PacMan
        ghostList = [Ghost.getPosition() for Ghost in newGhostStates]
        ghostList.sort(key=lambda pos: util.manhattanDistance(newPos, pos))
        minDistToGhost = util.manhattanDistance(newPos, ghostList[0])

        # If PacMan steps on ghost, game over
        if util.manhattanDistance(newPos, ghostList[0]) == 0:
            # PacMan dies
            return float('-inf')

        else:
            # The greater the distance to a ghost, the less the penalty
            # scoreGamma is a penalty factor
            scoreGamma = 1.0 / minDistToGhost

            # If PacMan steps on food, eats it
            if minDistToFood  == 0:
                # Even if the pacman is to get a food, if too close to ghost, less reward
                # PacMan should behave cautiously
                returnScore = 10.0 - pow(scoreGamma, 2.0)
            else:
                # PacMan has a negative reward if it is too far from food
                # Reward based on distance to food (less distance, more reward),
                # and distance to ghosts (more distance, more reward).
                returnScore = 1.0 / minDistToFood - pow(scoreGamma, 1.5)

        return returnScore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        depth = 1.0
        return self.maxNode(depth, self.index, gameState)

    def maxNode(self, depth, agent, state):
        # This function receives an agent state and returns an utility value for it.

        # Checking if it's a terminal state.
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state); # If it's fetch score.

        # Initialization of nodeValue. Since we're looking for a maxi  mum, get the minimum value available
        nodeValue = float("-inf")

        # Default action = stand still. Easy to debug.
        maxAction = Directions.STOP

        # We get all possible actions - or the tree children, if one is to see it that way

        for (action,successor) in [(a,state.generateSuccessor(agent,a)) for a in state.getLegalActions(agent)]:

            # For each children, we check the minimum value on every other agent than PacMan, and get the maximum from
            # those
            minNodes = max(nodeValue, self.minNode(depth, agent+1, successor))

            # In case that current best value is lower than value returned, update the values and set
            # the best path to the one we've just found
            if minNodes > nodeValue:
                nodeValue = minNodes
                maxAction = action
        # If current node is root
        if depth == 1:
            # Returning the best path we've found
            return maxAction
        else:
            # Returning the maximum value found from children nodes
            return nodeValue



    def minNode(self, depth, agent, state):
        # Checking the amount of legal actions the agent can do. If none, initialize nodeValue.

        # Checking if it's a terminal state.
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state); # If it's fetch score.

        nodeValue= float("inf")

        # We get all possible actions - or the tree children, if one is to see it that way
        for (action,successor) in [(a,state.generateSuccessor(agent,a)) for a in state.getLegalActions(agent)]:

            # If last ghost from ghost agents
            if agent == state.getNumAgents()-1:
                # If last ghost before the leaves of the tree - last minimizer node
                if depth == self.depth:
                    # Get value of the leaf
                    nodeValue = min(nodeValue,self.evaluationFunction(successor))
                else:
                    # Get PacMan another chance to maximize score
                    nodeValue = min(nodeValue,self.maxNode(depth+1, 0, successor))
            else:
                # Go to the next ghost
                nodeValue = min(nodeValue,self.minNode(depth, agent+1, successor))

        return nodeValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        depth = 1.0
        alpha = float('-inf')
        beta = float('inf')

        return self.maxNode(depth, AGENT, gameState, alpha, beta)


    def maxNode(self, depth, agent, state, alpha, beta):
        # Checking the amount of legal actions the agent can do. If none, initialize nodeValue.
        if state.isWin() or state.isLose(): # Terminal State?
            return self.evaluationFunction(state) # Utility

        # Initialization of nodeValue. Since we're looking for a maximum, get the minimum value available
        nodeValue = float("-inf")

        # Default action = stand still. Easy to debug.
        maxAction = Directions.STOP

        # We get all possible actions - or the tree children, if one is to see it that way
        for action in state.getLegalActions(agent):
            # For each children, we check the minimum value on every other agent than PacMan, and get the maximum from
            # those
            minNodes = max(nodeValue, self.minNode(depth, agent+1, state.generateSuccessor(agent, action), alpha, beta))

            # In case that current best value is lower than value returned, update the values and set
            # the best path to the one we've just found
            if minNodes > nodeValue:
                nodeValue = minNodes
                maxAction = action

            # Pruning case. Since we're on maxNode, if next value retrieved from the tree is higher than beta,
            # we get that one already, no need to check other nodes, since it means we won't get a lower value anyway.
            if nodeValue > beta:
                return nodeValue

            # Updating alpha : best option on path to root
            alpha = max(alpha, nodeValue)

        # If current node is root
        if depth == 1:
            # Returning the best path we've found
            return maxAction
        else:
            # Returning the maximum value found from children nodes
            return nodeValue

    def minNode(self, depth, agent, state, alpha, beta):
        # Checking the amount of legal actions the agent can do. If none, initialize nodeValue.
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        nodeValue= float("inf")

        # We get all possible actions - or the tree children, if one is to see it that way
        for action in state.getLegalActions(agent):
            # If last ghost
            if agent == state.getNumAgents()-1:
                # If last ghost from the tree before reaching the leaves - last minimizer node
                if depth == self.depth:
                    # Get the value from the leaf
                    nodeValue = min(nodeValue, self.evaluationFunction(state.generateSuccessor(agent, action)) )
                else:
                    # Return control to the next maximizer node
                    nodeValue = min(nodeValue,self.maxNode(depth+1, 0, state.generateSuccessor(agent, action), alpha, beta) )
            # IF THERE ARE GHOSTS WAITING TO PLAY (and a little bit of sympathy)
            else:
                # Go to the next ghost
                nodeValue = min( nodeValue, self.minNode(depth, agent+1, state.generateSuccessor(agent, action), alpha, beta))

            # Updating beta
            beta = min(beta, nodeValue)

            # Pruning case. Since we're on minNode, if next value retrieved from the tree is smaller than alpha,
            # we get that one already, no need to check other nodes, since it means we won't get a higher value anyway.
            if nodeValue < alpha:
                return nodeValue

        return nodeValue

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Setting nodeValue as the lowest value possible in order to get any value bigger than it
        nodeValue = float('-inf')
        # Final depth of the tree is going to be depth == 0
        depth = 0.0

        # Stop by default
        nextAction = Directions.STOP

        # Successors of root are expectation, thus they are not maximizing
        maximizingPlayer = False

        # Only one maximizer, others are probability nodes.

        # For every possible legal movement in root
        for action in gameState.getLegalActions():
            # Root node is a Maximizer, starting recursive function
            minNodes = max( nodeValue, self.expectiMax(depth, self.index+1, gameState.generateSuccessor(self.index, action), maximizingPlayer) )
            if minNodes > nodeValue and action != Directions.STOP:
                nodeValue = minNodes
                nextAction = action

        return nextAction

    def expectiMax(self, depth, agent, state, maximizingPlayer):

        # Case we're on a leaf,  we return the  utility value
        # Evaluate the current successor that is terminal
        # self.evaluationFunction defaults to scoreEvaluationFunction
        if depth == self.depth or  state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        # Getting all the actions that the current agent can do
        actions = state.getLegalActions(agent)

        # Agent is Pac-Man
        if maximizingPlayer:
            # Pacman  is max we initialize nodeValue to -infinity
            nodeValue = float('-inf')

            # Once values are initialized and extra cases discarded, we progress to check the lower nodes of the tree
            for action in actions:
                nodeValue = max( nodeValue, self.expectiMax(depth, agent+1, state.generateSuccessor(agent, action), False) )

        else:
            # Setting our expectations - Probabilistically that means a value in range [0,1]
            expectation = 1

            # As suggested, our expectation will be equally distributed over our possible moves
            expectation = 1.0/(1.0*len(actions))
            nodeValue = 0.0


            # For every possible legal action
            nextDepth = depth
            for action in actions:
                # If agent is last agent, next is first
                if agent == state.getNumAgents()-1:
                    maximizingPlayer = True # Pacman next
                    nextDepth = depth+1
                    agent = AGENT-1 # Will be 0
                nodeValue = nodeValue + (expectation * self.expectiMax(nextDepth, agent+1, state.generateSuccessor(agent, action), maximizingPlayer))

        return nodeValue

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: We are going to calculate a  positive linear transformation of the probability of
      the expected utility of a position taking into account the distance to the nearest food,
      the distance to the nearest ghost and the distance to the nearest scaredGhost.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()

    # Store and order the food by distance to PacMan agent
    currentFoods = currentGameState.getFood().asList()

    if(len(currentFoods)!=0):
        minDistToFood = min([util.manhattanDistance(newPos, pos) for pos in currentFoods])
    else: # No food remaining?
        return currentGameState.getScore()

    # Store and order the ghosts by distance to PacMan agent
    newGhostStates = currentGameState.getGhostStates()
    nearestGhost = min([(util.manhattanDistance(newPos,ghost.getPosition()), ghost) for ghost in newGhostStates])
    minDistToGhost = nearestGhost[0]

    newScaredTimes = nearestGhost[1].scaredTimer

    returnScore = currentGameState.getScore()

    # Get closest food and ghost to PacMan agent

    # If ghosts are not scared
    if newScaredTimes == 0:
        # If pacman steps on ghost, game over
        if minDistToGhost == 0:
            # PacMan dies
            return float('-inf')
    # If ghosts are scared
    else:
        # You get points for trying to eat them
        returnScore += (100 - minDistToGhost)
    # The greater the distance to a ghost, the greater the reward
    # scoreGamma is a penalty factor
    scoreGamma = 1.0 / minDistToGhost

    # If PacMan steps on food, eat it
    if minDistToFood  == 0:
        # Even if PacMan is to get a food, if too close to ghost, less reward
        returnScore += (10.0 - pow(scoreGamma, 2.0))
    else:
        # PacMan has a negative reward if it is too far from food and too close to a ghost
        returnScore += (1.0 / minDistToFood - pow(scoreGamma, 1.5))

    return returnScore

# Abbreviation
better = betterEvaluationFunction

