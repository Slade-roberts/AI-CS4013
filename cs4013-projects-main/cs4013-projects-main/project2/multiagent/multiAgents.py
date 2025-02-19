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

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        # Base score from the game's evaluation function
        score = successorGameState.getScore()

        # Food-related scoring: prioritize moving closer to the nearest food pellet
        foodList = newFood.asList()
        if foodList:
            closestFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
            score += 10.0 / (closestFoodDist + 1)  # Encourage getting closer to food

        # Ghost-related scoring: avoid ghosts if they are not scared
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)

            if ghostState.scaredTimer > 0:
                # If ghosts are scared, encourage chasing them
                score += 50.0 / (ghostDist + 1)
            else:
                # If ghosts are dangerous, strongly discourage moving close
                if ghostDist < 2:
                    score -= 100  # Penalize heavily for immediate danger

        # Penalize stopping unless no other options exist
        if action == Directions.STOP:
            score -= 10

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(state, agentIndex, depth):
            """
            Recursively performs the minimax algorithm.
            """
            # Check for terminal state (win/loss or max depth reached)
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Pacman (maximizing agent)
            if agentIndex == 0:
                return max_value(state, agentIndex, depth)

            # Ghosts (minimizing agents)
            else:
                return min_value(state, agentIndex, depth)

        def max_value(state, agentIndex, depth):
            """
            Maximizing function for Pacman.
            """
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:  # No available moves
                return self.evaluationFunction(state)

            # Find the max score among all successors
            return max(minimax(state.generateSuccessor(agentIndex, action), 1, depth) for action in legalActions)

        def min_value(state, agentIndex, depth):
            """
            Minimizing function for ghosts.
            """
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:  # No available moves
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()
            if agentIndex == numAgents - 1:  # Last ghost, go to next Pacman layer
                return min(minimax(state.generateSuccessor(agentIndex, action), 0, depth + 1) for action in legalActions)
            else:  # Next ghost in the same depth layer
                return min(minimax(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth) for action in legalActions)

        # Choose the action leading to the best minimax value
        legalMoves = gameState.getLegalActions(0)
        bestAction = max(legalMoves, key=lambda action: minimax(gameState.generateSuccessor(0, action), 1, 0))
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(state, agentIndex, depth, alpha, beta):
            """
            Performs Minimax with Alpha-Beta Pruning.
            """
            # Terminal condition: game over or max depth reached
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Pacman (Maximizing agent)
            if agentIndex == 0:
                return max_value(state, agentIndex, depth, alpha, beta)

            # Ghosts (Minimizing agents)
            else:
                return min_value(state, agentIndex, depth, alpha, beta)

        def max_value(state, agentIndex, depth, alpha, beta):
            """
            Maximizing function for Pacman.
            """
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:  # No available moves
                return self.evaluationFunction(state)

            v = float("-inf")
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, alphabeta(successor, 1, depth, alpha, beta))

                # Alpha-Beta Pruning
                if v > beta:
                    return v
                alpha = max(alpha, v)

            return v

        def min_value(state, agentIndex, depth, alpha, beta):
            """
            Minimizing function for ghosts.
            """
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:  # No available moves
                return self.evaluationFunction(state)

            v = float("inf")
            numAgents = state.getNumAgents()

            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)

                # If this is the last ghost, go to Pacman (depth + 1)
                if agentIndex == numAgents - 1:
                    v = min(v, alphabeta(successor, 0, depth + 1, alpha, beta))
                else:
                    v = min(v, alphabeta(successor, agentIndex + 1, depth, alpha, beta))

                # Alpha-Beta Pruning
                if v < alpha:
                    return v
                beta = min(beta, v)

            return v

        # Start Alpha-Beta Search
        legalMoves = gameState.getLegalActions(0)
        bestAction = None
        alpha, beta = float("-inf"), float("inf")
        bestScore = float("-inf")

        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)
            score = alphabeta(successor, 1, 0, alpha, beta)

            if score > bestScore:
                bestScore = score
                bestAction = action

            alpha = max(alpha, bestScore)

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction.

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(state, depth, agentIndex):
            """
            Recursively performs the expectimax algorithm.
            """
            # Terminal condition: game over or depth limit reached
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Pacman (Max Node)
            if agentIndex == 0:
                return max(expectimax(state.generateSuccessor(agentIndex, action), depth, 1)
                           for action in state.getLegalActions(agentIndex))

            # Ghosts (Chance Nodes)
            else:
                nextAgent = agentIndex + 1 if agentIndex < state.getNumAgents() - 1 else 0
                nextDepth = depth + 1 if nextAgent == 0 else depth

                actions = state.getLegalActions(agentIndex)
                if not actions:
                    return self.evaluationFunction(state)

                # Calculate expected value by averaging over all actions
                return sum(expectimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent) 
                           for action in actions) / len(actions)

        # Choose the action leading to the highest expectimax value
        legalMoves = gameState.getLegalActions(0)
        bestAction = max(legalMoves, key=lambda action: expectimax(gameState.generateSuccessor(0, action), 0, 1))
        return bestAction

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
