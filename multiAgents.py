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
from typing import Optional
from math import inf

from pacman import GameState
from game import Agent

import util


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
      This class provides some common elements to all of your multi-agent
      searchers.  Any methods defined here will be available to the
      MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.
      It's only partially specified, and designed to be extended.  Agent
      (game.py) is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, game_state: GameState) -> str:
        """
          Returns the minimax action from the current gameState using
          self.depth and self.evaluationFunction.

          Here are some method calls that might be useful when implementing
          minimax.

          gameState.getLegalActions(agentIndex): Returns a list of legal
          actions for an agent agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action): Returns the
          successor game state after an agent takes an action

          gameState.getNumAgents(): Returns the total number of agents in the
          game """
        legal_actions = game_state.getLegalActions(agentIndex=0)

        best_action_index = max(
            range(len(legal_actions)),
            key=lambda action_num: self.min_value(
                state=game_state.generateSuccessor(
                    agentIndex=0,
                    action=legal_actions[action_num],
                ),
                depth=self.depth,
                ghost_num=1,
            )
        )
        return legal_actions[best_action_index]

    def max_value(
            self,
            state: GameState,
            depth: int,
            actor: Optional[int] = None
    ) -> int:
        # Sanity check: have all the ghosts been evaluated the last round?
        if actor is not None:
            assert actor == state.getNumAgents()

        # Game over or search depth has been reached
        if state.isLose() or state.isWin() or depth <= 0:
            return self.evaluationFunction(state)

        legal_actions = state.getLegalActions(agentIndex=0)
        successors = [
            state.generateSuccessor(agentIndex=0, action=action)
            for action
            in legal_actions
        ]
        utilities = [
            self.min_value(state, depth, ghost_num=1)
            for state
            in successors
        ]

        return max(utilities)

    def min_value(self, state: GameState, depth: int, ghost_num: int) -> int:

        # Game over or search depth has been reached
        if state.isLose() or state.isWin() or depth <= 0:
            return self.evaluationFunction(state)

        # Sanity check: valid ghost number?
        assert 1 <= ghost_num < state.getNumAgents()

        legal_actions = state.getLegalActions(ghost_num)

        successors = [
            state.generateSuccessor(ghost_num, ghost_action)
            for ghost_action
            in legal_actions
        ]

        # If this is the last ghost, next optimizer should be from pacman's
        # perspective
        next_optimizer = self.max_value \
            if ghost_num == state.getNumAgents() - 1 \
            else self.min_value

        # If this is the last ghost, decrement depth
        next_depth = depth - 1 \
            if ghost_num == state.getNumAgents() - 1 \
            else depth

        utilities = [
            next_optimizer(state, next_depth, ghost_num + 1)
            for state
            in successors
        ]

        return min(utilities)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, game_state: GameState) -> str:
        """
          Returns the minimax action from the current gameState using
          self.depth and self.evaluationFunction.

          Here are some method calls that might be useful when implementing
          minimax.

          gameState.getLegalActions(agentIndex): Returns a list of legal
          actions for an agent agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action): Returns the
          successor game state after an agent takes an action

          gameState.getNumAgents(): Returns the total number of agents in the
          game
        """
        legal_actions = game_state.getLegalActions(agentIndex=0)

        alpha, beta = -inf, inf
        utility = -inf

        for action_num in range(len(legal_actions)):
            successor = game_state.generateSuccessor(
                agentIndex=0,
                action=legal_actions[action_num],
            )
            utility = max(
                utility,
                self.min_value(
                    successor,
                    depth=self.depth,
                    alpha=alpha,
                    beta=beta,
                    ghost_num=1,
                ),
            )

            if utility > alpha:
                best_action_index = action_num
                alpha = utility

        return legal_actions[best_action_index]

    def max_value(
        self,
        state: GameState,
        depth: int,
        alpha: int,
        beta: int,
        actor: Optional[int] = None,
    ) -> int:
        # Sanity check: have all the ghosts been evaluated the last round?
        if actor is not None:
            assert actor == state.getNumAgents()

        # Game over or search depth has been reached
        if state.isLose() or state.isWin() or depth <= 0:
            return self.evaluationFunction(state)

        legal_actions = state.getLegalActions(agentIndex=0)

        utility = -inf
        for action in legal_actions:
            successor = state.generateSuccessor(agentIndex=0, action=action)
            utility = max(
                utility,
                self.min_value(successor, depth, alpha, beta, ghost_num=1),
            )

            if utility > beta:
                return utility

            alpha = max(alpha, utility)

        return utility

    def min_value(
        self,
        state: GameState,
        depth: int,
        alpha: int,
        beta: int,
        ghost_num: int,
    ) -> int:

        # Game over or search depth has been reached
        if state.isLose() or state.isWin() or depth <= 0:
            return self.evaluationFunction(state)

        # Sanity check: valid ghost number?
        assert 1 <= ghost_num < state.getNumAgents()

        legal_actions = state.getLegalActions(ghost_num)

        # If this is the last ghost, next optimizer should be from pacman's
        # perspective
        next_optimizer = self.max_value \
            if ghost_num == state.getNumAgents() - 1 \
            else self.min_value

        # If this is the last ghost, decrement depth
        next_depth = depth - 1 if ghost_num == state.getNumAgents() - 1 else depth

        utility = inf
        for action in legal_actions:
            successor = state.generateSuccessor(
                agentIndex=ghost_num,
                action=action,
            )
            utility = min(
                utility,
                next_optimizer(
                    successor,
                    next_depth,
                    alpha,
                    beta,
                    ghost_num + 1,
                ),
            )

            if utility < alpha:
                return utility

            beta = min(beta, utility)

        return utility


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
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
