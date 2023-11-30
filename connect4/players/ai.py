import random
import time
import numpy as np
from typing import Tuple, Dict, Union, List
from connect4.utils import get_pts, get_diagonals_primary, get_diagonals_secondary, get_valid_actions, Integer

class AIPlayer:

    def __init__(self, player_number: int, time: int):
        """
        :param player_number: Current player number
        :param time: Time per move (seconds)
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.time = time
        self.buffer = 1

        # Do the rest of your implementation here

    def get_move_number(self, state):
        board, _ = state
        move_num = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == self.player_number:
                    move_num += 1
        return move_num

    def get_total_moves(self, state):
        return len(state[0])*len(state[0][0])//2

    def col_fraction(self, state):
        n_cols = state[0].shape[1]
        col_0 = 0
        for col in range(n_cols):
            if 0 in state[0][:, col]:
                col_0 += 1
        if col_0 != 0:
            return n_cols//col_0
        return 5

    def get_next_state(self, col_num, is_popOut, mat, pops, player_num):
        popped = False
        if not is_popOut:
            for i in range(len(mat)-1, -1, -1):
                if mat[i][col_num] == 0:
                    mat[i][col_num] = player_num
                    break
        else:
            for i in range(len(mat)-1, 0, -1):
                if mat[i][col_num] != 0:
                    mat[i][col_num] = mat[i-1][col_num]
                    popped = True
                else:
                    break
            mat[0][col_num] = 0
            if (popped):
                pops[player_num].decrement()
        return popped

    def get_weight_central(self, state, min_weight, max_weight, flat_dist):
        total_moves = self.get_total_moves(state)
        move_number = self.get_move_number(state)
        a = 4*(max_weight - min_weight) / ((total_moves - flat_dist)**2)
        w1 = a*(move_number**2) + min_weight
        w2 = a*((total_moves - move_number)**2) + min_weight
        if (move_number > (total_moves + flat_dist)//2):
            return w2
        elif (move_number < (total_moves - flat_dist)//2):
            return w1
        else:
            return max_weight

    def get_weight_inc(self, state, min_weight, max_weight, flat_dist):
        total_moves = self.get_total_moves(state)
        move_number = self.get_move_number(state)
        a = (max_weight - min_weight) / ((total_moves - flat_dist)**2)
        w = a*(move_number**2) + min_weight
        return w

    def get_weight_quad(self, state, min_weight, max_weight, final_factor):
        total_moves = final_factor*self.get_total_moves(state)
        move_number = self.get_move_number(state) % total_moves
        a = 4*(max_weight - min_weight)/(total_moves)**2
        w = a*move_number * (total_moves - move_number) + min_weight
        return w

    def get_weight_linear(self, state, init_weight, final_weight):
        m = (final_weight-init_weight)/(self.get_total_moves(state))
        w = m*self.get_move_number(state) + init_weight
        return w

    def get_weight_fast(self, state, min_weight, max_weight):
        total_moves = self.get_total_moves(state)
        if (self.get_move_number(state) < total_moves//2):
            return self.get_weight_quad(state, min_weight, max_weight, 1)
        else:
            return self.get_weight_central(state, min_weight, max_weight, total_moves//8)

    def get_weight_sigmoid(self, state, min_weight, max_weight):
        total_moves = self.get_total_moves(state)
        if (self.get_move_number(state) < total_moves//2):
            return self.get_weight_central(state, min_weight, max_weight//2, 0)
        else:
            return self.get_weight_quad(state, max_weight//2, max_weight, 2)

    def getExpectimaxUtility(self, board, num_popouts):
        num1 = get_pts(self.player_number, board)
        num2 = get_pts(3-self.player_number, board)
        return (num1 - (2*(num2*num2))/(num1+0.1))

    def get_dynamic_weight(self, state):
        num1 = get_pts(self.player_number, state[0])
        num2 = get_pts(3-self.player_number, state[0])
        return (num1 - (2*(num2*num2))/(num1+0.1))

    def getMinimaxUtility(self, board, num_popouts):
        return self.get_dynamic_weight((board, num_popouts))

    def feature_extract(self, player_number: int, row: Union[np.array, List[int]]):
        n = len(row)
        j = 0
        res = 0
        while j < n:
            if row[j] == player_number:
                count = 0
                while j < n and row[j] == player_number:
                    count += 1
                    j += 1
                if count % 4 == 3:
                    res += 1
            else:
                j += 1
        return res

    def added_score(self, w1, w2, board: np.array) -> int:
        """
        :return: Returns the total score of player (with player number) on the board
        """
        res = 0
        opp_res = 0
        m, n = board.shape
        # score in rows
        for i in range(m):
            res += self.feature_extract(self.player_number, board[i])
            opp_res += self.feature_extract(3 - self.player_number, board[i])
        # score in columns
        for j in range(n):
            res += self.feature_extract(self.player_number, board[:, j])
            opp_res += self.feature_extract(3 -
                                            self.player_number, board[:, j])
        # scores in diagonals_primary
        for diag in get_diagonals_primary(board):
            res += self.feature_extract(self.player_number, diag)
            opp_res += self.feature_extract(3 - self.player_number, diag)
        # scores in diagonals_secondary
        for diag in get_diagonals_secondary(board):
            res += self.feature_extract(self.player_number, diag)
            opp_res += self.feature_extract(3 - self.player_number, diag)

        # return res*w1 - opp_res*w2
        return res*w1 - opp_res*w2

    def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        """
        Given the current state of the board, return the next move
        This will play against either itself or a human player
        :param state: Contains:
                        1. board
                            - a numpy array containing the state of the board using the following encoding:
                            - the board maintains its same two dimensions
                                - row 0 is the top of the board and so is the last row filled
                            - spaces that are unoccupied are marked as 0
                            - spaces that are occupied by player 1 have a 1 in them
                            - spaces that are occupied by player 2 have a 2 in them
                        2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
        :return: action (0 based index of the column and if it is a popout move)
        """

        def minimax(state, depth, start_time):
            return max_value(state, depth, -np.inf, np.inf, start_time)

        def max_value(state, depth, alpha, beta, start_time):
            value = -np.inf
            best_action = None
            board, popouts = state
            actions = get_valid_actions(self.player_number, state)

            if depth == 0 or len(actions) == 0 or time.time() - start_time >= self.time - self.buffer:
                return best_action, (self.getMinimaxUtility(board, popouts))

            while len(actions) > 0:
                col, is_popOut = random.choice(actions)
                new_mat = board.copy()
                new_pops = popouts.copy()
                popped = self.get_next_state(
                    col, is_popOut, new_mat, new_pops, self.player_number)

                if time.time() - start_time >= self.time - self.buffer:
                    if self.getMinimaxUtility(new_mat, new_pops) > value:
                        value = self.getMinimaxUtility(new_mat, new_pops)
                        best_action = (col, is_popOut)
                    if popped:
                        new_pops[self.player_number].increment()
                    break

                _, new_value = min_value(
                    (new_mat, new_pops), depth - 1, alpha, beta, start_time)

                if new_value > value:
                    value = new_value
                    best_action = (col, is_popOut)
                if popped:
                    new_pops[self.player_number].increment()

                if value >= beta or (time.time() - start_time >= self.time - self.buffer):
                    break
                alpha = max(alpha, value)
                actions.remove((col, is_popOut))
            return best_action, value

        def min_value(state, depth, alpha, beta, start_time):
            value = np.inf
            best_action = None
            board, popouts = state
            actions = get_valid_actions(3 - self.player_number, state)
            if depth == 0 or len(actions) == 0 or time.time() - start_time >= self.time - self.buffer:
                return best_action, (self.getMinimaxUtility(board, popouts))

            while len(actions) > 0:
                col, is_popOut = random.choice(actions)
                new_mat = board.copy()
                new_pops = popouts.copy()
                popped = self.get_next_state(
                    col, is_popOut, new_mat, new_pops, 3 - self.player_number)

                if time.time() - start_time >= self.time - self.buffer:
                    if self.getMinimaxUtility(new_mat, new_pops) < value:
                        value = self.getMinimaxUtility(new_mat, new_pops)
                        best_action = (col, is_popOut)
                    if popped:
                        new_pops[3-self.player_number].increment()
                    break

                _, new_value = max_value(
                    (new_mat, new_pops), depth - 1, alpha, beta, start_time)

                if new_value < value:
                    value = new_value
                    best_action = (col, is_popOut)
                if popped:
                    new_pops[3-self.player_number].increment()

                if value <= alpha or (time.time() - start_time >= self.time - self.buffer):
                    break
                beta = min(beta, value)
                actions.remove((col, is_popOut))
            return best_action, value

        start_time = time.time()

        depth = 3 + self.col_fraction(state)
        if (depth > 10):
            depth = 10
        max_action, _ = minimax(state, depth, start_time)

        return max_action

        # Do the rest of your implementation here
        # raise NotImplementedError('Whoops I don\'t know what to do')

    def get_expectimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        """
        Given the current state of the board, return the next move based on
        the Expecti max algorithm.
        This will play against the random player, who chooses any valid move
        with equal probability
        :param state: Contains:
                        1. board
                            - a numpy array containing the state of the board using the following encoding:
                            - the board maintains its same two dimensions
                                - row 0 is the top of the board and so is the last row filled
                            - spaces that are unoccupied are marked as 0
                            - spaces that are occupied by player 1 have a 1 in them
                            - spaces that are occupied by player 2 have a 2 in them
                        2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
        :return: action (0 based index of the column and if it is a popout move)
        """

        def expectimax(state, depth, alpha, beta, start_time, isThisPlayerAI):
            best_action = None
            board, popouts = state

            if (isThisPlayerAI):
                value = -np.inf
                actions = get_valid_actions(self.player_number, state)

                if depth == 0 or len(actions) == 0 or time.time() - start_time >= self.time - self.buffer:
                    return best_action, (self.getExpectimaxUtility(board, popouts))

                while len(actions) > 0:
                    col, is_popOut = random.choice(actions)
                    new_mat = board.copy()
                    new_pops = popouts.copy()
                    popped = self.get_next_state(
                        col, is_popOut, new_mat, new_pops, self.player_number)

                    if time.time() - start_time >= self.time - self.buffer:
                        if self.getExpectimaxUtility(new_mat, new_pops) > value:
                            value = self.getExpectimaxUtility(
                                new_mat, new_pops)
                            best_action = (col, is_popOut)
                        if popped:
                            new_pops[self.player_number].increment()
                        break

                    _, new_value = expectimax(
                        (new_mat, new_pops), depth - 1, alpha, beta, start_time, False)

                    if new_value > value:
                        value = new_value
                        best_action = (col, is_popOut)
                    if popped:
                        new_pops[self.player_number].increment()

                    if value >= beta or (time.time() - start_time >= self.time - self.buffer):
                        break
                    alpha = max(alpha, value)
                    actions.remove((col, is_popOut))
                return best_action, value
            else:
                value = np.inf
                opp_cost = self.getExpectimaxUtility(board, popouts)
                explored = 0
                actions = get_valid_actions(3 - self.player_number, state)
                if depth == 0 or len(actions) == 0 or time.time() - start_time >= self.time - self.buffer:
                    return best_action, opp_cost
                opp_cost = 0
                while len(actions) > 0:
                    explored += 1
                    col, is_popOut = random.choice(actions)
                    new_mat = board.copy()
                    new_pops = popouts.copy()
                    popped = self.get_next_state(
                        col, is_popOut, new_mat, new_pops, 3 - self.player_number)

                    if time.time() - start_time >= self.time - self.buffer:
                        if self.getExpectimaxUtility(new_mat, new_pops) < value:
                            value = self.getExpectimaxUtility(
                                new_mat, new_pops)
                            best_action = (col, is_popOut)
                        if popped:
                            new_pops[3-self.player_number].increment()
                        break

                    _, new_value = expectimax(
                        (new_mat, new_pops), depth - 1, alpha, beta, start_time, True)
                    opp_cost += value

                    if new_value < value:
                        value = new_value
                        best_action = (col, is_popOut)
                    if popped:
                        new_pops[3-self.player_number].increment()

                    if value <= alpha or (time.time() - start_time >= self.time - self.buffer):
                        break
                    beta = min(beta, value)
                    actions.remove((col, is_popOut))
                if (explored > 0):
                    opp_cost /= explored
                return best_action, value

        start_time = time.time()
        depth = 3 + self.col_fraction(state)
        max_action, _ = expectimax(
            state, depth, -np.inf, np.inf, start_time, True)
        return max_action
