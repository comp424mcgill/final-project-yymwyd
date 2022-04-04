import numpy as np
from copy import deepcopy
from time import sleep, time

import signal
from contextlib import contextmanager

#1 simulation
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Execution timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
            
def random_walk(my_pos, adv_pos,max_step, chess_board):
        """
        Randomly walk to the next position in the board.
        Parameters
        ----------
        my_pos : tuple
            The position of the agent.
        adv_pos : tuple
            The position of the adversary.
        """
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        
        ori_pos = deepcopy(my_pos)
        steps = np.random.randint(0, max_step + 1)
        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir

def check_endgame(board_size,chess_board,p0_pos,p1_pos):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        
        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)
                
        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]
        
        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(p0_pos))
        p1_r = find(tuple(p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        return True, p0_score, p1_score


        


#declare the global variables
self_turn = 0

p0_step=0
p1_step=0
p0_time = 0
p1_time=0
#random initialize p0 and p1 position, remember to change it after expansion
p0_pos = [0,0]
p1_pos = [1,1]

def simulation_step(chess_board, my_pos, adv_pos):

        cur_pos=np.asarray(my_pos)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        board_size = chess_board[0].shape[0]
        max_step = (board_size +1) //2
        global self_turn
        global p0_pos
        global p1_pos

        if not self_turn:
                global p0_step
                cur_player_step = p0_step
        else:
                global p1_step
                cur_player_step = p1_step

        # Get allowed time in this step
        if cur_player_step == 0:
                allowed_time_seconds = 30
        else:
                allowed_time_seconds = 2

        with time_limit(allowed_time_seconds):
                if not self_turn:
                        p0_step += 1
                else:
                        p1_step += 1

                start_time = time()
                next_pos, dir = random_walk(tuple(cur_pos), tuple(adv_pos),max_step,chess_board)

                #update player time
                if not self_turn:
                        global p0_time
                        p0_time += time() - start_time
                else:
                        global p1_time
                        p1_time += time() - start_time

                next_pos = np.asarray(next_pos, dtype=cur_pos.dtype)


        if not self_turn:
                        p0_pos = next_pos
        else:
                        p1_pos = next_pos

                        
        # Set the barrier to True
        r, c = next_pos
        chess_board[r, c, dir] = True
        move = moves[dir]
        chess_board[r+ move[0], c + move[1], opposites[dir]] = True


        # Change turn
        self_turn = 1 - self_turn

        results = check_endgame(board_size,chess_board,p0_pos,p1_pos)

        return chess_board, results
        #remember to return the updated chessboard

def run_simulation():
        
                

            

if __name__ == "__main__":
        chess_board = np.zeros((4,4, 4), dtype=bool)
        chess_board[0, :, 0] = True
        chess_board[:, 0, 3] = True
        chess_board[-1, :, 2] = True
        chess_board[:, -1, 1] = True

        my_pos = (2,3)
        adv_pos = (0,1)
        print(chess_board)
        print(simulation_step(chess_board, my_pos, adv_pos))


        
