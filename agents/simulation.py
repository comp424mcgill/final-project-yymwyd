from random_agent import *
import numpy as np
from time import sleep, time
#1 simulation
            
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

def simulation(chess_board, my_pos, adv_pos):
        self_turn = 1
        p0 = RandomAgent()
        p1 = RandomAgent()
        cur_player=p0
        cur_pos=my_pos

        p0_step=0
        p1_step=0
        p0_time = 0
        p1_time=0
        board_size = chess_board[0].shape[0]
        max_step = (board_size +1) //2

        with time_limit(2):
            if not self_turn:
                p0_step += 1
            else:
                p1_step += 1

            start_time = time()
            
            next_pos, dir = cur_player.step(
                    deepcopy(chess_board),
                    tuple(cur_pos),
                    tuple(adv_pos),
                    max_step,
                )

            #update player time
            if not self_turn:
                p0_time += time() - start_time
            else:
                p1_time += time() - start_time


        next_pos = np.asarray(next_pos, dtype=cur_pos.dtype)

        next_pos, dir = random_walk(tuple(cur_pos), tuple(adv_pos),max_step,chess_board)
        next_pos = np.asarray(next_pos, dtype=cur_pos.dtype)
        
if __name__ == "__main__":
        chess_board = np.zeros((4,4, 4), dtype=bool)
        chess_board[0, :, 0] = True
        chess_board[:, 0, 3] = True
        chess_board[-1, :, 2] = True
        chess_board[:, -1, 1] = True

        my_pos = (2,3)
        adv_pos = (0,1)
        simulation(chess_board, my_pos, adv_pos)


        
