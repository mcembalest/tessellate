#!/usr/bin/env python3
"""
Tessellate Game Implementation
Ported from JavaScript to Python for RL training
"""

from enum import IntEnum
from typing import List, Tuple, Optional

class TileState(IntEnum):
    EMPTY = 0
    RED = 1
    BLUE = 2
    BLOCKED = 3  # OUT_OF_PLAY in JS

class TessellateGame:
    def __init__(self):
        self.VISUAL_GRID_SIZE = 5
        self.LOGICAL_GRID_SIZE = 10  # VISUAL_GRID_SIZE * 2
        self.TOTAL_TILES = 50  # VISUAL_GRID_SIZE * VISUAL_GRID_SIZE * 2
        
        self.board = None
        self.current_turn = None
        self.scores = None
        self.game_over = False
        self.placed_tiles_count = 0
        self.move_history = []
        
        self.reset()
    
    def reset(self):
        """Initialize/reset the game board"""
        self.board = [[TileState.EMPTY for _ in range(self.LOGICAL_GRID_SIZE)] 
                      for _ in range(self.LOGICAL_GRID_SIZE)]
        self.current_turn = TileState.RED
        self.scores = {TileState.RED: 1, TileState.BLUE: 1}
        self.game_over = False
        self.placed_tiles_count = 0
        self.move_history = []
        return self.board.copy()
    
    def is_valid_coord(self, r: int, c: int) -> bool:
        """Check if coordinates are within board bounds"""
        return 0 <= r < self.LOGICAL_GRID_SIZE and 0 <= c < self.LOGICAL_GRID_SIZE
    
    def is_playable(self, r: int, c: int) -> bool:
        """Check if a position is playable (valid and empty)"""
        return self.is_valid_coord(r, c) and self.board[r][c] == TileState.EMPTY
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid moves for current player"""
        moves = []
        for r in range(self.LOGICAL_GRID_SIZE):
            for c in range(self.LOGICAL_GRID_SIZE):
                if self.is_playable(r, c):
                    moves.append((r, c))
        return moves
    
    def add_tile(self, r: int, c: int) -> bool:
        """
        Add a tile at position (r, c) and block adjacent corners.
        This is the core game mechanic from JavaScript.
        """
        if not self.is_playable(r, c):
            return False
        
        # Place the tile
        self.board[r][c] = self.current_turn
        self.placed_tiles_count += 1
        
        # Block horizontally adjacent corner in same square
        c_adj = c + (1 if c % 2 == 0 else -1)
        if self.is_valid_coord(r, c_adj):
            if self.board[r][c_adj] == TileState.EMPTY:
                self.board[r][c_adj] = TileState.BLOCKED
        
        # Block vertically adjacent corner in same square  
        r_adj = r + (1 if r % 2 == 0 else -1)
        if self.is_valid_coord(r_adj, c):
            if self.board[r_adj][c] == TileState.EMPTY:
                self.board[r_adj][c] = TileState.BLOCKED
        
        return True
    
    def get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """
        Get all neighbors that share a full edge (hypotenuse or leg).
        This is the complex adjacency logic from JavaScript.
        """
        neighbors = []
        
        # These calculations come directly from the JS implementation
        pow_neg1_r = 1 if r % 2 == 0 else -1
        pow_neg1_c = 1 if c % 2 == 0 else -1
        pow_neg1_r_plus_1 = 1 if (r + 1) % 2 == 0 else -1
        pow_neg1_c_plus_1 = 1 if (c + 1) % 2 == 0 else -1
        pow_neg1_r_c_1 = 1 if (r + c + 1) % 2 == 0 else -1
        
        # Five potential neighbors based on the JS logic
        potential = [
            (r + pow_neg1_r, c + pow_neg1_c),
            (r - 1, c - pow_neg1_r_c_1),
            (r + 1, c + pow_neg1_r_c_1),
            (r + pow_neg1_r_plus_1, c),
            (r, c + pow_neg1_c_plus_1)
        ]
        
        # Filter for valid coordinates
        for nr, nc in potential:
            if self.is_valid_coord(nr, nc):
                neighbors.append((nr, nc))
        
        return neighbors
    
    def calculate_scores(self):
        """Calculate scores by finding islands and multiplying their sizes"""
        self.scores = {TileState.RED: 1, TileState.BLUE: 1}
        visited = [[False for _ in range(self.LOGICAL_GRID_SIZE)] 
                   for _ in range(self.LOGICAL_GRID_SIZE)]
        
        for r in range(self.LOGICAL_GRID_SIZE):
            for c in range(self.LOGICAL_GRID_SIZE):
                color = self.board[r][c]
                if color in [TileState.RED, TileState.BLUE] and not visited[r][c]:
                    # Find island size using DFS
                    island_size = self._dfs_island(r, c, color, visited)
                    if island_size > 0:
                        self.scores[color] *= island_size
    
    def _dfs_island(self, start_r: int, start_c: int, color: TileState, 
                    visited: list) -> int:
        """DFS to find connected component (island) size"""
        stack = [(start_r, start_c)]
        size = 0
        
        while stack:
            r, c = stack.pop()
            if self.is_valid_coord(r, c) and self.board[r][c] == color and not visited[r][c]:
                visited[r][c] = True
                size += 1
                
                # Add all neighbors of the same color
                for nr, nc in self.get_neighbors(r, c):
                    if self.is_valid_coord(nr, nc) and self.board[nr][nc] == color and not visited[nr][nc]:
                        stack.append((nr, nc))
        
        return size
    
    def make_move(self, r: int, c: int) -> bool:
        """Make a move and update game state"""
        if self.game_over:
            return False
        
        # Store move history
        prev_scores = self.scores.copy()
        board_snapshot = self.board.copy()
        
        # Make the move
        if not self.add_tile(r, c):
            return False
        
        self.move_history.append({
            'r': r,
            'c': c, 
            'player': self.current_turn,
            'prev_scores': prev_scores,
            'board_snapshot': board_snapshot
        })
        
        # Update scores
        self.calculate_scores()
        
        # Switch turns
        self.current_turn = TileState.BLUE if self.current_turn == TileState.RED else TileState.RED
        
        # Check game over
        if self.placed_tiles_count >= self.TOTAL_TILES:
            self.game_over = True
        
        return True
    
    def get_winner(self) -> Optional[TileState]:
        """Get the winner of the game (None if tie or not over)"""
        if not self.game_over:
            return None
        
        red_score = self.scores[TileState.RED]
        blue_score = self.scores[TileState.BLUE]
        
        if red_score > blue_score:
            return TileState.RED
        elif blue_score > red_score:
            return TileState.BLUE
        else:
            return None  # Tie
    
    def render_ascii(self) -> str:
        """Simple ASCII visualization for debugging"""
        symbols = {
            TileState.EMPTY: '.',
            TileState.RED: 'R',
            TileState.BLUE: 'B',
            TileState.BLOCKED: 'x'
        }
        
        result = []
        result.append(f"Turn: {'RED' if self.current_turn == TileState.RED else 'BLUE'}")
        result.append(f"Scores: RED={self.scores[TileState.RED]}, BLUE={self.scores[TileState.BLUE]}")
        result.append(f"Tiles placed: {self.placed_tiles_count}/{self.TOTAL_TILES}")
        result.append("")
        
        # Board with coordinates
        result.append("   " + " ".join(str(i) for i in range(10)))
        for r in range(self.LOGICAL_GRID_SIZE):
            row_str = f"{r:2} "
            for c in range(self.LOGICAL_GRID_SIZE):
                row_str += symbols[self.board[r][c]] + " "
            result.append(row_str)
        
        return "\n".join(result)

