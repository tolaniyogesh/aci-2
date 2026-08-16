"""
Hex Game AI Agent — Assignment 2

This module implements the Hex board game with an AI agent using
Monte Carlo Tree Search (MCTS).

"""

import time
import math
import random
import sys

# =============================================================================
# Constants
# =============================================================================

EMPTY = 0
PLAYER_A = 1  # AI — connects top row (row 0) to bottom row (row N-1)
PLAYER_B = 2  # Human — connects left col (col 0) to right col (col N-1)

# Hex adjacency offsets: 4 orthogonal + 2 allowed diagonals
# Forbidden: top-left (r-1, c-1) and bottom-right (r+1, c+1)
HEX_DIRECTIONS = [
    (-1,  0),  # Top
    ( 1,  0),  # Bottom
    ( 0, -1),  # Left
    ( 0,  1),  # Right
    (-1,  1),  # Top-Right   (allowed diagonal)
    ( 1, -1),  # Bottom-Left (allowed diagonal)
]


# =============================================================================
# Board Class
# =============================================================================

class HexBoard:
    """
    Represents an N×N Hex game board using a 2D list (matrix).

    Cell values:
        0 — empty
        1 — Player A (AI, connects rows top-to-bottom)
        2 — Player B (Human, connects columns left-to-right)

    Adjacency model:
        Each cell (r, c) has up to 6 hex-neighbors:
        4 orthogonal + top-right (r-1, c+1) + bottom-left (r+1, c-1).
        Top-left and bottom-right diagonals are forbidden.
    """

    def __init__(self, size):
        """
        Initialize an empty Hex board of the given size.

        Parameters
        ----------
        size : int
            Board dimension N (must satisfy 7 <= N <= 11).

        Raises
        ------
        ValueError
            If size is outside the allowed range [7, 11].
        """
        if not (7 <= size <= 11):
            raise ValueError(
                f"Board size must be between 7 and 11 (inclusive). Got: {size}"
            )
        self.size = size
        self.grid = [[EMPTY] * size for _ in range(size)]
        # Track the number of moves played (useful for full-board detection)
        self.move_count = 0
        # History stack for undo support: stores (row, col, player) tuples
        self._history = []

    # -------------------------------------------------------------------------
    # Adjacency & Neighbor Queries
    # -------------------------------------------------------------------------

    def _is_within_bounds(self, row, col):
        """Check if (row, col) is within the board boundaries."""
        return 0 <= row < self.size and 0 <= col < self.size

    def get_neighbors(self, row, col):
        """
        Return the list of valid hex-adjacent cells for position (row, col).

        Uses 6-directional hex adjacency:
            Orthogonal : (r-1,c), (r+1,c), (r,c-1), (r,c+1)
            Diagonals  : (r-1,c+1) [top-right], (r+1,c-1) [bottom-left]
            Forbidden  : (r-1,c-1) [top-left],  (r+1,c+1) [bottom-right]

        Parameters
        ----------
        row : int
            Row index.
        col : int
            Column index.

        Returns
        -------
        list of tuple
            List of (row, col) tuples representing valid neighbors.
        """
        neighbors = []
        for dr, dc in HEX_DIRECTIONS:
            nr, nc = row + dr, col + dc
            if self._is_within_bounds(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    # -------------------------------------------------------------------------
    # Move Validation
    # -------------------------------------------------------------------------

    def is_valid_move(self, row, col):
        """
        Check whether placing a piece at (row, col) is a legal move.

        A move is valid if:
            1. (row, col) is within board bounds.
            2. The cell at (row, col) is currently empty.

        Parameters
        ----------
        row : int
            Row index of the intended move.
        col : int
            Column index of the intended move.

        Returns
        -------
        bool
            True if the move is valid, False otherwise.
        """
        if not self._is_within_bounds(row, col):
            return False
        if self.grid[row][col] != EMPTY:
            return False
        return True

    def validate_move(self, row, col):
        """
        Validate a move and return an error message if invalid.

        Parameters
        ----------
        row : int
            Row index.
        col : int
            Column index.

        Returns
        -------
        tuple (bool, str)
            (True, "VALID") if the move is legal.
            (False, <error_message>) if the move is illegal.
        """
        if not isinstance(row, int) or not isinstance(col, int):
            return False, f"Invalid input: coordinates must be integers. Got ({row}, {col})"
        if not self._is_within_bounds(row, col):
            return False, (
                f"Move ({row},{col}) is out of bounds. "
                f"Valid range: 0 to {self.size - 1}."
            )
        if self.grid[row][col] != EMPTY:
            occupant = "Player A" if self.grid[row][col] == PLAYER_A else "Player B"
            return False, (
                f"Cell ({row},{col}) is already occupied by {occupant}."
            )
        return True, "VALID"

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def make_move(self, row, col, player):
        """
        Place a piece for the given player at (row, col).

        Parameters
        ----------
        row : int
            Row index.
        col : int
            Column index.
        player : int
            PLAYER_A (1) or PLAYER_B (2).

        Raises
        ------
        ValueError
            If the move is invalid or the player value is incorrect.
        """
        if player not in (PLAYER_A, PLAYER_B):
            raise ValueError(f"Invalid player: {player}. Must be 1 or 2.")
        valid, msg = self.validate_move(row, col)
        if not valid:
            raise ValueError(f"Invalid move: {msg}")

        self.grid[row][col] = player
        self.move_count += 1
        self._history.append((row, col, player))

    def undo_move(self):
        """
        Undo the last move made on the board.

        Returns
        -------
        tuple (int, int, int)
            The (row, col, player) of the undone move.

        Raises
        ------
        IndexError
            If there are no moves to undo (history is empty).
        """
        if not self._history:
            raise IndexError("Cannot undo: no moves have been made.")

        row, col, player = self._history.pop()
        self.grid[row][col] = EMPTY
        self.move_count -= 1
        return row, col, player

    def get_legal_moves(self):
        """
        Return a list of all empty cells on the board.

        Returns
        -------
        list of tuple
            List of (row, col) tuples where a piece can be placed.
        """
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] == EMPTY:
                    moves.append((r, c))
        return moves

    def is_full(self):
        """Check if the board has no empty cells remaining."""
        return self.move_count >= self.size * self.size

    def clone(self):
        """
        Create a deep copy of the current board state.

        Returns
        -------
        HexBoard
            A new HexBoard instance with identical state.
        """
        new_board = HexBoard.__new__(HexBoard)
        new_board.size = self.size
        new_board.grid = [row[:] for row in self.grid]  # shallow copy of rows
        new_board.move_count = self.move_count
        new_board._history = list(self._history)  # copy history
        return new_board

    def load_state(self, grid):
        """
        Load a board state from a 2D list.

        Parameters
        ----------
        grid : list of list of int
            An N×N matrix with values 0, 1, or 2.

        Raises
        ------
        ValueError
            If the grid dimensions don't match or contain invalid values.
        """
        if len(grid) != self.size:
            raise ValueError(
                f"Grid has {len(grid)} rows, expected {self.size}."
            )
        self.move_count = 0
        for r in range(self.size):
            if len(grid[r]) != self.size:
                raise ValueError(
                    f"Row {r} has {len(grid[r])} columns, expected {self.size}."
                )
            for c in range(self.size):
                val = grid[r][c]
                if val not in (EMPTY, PLAYER_A, PLAYER_B):
                    raise ValueError(
                        f"Invalid cell value {val} at ({r},{c}). "
                        f"Must be 0, 1, or 2."
                    )
                self.grid[r][c] = val
                if val != EMPTY:
                    self.move_count += 1
        self._history = []  # history not available for loaded states

    # -------------------------------------------------------------------------
    # Win Detection
    # -------------------------------------------------------------------------

    def check_winner(self, player):
        """
        Check whether the given player has formed a winning connection.

        Player A (1): connects the TOP row to the BOTTOM row.
        Player B (2): connects the LEFT column to the RIGHT column.

        Uses DFS with the Hex-specific six-neighbor adjacency rules.

        Returns
        -------
        bool
            True if the player has a connected path between their target sides,
            otherwise False.
        """
        if player not in (PLAYER_A, PLAYER_B):
            raise ValueError(f"Invalid player: {player}. Must be 1 or 2.")

        visited = set()
        stack = []

        # Starting cells:
        # Player A starts from every Player-A piece in the top row.
        # Player B starts from every Player-B piece in the left column.
        if player == PLAYER_A:
            for col in range(self.size):
                if self.grid[0][col] == player:
                    stack.append((0, col))
                    visited.add((0, col))
        else:
            for row in range(self.size):
                if self.grid[row][0] == player:
                    stack.append((row, 0))
                    visited.add((row, 0))

        # DFS through the six valid Hex neighbors.
        while stack:
            row, col = stack.pop()

            # Check whether the target side has been reached.
            if player == PLAYER_A and row == self.size - 1:
                return True

            if player == PLAYER_B and col == self.size - 1:
                return True

            for nr, nc in self.get_neighbors(row, col):
                if (nr, nc) not in visited and self.grid[nr][nc] == player:
                    visited.add((nr, nc))
                    stack.append((nr, nc))

        return False

    def get_winner(self):
        """Return the winning player (1 or 2), or None if the game continues."""
        if self.check_winner(PLAYER_A):
            return PLAYER_A
        if self.check_winner(PLAYER_B):
            return PLAYER_B
        return None

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def display(self):
        """
        Print the board state as a formatted matrix.

        Output format matches the assignment specification:
        rows of space-separated values (0, 1, or 2).
        """
        for r in range(self.size):
            print(" ".join(str(self.grid[r][c]) for c in range(self.size)))

    def get_board_string(self):
        """
        Return the board state as a formatted string for file output.

        Returns
        -------
        str
            Multi-line string of the board matrix.
        """
        lines = []
        for r in range(self.size):
            lines.append(" ".join(str(self.grid[r][c]) for c in range(self.size)))
        return "\n".join(lines)

    def __str__(self):
        """String representation of the board."""
        return self.get_board_string()

    def __repr__(self):
        return f"HexBoard(size={self.size}, moves={self.move_count})"



# =============================================================================
# MCTS — Monte Carlo Tree Search with UCT
# =============================================================================

class MCTSNode:
    """A node in the MCTS search tree."""

    def __init__(self, board, player_to_move, parent=None, move=None):
        self.board = board
        self.player_to_move = player_to_move
        self.parent = parent
        self.move = move
        self.children = []
        self.untried_moves = board.get_legal_moves()
        self.visits = 0
        self.wins = 0.0

    @property
    def win_rate(self):
        return self.wins / self.visits if self.visits else 0.0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0


class MCTSAgent:
    """
    Time-limited Monte Carlo Tree Search agent using UCT.

    The four required MCTS stages are:
        1. Selection
        2. Expansion
        3. Simulation / Rollout
        4. Backpropagation
    """

    def __init__(self, player=PLAYER_A, exploration_constant=math.sqrt(2.0), seed=None):
        self.player = player
        self.opponent = PLAYER_B if player == PLAYER_A else PLAYER_A
        self.C = exploration_constant
        self.rng = random.Random(seed)

        self.nodes_expanded = 0
        self.iterations = 0
        self.max_depth = 0

    def _uct(self, child, parent_visits, parent_player):
        """UCT score using the root AI reward perspective."""
        if child.visits == 0:
            return float("inf")

        root_win_rate = child.wins / child.visits
        exploitation = root_win_rate if parent_player == self.player else 1.0 - root_win_rate
        exploration = self.C * math.sqrt(
            math.log(max(1, parent_visits)) / child.visits
        )
        return exploitation + exploration

    def _select_child(self, node):
        return max(
            node.children,
            key=lambda child: self._uct(child, node.visits, node.player_to_move)
        )

    def _expand(self, node):
        """Expand one previously unexplored legal move."""
        index = self.rng.randrange(len(node.untried_moves))
        move = node.untried_moves.pop(index)

        next_board = node.board.clone()
        next_board.make_move(move[0], move[1], node.player_to_move)

        next_player = (
            PLAYER_B if node.player_to_move == PLAYER_A else PLAYER_A
        )

        child = MCTSNode(
            next_board,
            next_player,
            parent=node,
            move=move
        )
        node.children.append(child)
        self.nodes_expanded += 1
        return child

    def _rollout_policy(self, board, player, legal_moves):
        """
        Lightweight rollout policy.

        First look for an immediate winning move. Otherwise choose a
        random legal move. This is still a Monte Carlo rollout policy,
        but is stronger than completely uniform random play.
        """
        moves = legal_moves[:]
        self.rng.shuffle(moves)

        for move in moves:
            test = board.clone()
            test.make_move(move[0], move[1], player)
            if test.check_winner(player):
                return move

        return self.rng.choice(moves)

    def _simulate(self, board, player_to_move, deadline=None):
        """Run one rollout until a terminal state is reached."""
        current_player = player_to_move
        depth = 0

        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                return None, depth
            winner = board.get_winner()
            if winner is not None:
                return winner, depth

            legal_moves = board.get_legal_moves()
            if not legal_moves:
                # Hex has no theoretical draws, but keep a defensive case.
                return None, depth

            move = self._rollout_policy(
                board, current_player, legal_moves
            )
            board.make_move(move[0], move[1], current_player)

            depth += 1
            current_player = (
                PLAYER_B if current_player == PLAYER_A else PLAYER_A
            )

    def _backpropagate(self, node, winner):
        """Update visits and wins from the rollout node to the root."""
        while node is not None:
            node.visits += 1

            if winner == self.player:
                node.wins += 1.0
            elif winner is None:
                node.wins += 0.5

            node = node.parent

    def search(self, root_board, time_limit_ms):
        """
        Search for the best move until the supplied time limit.

        Returns:
            (move, statistics)
        """
        self.nodes_expanded = 0
        self.iterations = 0
        self.max_depth = 0

        legal_moves = root_board.get_legal_moves()

        if not legal_moves:
            return None, {
                "iterations": 0,
                "nodes_expanded": 0,
                "max_depth": 0,
                "root_visits": 0,
                "root_win_rate": 0.0,
                "selected_visits": 0,
                "selected_win_rate": 0.0,
            }

        # Tactical shortcut: if AI can win immediately, take that move.
        for move in legal_moves:
            test = root_board.clone()
            test.make_move(move[0], move[1], self.player)

            if test.check_winner(self.player):
                return move, {
                    "iterations": 1,
                    "nodes_expanded": 1,
                    "max_depth": 1,
                    "root_visits": 1,
                    "root_win_rate": 1.0,
                    "selected_visits": 1,
                    "selected_win_rate": 1.0,
                }

        root = MCTSNode(root_board.clone(), self.player)

        start = time.perf_counter()
        safety_margin_ms = max(5, min(25, int(time_limit_ms * 0.01)))
        usable_ms = max(1, time_limit_ms - safety_margin_ms)
        deadline = start + usable_ms / 1000.0

        while time.perf_counter() < deadline:
            node = root
            depth = 0

            # 1. Selection
            while node.board.get_winner() is None and node.is_fully_expanded() and node.children:
                node = self._select_child(node)
                depth += 1

            # 2. Expansion
            if node.board.get_winner() is None and node.untried_moves:
                node = self._expand(node)
                depth += 1

            # 3. Simulation
            winner, rollout_depth = self._simulate(
                node.board.clone(),
                node.player_to_move, deadline
            )
            depth += rollout_depth
            self.max_depth = max(self.max_depth, depth)

            # 4. Backpropagation
            self._backpropagate(node, winner)
            self.iterations += 1

        if not root.children:
            move = self.rng.choice(legal_moves)
            return move, {
                "iterations": self.iterations,
                "nodes_expanded": self.nodes_expanded,
                "max_depth": self.max_depth,
                "root_visits": root.visits,
                "root_win_rate": root.win_rate,
                "selected_visits": 0,
                "selected_win_rate": 0.0,
            }

        # Robust final choice: most visited child.
        best_child = max(root.children, key=lambda child: child.visits)

        return best_child.move, {
            "iterations": self.iterations,
            "nodes_expanded": self.nodes_expanded,
            "max_depth": self.max_depth,
            "root_visits": root.visits,
            "root_win_rate": root.win_rate,
            "selected_visits": best_child.visits,
            "selected_win_rate": best_child.win_rate,
        }


# =============================================================================
# Input File Parser
# =============================================================================

def read_input_file(filepath="inputPS04.txt"):
    """
    Read the game configuration from the input file.

    File format (per assignment spec):
        Line 1  : Board size N (integer, 7 <= N <= 11)
        Line 2  : Time limit in milliseconds (integer)
        Lines 3+: N rows of space-separated cell values (0, 1, or 2)

    Parameters
    ----------
    filepath : str
        Path to the input file. Defaults to 'inputPS04.txt'.

    Returns
    -------
    tuple (HexBoard, int)
        - A HexBoard initialized with the state from the file.
        - The time limit in milliseconds.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the file content is malformed.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tokens = f.read().split()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Input file '{filepath}' not found. "
            f"Please ensure the file exists in the current directory."
        )

    if len(tokens) < 2:
        raise ValueError(
            f"Input file must have at least 2 lines "
            f"(board size and time limit). Got {len(tokens)} tokens."
        )

    # Line 1: Board size
    try:
        board_size = int(tokens[0])
    except ValueError:
        raise ValueError(
            f"First token must be an integer board size. Got: '{tokens[0]}'"
        )

    # Line 2: Time limit in milliseconds
    try:
        time_limit_ms = int(tokens[1])
    except ValueError:
        raise ValueError(
            f"Second token must be an integer time limit. Got: '{tokens[1]}'"
        )

    if time_limit_ms <= 0:
        raise ValueError(
            f"Time limit must be a positive integer. Got: {time_limit_ms}"
        )

    # Create board
    board = HexBoard(board_size)

    # Lines 3 to N+2: Board grid
    cell_tokens = tokens[2:]
    expected_cells = board_size * board_size
    if len(cell_tokens) != expected_cells:
        raise ValueError(
            f"Expected {board_size} rows for the board grid, "
            f"but found {len(cell_tokens)} board values."
        )

    grid = []
    try:
        cell_values = list(map(int, cell_tokens))
    except ValueError as exc:
        raise ValueError("Board values must all be integers 0, 1, or 2.") from exc
    grid = [cell_values[i * board_size:(i + 1) * board_size] for i in range(board_size)]

    board.load_state(grid)
    return board, time_limit_ms


# =============================================================================
# Human Player Input Handling
# =============================================================================

def get_human_move(board, player_label="B"):
    """
    Prompt the human player for a move, validate it, and return the result.

    Displays the current board, asks for input in row,col format,
    validates the move, and re-prompts on invalid input until a
    valid move is entered. Handles edge cases: non-numeric input,
    out-of-bounds, occupied cells, and EOF / empty input.

    Parameters
    ----------
    board : HexBoard
        The current board state.
    player_label : str
        Display label for the player (e.g., "B").

    Returns
    -------
    tuple (int, int)
        The validated (row, col) move.
    """

    while True:
        try:
            raw = input(f"\nPlayer {player_label}, Enter your move: ")
            raw = raw.strip()

            if not raw:
                print("No input entered. Please enter a move in row,col format (e.g., 3,4)")
                continue

            # Parse "row,col" format
            parts = raw.split(",")
            if len(parts) != 2:
                print("Invalid format. Please enter as row,col (e.g., 3,4)")
                continue

            try:
                row = int(parts[0].strip())
                col = int(parts[1].strip())
            except ValueError:
                print("Invalid input: please enter integers in row,col format.")
                continue

            # Validate the move
            valid, msg = board.validate_move(row, col)
            if not valid:
                print(f"Invalid move: {msg}")
                continue

            print(f"Entered Move: ({row},{col})")
            return row, col

        except EOFError:
            print("\nNo input received. Exiting game gracefully.")
            return None


def format_human_turn(turn_num, player_label, move, board, game_status,
                      move_validation="VALID", execution_time_ms=None):
    """
    Format and return the output string for a human player's turn.

    Parameters
    ----------
    turn_num : int
        Current turn number (1-indexed).
    player_label : str
        "A" or "B".
    move : tuple
        (row, col) of the move played.
    board : HexBoard
        Board state after the move.
    game_status : str
        "CONTINUE", "PLAYER_A_WINS", or "PLAYER_B_WINS".
    move_validation : str
        "VALID" or an error description.
    execution_time_ms : int or None
        Time taken for the move in milliseconds.

    Returns
    -------
    str
        Formatted turn output string.
    """
    lines = []
    lines.append(f"{'='*20}Turn {turn_num}{'='*32}")
    lines.append(f"Player: {player_label} (Human)")
    lines.append(f"Move Entered: ({move[0]},{move[1]})")
    lines.append(f"Move Validation: {move_validation}")
    exec_str = f"{execution_time_ms} ms" if execution_time_ms is not None else ""
    lines.append(f"Execution Time: {exec_str}")
    lines.append(f"Game Status: {game_status}")
    lines.append(f"Current Board")
    lines.append(board.get_board_string())
    return "\n".join(lines)

# =============================================================================
# Output Writer
# =============================================================================

class OutputWriter:
    """
    Collects all game output and writes it to outputPS04.txt.

    Usage:
        writer = OutputWriter("outputPS04.txt")
        writer.log("some output")        # prints to console + stores
        writer.log_silent("hidden data") # stores only, no console print
        writer.write_to_file()           # writes everything to file
    """

    def __init__(self, filepath="outputPS04.txt"):
        """
        Initialize the output writer.

        Parameters
        ----------
        filepath : str
            Path to the output file.
        """
        self.filepath = filepath
        self._buffer = []

    def log(self, text):
        """Print text to console and store it for file output."""
        print(text)
        self._buffer.append(text)

    def log_silent(self, text):
        """Store text for file output without printing to console."""
        self._buffer.append(text)

    def write_to_file(self):
        """
        Write all collected output to the output file.

        Raises
        ------
        IOError
            If the file cannot be written.
        """
        try:
            with open(self.filepath, "w") as f:
                f.write("\n".join(self._buffer))
            print(f"\nOutput written to {self.filepath}")
        except IOError as e:
            print(f"Error writing output file: {e}")



# =============================================================================
# Game Loop
# =============================================================================

def get_game_status(board):
    """Return the current game status."""
    winner = board.get_winner()

    if winner == PLAYER_A:
        return "PLAYER_A_WINS"
    if winner == PLAYER_B:
        return "PLAYER_B_WINS"

    return "CONTINUE"


def run_game(board, time_limit_ms, output_path="outputPS04.txt", seed=None):
    """
    Run the complete Hex game.

    Player A (AI) always moves first.
    Player B (human) enters one move on every alternate turn.

    Parameters
    ----------
    board : HexBoard
        Current game board.
    time_limit_ms : int
        MCTS computation budget for each AI move.
    output_path : str
        Output file path.
    seed : int or None
        Optional random seed for reproducible MCTS rollouts.
    """
    writer = OutputWriter(output_path)
    agent = MCTSAgent(player=PLAYER_A, seed=seed)

    turn = 0
    total_ai_moves = 0
    total_human_moves = 0

    ai_times = []
    search_depths = []
    expanded_nodes = []

    writer.log("Hex Game AI - Monte Carlo Tree Search (MCTS) + UCT")
    writer.log(f"Board Size : {board.size}x{board.size}")
    writer.log(f"Time Limit : {time_limit_ms} ms per AI move")
    writer.log("Player A (1) : AI - connects Top row to Bottom row")
    writer.log("Player B (2) : Human - connects Left col to Right col")
    writer.log("Initial Board")
    writer.log(board.get_board_string())

    # Handle an input board that is already terminal.
    winner = board.get_winner()
    if winner is not None:
        writer.log("\n" + "=" * 25 + " GAME OVER " + "=" * 25)
        writer.log(
            f"Winner : {'Player A' if winner == PLAYER_A else 'Player B'}"
        )
        writer.log(
            f"Game Result : "
            f"{'PLAYER_A_WINS' if winner == PLAYER_A else 'PLAYER_B_WINS'}"
        )
        writer.write_to_file()
        return

    current_player = PLAYER_A

    while True:
        turn += 1

        # ================================================================
        # AI TURN
        # ================================================================
        if current_player == PLAYER_A:
            start = time.perf_counter()

            move, stats = agent.search(
                board,
                time_limit_ms
            )

            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if move is None:
                writer.log("No legal AI move available. Game ended gracefully.")
                break

            board.make_move(move[0], move[1], PLAYER_A)

            total_ai_moves += 1
            ai_times.append(elapsed_ms)
            search_depths.append(stats["max_depth"])
            expanded_nodes.append(stats["nodes_expanded"])

            status = get_game_status(board)

            writer.log("\n" + "=" * 20 + f"Turn {turn}" + "=" * 32)
            writer.log("Player : A (AI)")
            writer.log(f"Move Selected : ({move[0]},{move[1]})")
            writer.log("Search Algorithm : MCTS + UCT")
            writer.log(f"MCTS Iterations : {stats['iterations']}")
            writer.log(f"Nodes Expanded : {stats['nodes_expanded']}")
            writer.log(f"Search Depth Reached : {stats['max_depth']}")
            writer.log(f"Root Win Rate : {stats['root_win_rate']:.4f}")
            writer.log(f"Selected Move Visits : {stats['selected_visits']}")
            writer.log(
                f"Selected Move Win Rate : "
                f"{stats['selected_win_rate']:.4f}"
            )
            writer.log(f"AI Execution Time : {elapsed_ms:.2f} ms")
            writer.log(f"Game Status : {status}")
            writer.log("Current Board")
            writer.log(board.get_board_string())

            if status != "CONTINUE":
                winner = board.get_winner()
                writer.log(
                    "\n" + "=" * 25 + " GAME OVER " + "=" * 25
                )
                writer.log(
                    f"Winner : "
                    f"{'Player A' if winner == PLAYER_A else 'Player B'}"
                )
                break

            current_player = PLAYER_B

        # ================================================================
        # HUMAN TURN
        # ================================================================
        else:
            start = time.perf_counter()

            move = get_human_move(board, "B")

            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if move is None:
                writer.log(
                    "No valid human move received. "
                    "Game ended gracefully."
                )
                break

            board.make_move(move[0], move[1], PLAYER_B)

            total_human_moves += 1

            status = get_game_status(board)

            writer.log("\n" + "=" * 20 + f"Turn {turn}" + "=" * 32)
            writer.log("Player : B (Human)")
            writer.log(f"Move Entered : ({move[0]},{move[1]})")
            writer.log("Move Validation : VALID")
            writer.log(f"Human Input Time : {elapsed_ms:.2f} ms")
            writer.log(f"Game Status : {status}")
            writer.log("Current Board")
            writer.log(board.get_board_string())

            if status != "CONTINUE":
                winner = board.get_winner()
                writer.log(
                    "\n" + "=" * 25 + " GAME OVER " + "=" * 25
                )
                writer.log(
                    f"Winner : "
                    f"{'Player A' if winner == PLAYER_A else 'Player B'}"
                )
                break

            current_player = PLAYER_A

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    winner = board.get_winner()

    writer.log(
        "\n" + "=" * 25 + " FINAL SUMMARY " + "=" * 25
    )

    if winner == PLAYER_A:
        winner_name = "Player A"
        result = "PLAYER_A_WINS"
    elif winner == PLAYER_B:
        winner_name = "Player B"
        result = "PLAYER_B_WINS"
    else:
        winner_name = "NONE"
        result = "GAME_NOT_COMPLETED"

    writer.log(f"Winner : {winner_name}")
    writer.log(f"Total Turns : {turn}")
    writer.log(f"Total AI Moves : {total_ai_moves}")
    writer.log(f"Total Human Moves : {total_human_moves}")

    if search_depths:
        writer.log(
            f"Average Search Depth : "
            f"{sum(search_depths) / len(search_depths):.2f}"
        )
        writer.log(f"Maximum Search Depth : {max(search_depths)}")
    else:
        writer.log("Average Search Depth : 0.00")
        writer.log("Maximum Search Depth : 0")

    if expanded_nodes:
        writer.log(
            f"Average Nodes Expanded : "
            f"{sum(expanded_nodes) / len(expanded_nodes):.2f}"
        )
        writer.log(f"Maximum Nodes Expanded : {max(expanded_nodes)}")
    else:
        writer.log("Average Nodes Expanded : 0.00")
        writer.log("Maximum Nodes Expanded : 0")

    if ai_times:
        writer.log(
            f"Average AI Move Time : "
            f"{sum(ai_times) / len(ai_times):.2f} ms"
        )
    else:
        writer.log("Average AI Move Time : 0.00 ms")

    writer.log(f"Game Result : {result}")

    writer.write_to_file()


# =============================================================================
# Main Entry Point
# =============================================================================

def run_self_tests():
    board = HexBoard(7)
    neighbors = set(board.get_neighbors(3, 3))
    assert (2, 2) not in neighbors and (4, 4) not in neighbors
    assert (2, 4) in neighbors and (4, 2) in neighbors

    for row in range(7):
        board.grid[row][2] = PLAYER_A
    board.move_count = 7
    assert board.check_winner(PLAYER_A)

    board = HexBoard(7)
    for col in range(7):
        board.grid[4][col] = PLAYER_B
    board.move_count = 7
    assert board.check_winner(PLAYER_B)

    board = HexBoard(7)
    board.make_move(0, 0, PLAYER_A)
    assert not board.is_valid_move(0, 0)
    assert not board.is_valid_move(-1, 0)

    board = HexBoard(7)
    for row in range(6):
        board.grid[row][3] = PLAYER_A
    board.move_count = 6
    agent = MCTSAgent(seed=7)
    move, _ = agent.search(board, 100)
    winning = []
    for candidate in board.get_legal_moves():
        trial = board.clone()
        trial.make_move(candidate[0], candidate[1], PLAYER_A)
        if trial.check_winner(PLAYER_A):
            winning.append(candidate)
    assert move in winning
    print("SELF_TESTS_PASSED: 5 test groups")

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--test":
        run_self_tests()
        sys.exit(0)

    input_path = "inputPS04.txt"
    output_path = "outputPS04.txt"

    # Optional command-line paths:
    # python hexGamePS04.py inputPS04.txt outputPS04.txt
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]

    try:
        board, time_limit = read_input_file(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"\nHex Game - Board Size: {board.size}x{board.size}")
    print(f"Time Limit: {time_limit} ms")
    print("Player A (1) = AI - connects Top row to Bottom row")
    print("Player B (2) = Human - connects Left column to Right column")
    print("\nInitial Board:")
    board.display()

    run_game(
        board,
        time_limit,
        output_path=output_path
    )
