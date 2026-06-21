import re
import os
import shutil
import threading
import atexit
import logging
from pathlib import Path
import chess
import chess.engine

logger = logging.getLogger("chess_ai.expert")

# Initialize the chess-detect analyzer with arrows enabled
try:
    from chess_detect import ChessDetector
    detector = ChessDetector(lang="en", tactics=True, strategy=True, arrows=True) # Changed arrows to True
except Exception as e:
    logger.error("Failed to load chess_detect: %s", e)
    detector = None

# Piece values for material checks
PVAL = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100
}


# ── Thread-Safe Engine Manager ────────────────────────────────────────
class ExpertEngineManager:
    """Manages an auxiliary Stockfish instance to compute threats and refutations."""
    def __init__(self):
        self._e = None
        self.lock = threading.Lock()

    def _get_path(self) -> str:
        try:
            for p in Path(".").iterdir():
                if "stockfish" in p.name.lower() and p.is_file() and (os.name != 'nt' or p.suffix.lower() == '.exe'):
                    return str(p)
        except Exception as e:
            logger.error("Error scanning local files: %s", e)
        paths = ["stockfish", "./stockfish", "/usr/games/stockfish", "/usr/bin/stockfish", "/opt/homebrew/bin/stockfish"]
        return next((p for p in paths if shutil.which(p) or Path(p).exists()), "stockfish")

    def get(self) -> chess.engine.SimpleEngine | None:
        with self.lock:
            if not self._e:
                p = self._get_path()
                try:
                    self._e = chess.engine.SimpleEngine.popen_uci(p)
                    self._e.configure({"Threads": 1, "Hash": 32})
                except Exception as e:
                    logger.error("Engine start failed: %s", e)
            return self._e

    def close(self):
        with self.lock:
            if self._e:
                try:
                    self._e.quit()
                except Exception:
                    pass
                self._e = None


expert_eng = ExpertEngineManager()
atexit.register(expert_eng.close)


# ── Annotation Helper ───────────────────────────────────────────────
def get_annotations(prev_fen: str, move_uci: str) -> dict:
    """Uses chess-detect to analyze the move and return PGN arrows and circles."""
    if not detector or not prev_fen or not move_uci:
        return {"arrows": [], "circles": []}
    try:
        import chess.pgn
        board = chess.Board(prev_fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return {"arrows": [], "circles": []}

        # Setup single-move game sequence for the analyzer
        game = chess.pgn.Game()
        game.setup(board)
        game.add_variation(move)

        pgn_str = str(game)
        annotated_pgn = detector.analyze(pgn_str)

        arrows = []
        circles = []

        # Parse %cal (arrows) -> e.g., [%cal Gc7a8,Gc7e8]
        # Colors: G=green, R=red, B=blue, Y=yellow
        cal_matches = re.findall(r'\[%cal\s+([^\]]+)\]', annotated_pgn)
        for match in cal_matches:
            for item in match.split(','):
                item = item.strip()
                if len(item) >= 5:
                    color_char = item[0]
                    orig = item[1:3]
                    dest = item[3:5]
                    brush = 'green'
                    if color_char == 'R': brush = 'red'
                    elif color_char == 'B': brush = 'blue'
                    elif color_char == 'Y': brush = 'yellow'
                    arrows.append({"orig": orig, "dest": dest, "brush": brush})

        # Parse %csl (highlights) -> e.g., [%csl Ga8,Ge8]
        csl_matches = re.findall(r'\[%csl\s+([^\]]+)\]', annotated_pgn)
        for match in csl_matches:
            for item in match.split(','):
                item = item.strip()
                if len(item) >= 3:
                    color_char = item[0]
                    sq = item[1:3]
                    brush = 'green'
                    if color_char == 'R': brush = 'red'
                    elif color_char == 'B': brush = 'blue'
                    elif color_char == 'Y': brush = 'yellow'
                    circles.append({"orig": sq, "brush": brush})

        return {"arrows": arrows, "circles": circles}
    except Exception as e:
        logger.error("Error running chess-detect annotation parser: %s", e)
        return {"arrows": [], "circles": []}


# ── Dynamic Positional & Evaluation Extractors ──────────────────────────
def get_eval_description(ev, player_color: str) -> str:
    """Programmatically converts evaluations to active player perspective using absolute values to prevent sign contradictions."""
    if ev is None:
        return "maintains a highly balanced position"
    
    # Process Forced Mate Evaluations
    if isinstance(ev, str) and "M" in ev:
        try:
            mate_moves = int(ev.replace("M", "").replace("+", ""))
            if mate_moves < 0:
                return f"concedes a forced mate-in-{abs(mate_moves)}"
            else:
                return f"sets up a forced mate-in-{mate_moves}"
        except ValueError:
            return "leads to a forced mate sequence"
    
    # Process Standard Numeric Evaluations
    try:
        score = float(ev) / 100
        opp_color = "Black" if player_color == "White" else "White"
        
        # Determine absolute advantage level
        if abs(score) <= 0.3:
            return "maintains a highly balanced position"
        
        # Determine who actually holds the advantage based on engine sign (Positive=White, Negative=Black)
        favored_color = "White" if score > 0 else "Black"
        
        advantage_level = "slight"
        if abs(score) > 1.9:
            advantage_level = "decisive"
        elif abs(score) > 0.9:
            advantage_level = "clear"
        
        # Strip signs from absolute value to prevent prompt instructions from getting confused by numeric formatting
        abs_score_str = f"{abs(score):.2f}"
        
        if player_color == favored_color:
            return f"secures a {advantage_level} advantage of {abs_score_str} for {player_color}"
        else:
            return f"concedes a {advantage_level} advantage of {abs_score_str} to {opp_color}"
            
    except (ValueError, TypeError):
        return "maintains a stable position"


def is_tactical_threat(board: chess.Board, move: chess.Move) -> bool:
    """Filters out quiet development moves to ensure only genuine tactical threats are logged."""
    if board.is_capture(move) or board.gives_check(move):
        return True
    
    piece = board.piece_at(move.from_square)
    if piece:
        temp_board = board.copy()
        temp_board.push(move)
        attacks = temp_board.attacks(move.to_square)
        for sq in attacks:
            attacked_piece = temp_board.piece_at(sq)
            if attacked_piece and attacked_piece.color != piece.color:
                if PVAL.get(attacked_piece.piece_type, 0) > PVAL.get(piece.piece_type, 0):
                    return True
    return False


def get_pins(board: chess.Board, color: chess.Color) -> list:
    pins = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == color and p.piece_type != chess.KING:
            if board.is_pinned(color, sq):
                pins.append(f"{chess.piece_name(p.piece_type).capitalize()} on {chess.square_name(sq)}")
    return pins


def get_hanging_pieces(board: chess.Board, color: chess.Color) -> list:
    hanging = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == color and p.piece_type != chess.KING:
            if board.is_attacked_by(not color, sq) and not board.is_attacked_by(color, sq):
                hanging.append(f"{chess.piece_name(p.piece_type).capitalize()} on {chess.square_name(sq)}")
    return hanging


def get_rook_files(board: chess.Board, color: chess.Color) -> tuple:
    open_files, semi_open_files = [], []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == color and p.piece_type == chess.ROOK:
            file_idx = chess.square_file(sq)
            file_char = chess.FILE_NAMES[file_idx]
            file_name = f"{file_char}-file"
            has_own_pawns = False
            has_opp_pawns = False
            for r in range(8):
                p_chk = board.piece_at(chess.square(file_idx, r))
                if p_chk and p_chk.piece_type == chess.PAWN:
                    if p_chk.color == color:
                        has_own_pawns = True
                    else:
                        has_opp_pawns = True
            if not has_own_pawns and not has_opp_pawns:
                if file_name not in open_files:
                    open_files.append(file_name)
            elif not has_own_pawns and has_opp_pawns:
                if file_name not in semi_open_files:
                    semi_open_files.append(file_name)
    return open_files, semi_open_files


def get_move_fork(board: chess.Board, move: chess.Move, color: chess.Color) -> str:
    p = board.piece_at(move.to_square)
    if p and p.color == color:
        attacks = board.attacks(move.to_square)
        targets = []
        for atk_sq in attacks:
            tgt = board.piece_at(atk_sq)
            if tgt and tgt.color != color and tgt.piece_type != chess.KING:
                opp_color_str = "White" if color == chess.BLACK else "Black"
                if PVAL.get(tgt.piece_type, 0) >= PVAL.get(p.piece_type, 0) or not board.is_attacked_by(not color, atk_sq):
                    targets.append(f"{opp_color_str}'s {chess.piece_name(tgt.piece_type)} on {chess.square_name(atk_sq)}")
        if len(targets) >= 2:
            return f"{chess.piece_name(p.piece_type).capitalize()} on {chess.square_name(move.to_square)} attacks {', and '.join(targets)}"
    return ""


# ── Sub-Engine Calculations ──────────────────────────────────────────
def score_val(s: chess.engine.PovScore | None) -> int:
    if s is None:
        return 0
    if s.is_mate():
        m = s.white().mate()
        return 0 if m is None else (20000 - m if m > 0 else -20000 - m)
    v = s.white().score()
    return v if v is not None else 0


def perform_null_move_analysis(board: chess.Board, player_color: str) -> dict | None:
    if board.is_check() or board.is_game_over():
        return None
    
    non_pawn_pieces = [p for p in board.piece_map().values() if p.piece_type not in [chess.PAWN, chess.KING]]
    if not non_pawn_pieces:
        return None

    temp_board = board.copy()
    try:
        temp_board.push(chess.Move.null())
    except Exception:
        return None
    
    engine = expert_eng.get()
    if engine is None:
        return None
    
    try:
        with expert_eng.lock:
            result = engine.analyse(temp_board, chess.engine.Limit(time=0.10, depth=10))
        
        pv = result.get("pv")
        if pv and len(pv) > 0:
            threat_move = pv[0]
            if not is_tactical_threat(temp_board, threat_move):
                return None
                
            threat_san = temp_board.san(threat_move)
            captured_piece = board.piece_at(threat_move.to_square)
            
            threat_desc = f"{player_color} threatens to play {threat_san}"
            if captured_piece:
                victim_color = "White" if captured_piece.color == chess.WHITE else "Black"
                threat_desc += f" to capture {victim_color}'s {chess.piece_name(captured_piece.piece_type)} on {chess.square_name(threat_move.to_square)}"
            
            return {
                "threat_move_uci": threat_move.uci(),
                "threat_description": threat_desc
            }
    except Exception as e:
        logger.error("Null-move analysis failed: %s", e)
    return None


def perform_pre_move_threat_analysis(prev_board: chess.Board, opponent_color: str) -> dict | None:
    if not prev_board or prev_board.is_check() or prev_board.is_game_over():
        return None
    
    temp_board = prev_board.copy()
    try:
        temp_board.push(chess.Move.null())
    except Exception:
        return None
    
    engine = expert_eng.get()
    if engine is None:
        return None
    
    try:
        with expert_eng.lock:
            result = engine.analyse(temp_board, chess.engine.Limit(time=0.10, depth=10))
        
        pv = result.get("pv")
        if pv and len(pv) > 0:
            threat_move = pv[0]
            if not is_tactical_threat(temp_board, threat_move):
                return None
                
            threat_san = temp_board.san(threat_move)
            captured_piece = prev_board.piece_at(threat_move.to_square)
            
            threat_desc = f"{opponent_color} threatened to play {threat_san}"
            if captured_piece:
                victim_color = "White" if captured_piece.color == chess.WHITE else "Black"
                threat_desc += f" to capture {victim_color}'s {chess.piece_name(captured_piece.piece_type)} on {chess.square_name(threat_move.to_square)}"
                
            return {
                "threat_move_uci": threat_move.uci(),
                "threat_description": threat_desc
            }
    except Exception as e:
        logger.error("Pre-move analysis failed: %s", e)
    return None


def verify_threat_resolution(prev_board: chess.Board, player_move: chess.Move, prev_threat: dict) -> str:
    if not prev_threat:
        return ""
    threat_move_uci = prev_threat.get("threat_move_uci")
    if not threat_move_uci:
        return ""
        
    threat_move = chess.Move.from_uci(threat_move_uci)
    if player_move.to_square == threat_move.from_square:
        return f"Captured attacking piece on {chess.square_name(threat_move.from_square)}."
        
    target_sq = threat_move.to_square
    if player_move.from_square == target_sq:
        temp_board = prev_board.copy()
        temp_board.push(player_move)
        if not temp_board.is_attacked_by(temp_board.turn, player_move.to_square):
            return f"Moved threatened piece to safe square {chess.square_name(player_move.to_square)}."
        
    temp_board = prev_board.copy()
    temp_board.push(player_move)
    if threat_move not in temp_board.legal_moves:
        return "Blocked or prevented threat."
        
    if player_move.from_square != target_sq and temp_board.is_attacked_by(prev_board.turn, target_sq):
        return f"Protected threatened piece on {chess.square_name(target_sq)}."
        
    return "Left threat active."


def perform_refutation_analysis(actual_board: chess.Board) -> dict | None:
    if actual_board.is_game_over():
        return None
        
    engine = expert_eng.get()
    if engine is None:
        return None
        
    try:
        with expert_eng.lock:
            result = engine.analyse(actual_board, chess.engine.Limit(time=0.12, depth=11))
        
        pv = result.get("pv")
        if pv and len(pv) > 0:
            ref_move = pv[0]
            ref_san = actual_board.san(ref_move)
            
            pv_board = actual_board.copy()
            pv_san = []
            
            initial_w_val = sum(PVAL.get(p.piece_type, 0) for p in pv_board.piece_map().values() if p.color == chess.WHITE)
            initial_b_val = sum(PVAL.get(p.piece_type, 0) for p in pv_board.piece_map().values() if p.color == chess.BLACK)
            
            for m in pv[:4]:
                try:
                    san_m = pv_board.san(m)
                    pv_san.append(san_m)
                    pv_board.push(m)
                except Exception:
                    break
                    
            final_w_val = sum(PVAL.get(p.piece_type, 0) for p in pv_board.piece_map().values() if p.color == chess.WHITE)
            final_b_val = sum(PVAL.get(p.piece_type, 0) for p in pv_board.piece_map().values() if p.color == chess.BLACK)
            
            refuting_color = actual_board.turn
            ref_gain = (final_b_val - initial_b_val) - (final_w_val - initial_w_val) if refuting_color == chess.BLACK else (final_w_val - initial_w_val) - (final_b_val - initial_b_val)
            
            # Keep descriptions highly abstract to prevent the 0.8B model from hallucinating move sequences
            desc = f"opponent can play {ref_san}"
            if ref_san.endswith("#") or pv_board.is_checkmate():
                desc += " to deliver immediate checkmate"
            elif ref_gain > 0:
                desc += f" to gain a material advantage of {ref_gain} point{'s' if ref_gain > 1 else ''}"
            else:
                desc += " to gain a positional advantage"
                
            pv_formatted = [f"{i+1}. {m}" for i, m in enumerate(pv_san)]
            ref_pv_str = ", ".join(pv_formatted)
                
            return {
                "refutation_description": desc,
                "refutation_pv": ref_pv_str
            }
    except Exception as e:
        logger.error("Refutation computation failed: %s", e)
    return None


# ── Failsafe Post-Processing Filter ───────────────────────────────────
def clean_meta_text(text: str) -> str:
    """Programmatically cleans up instruction leakages."""
    text = re.sub(r"\bwithout repeating\b.*?\.(?=\s|$)", ".", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwithout repeating\b.*?(?=\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bas requested\b.*?\.(?=\s|$)", ".", text, flags=re.IGNORECASE)
    text = re.sub(r"\bper the provided\b.*?\.(?=\s|$)", ".", text, flags=re.IGNORECASE)
    text = re.sub(r"\bper the instructions\b.*?\.(?=\s|$)", ".", text, flags=re.IGNORECASE)
    text = re.sub(r"\baccording to the rules\b.*?\.(?=\s|$)", ".", text, flags=re.IGNORECASE)
    text = re.sub(r"\busing the provided\b.*?\b(feature|line|text|box)\b", "", text, flags=re.IGNORECASE)
    
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip()


# ── Context Pipeline (Optimized Context Pruning) ──────────────────────────
def prepare_coach_context(data: dict) -> dict:
    prev_fen = data.get("prev_fen")
    current_fen = data.get("fen")
    move_san = data.get("move_san") or "Unknown"
    move_uci = data.get("move_uci")
    classification = data.get("classification")
    best_move_san = data.get("best_move_san")
    ev = data.get("eval")
    opening_name = data.get("opening_name")

    cls_label = classification or "unknown"
    prev_board = chess.Board(prev_fen) if prev_fen else None
    board = chess.Board(current_fen) if current_fen else None

    # Initialize default features to prevent UnboundLocalError on checkmate
    fork_made = ""
    new_open_rooks = []
    new_semi_rooks = []

    # Determine Active Player and Game Context
    player_color = "Unknown"
    opponent_color = "Unknown"
    if board:
        player_color = "Black" if board.turn == chess.WHITE else "White"
        opponent_color = "White" if board.turn == chess.WHITE else "Black"
    elif prev_board:
        player_color = "White" if prev_board.turn == chess.WHITE else "Black"
        opponent_color = "Black" if prev_board.turn == chess.WHITE else "White"

    game_phase = "Middlegame"
    if board:
        w_p = [p for p in board.piece_map().values() if p.color == chess.WHITE and p.piece_type not in [chess.PAWN, chess.KING]]
        b_p = [p for p in board.piece_map().values() if p.color == chess.BLACK and p.piece_type not in [chess.PAWN, chess.KING]]
        if len(w_p) + len(b_p) <= 4 or board.fullmove_number >= 40:
            game_phase = "Endgame"
        elif board.fullmove_number <= 12:
            game_phase = "Opening"

    is_checkmate = board.is_checkmate() if board else False

    # 1. Thread-Safe Engine Scoring
    prev_eval_score = None
    curr_eval_score = None
    if not is_checkmate:
        engine = expert_eng.get()
        if engine is not None:
            if prev_board:
                try:
                    with expert_eng.lock:
                        prev_res = engine.analyse(prev_board, chess.engine.Limit(time=0.10, depth=10))
                    prev_eval_score = score_val(prev_res.get("score"))
                except Exception:
                    pass
            if board:
                try:
                    with expert_eng.lock:
                        curr_res = engine.analyse(board, chess.engine.Limit(time=0.10, depth=10))
                    curr_eval_score = score_val(curr_res.get("score"))
                except Exception:
                    pass

    eval_delta = 0
    if prev_eval_score is not None and curr_eval_score is not None:
        if player_color == "White":
            eval_delta = curr_eval_score - prev_eval_score
        else:
            eval_delta = prev_eval_score - curr_eval_score

    # Override classification if score drops significantly
    if cls_label == "book" and eval_delta < -80:
        cls_label = "inaccuracy" if eval_delta >= -150 else "mistake"

    is_bad_move = cls_label in ["inaccuracy", "mistake", "blunder"]

    eval_str = "0.0"
    if ev is not None:
        try:
            if isinstance(ev, str) and "M" in ev:
                eval_str = f"Forced mate ({ev})"
            else:
                score = float(ev) / 100
                eval_str = f"{score:+.2f}"
        except (ValueError, TypeError):
            eval_str = str(ev)

    eval_desc = get_eval_description(ev, player_color)

    # 2. Dynamic King Safety Shield Tracker
    is_king_shield_weakened = False
    if prev_board and move_uci and not is_checkmate:
        try:
            player_move = chess.Move.from_uci(move_uci)
            piece = prev_board.piece_at(player_move.from_square)
            if piece and piece.piece_type == chess.PAWN:
                from_file = chess.square_file(player_move.from_square)
                from_rank = chess.square_rank(player_move.from_square)
                player_col = chess.WHITE if player_color == "White" else chess.BLACK
                king_sq = prev_board.king(player_col)
                if king_sq:
                    kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
                    target_rank = 1 if player_col == chess.WHITE else 6
                    if from_rank == target_rank and from_file in [5, 6, 7] and abs(from_file - kf) <= 2:
                        is_king_shield_weakened = True
        except Exception:
            pass

    # 3. Dynamic Move Purpose Engine
    move_purpose = ""
    if prev_board and move_uci and not is_checkmate:
        try:
            player_move = chess.Move.from_uci(move_uci)
            piece = prev_board.piece_at(player_move.from_square)
            if piece:
                p_name = chess.piece_name(piece.piece_type).capitalize()
                to_name = chess.square_name(player_move.to_square)
                
                if is_king_shield_weakened:
                    move_purpose = f"advance the {p_name} to {to_name} but weaken king safety"
                elif prev_board.is_castling(player_move):
                    move_purpose = "secure king safety and activate the rook via castling"
                elif piece.piece_type == chess.PAWN:
                    if game_phase == "Opening":
                        opening_suffix = f" in the {opening_name}" if opening_name else ""
                        move_purpose = f"advance the pawn to {to_name} to claim space and contest the center{opening_suffix}"
                    elif game_phase == "Endgame":
                        move_purpose = f"advance the pawn to {to_name} to threaten promotion"
                    else:
                        move_purpose = f"advance the pawn to {to_name} to claim space"
                else:
                    if game_phase == "Opening":
                        move_purpose = f"develop the {p_name} to {to_name} to control key squares"
                    elif game_phase == "Endgame":
                        move_purpose = f"activate the {p_name} on {to_name} to support centralization"
                    else:
                        move_purpose = f"position the {p_name} on {to_name} to improve coordinate activity"
        except Exception:
            pass

    if not move_purpose:
        if game_phase == "Opening":
            move_purpose = "fight for central control in the opening"
        elif game_phase == "Endgame":
            move_purpose = "activate pieces in the endgame"
        else:
            move_purpose = "adjust piece activity in the middlegame"

    # 4. Extract Tactical Calculations
    prev_threat = None
    curr_threat = None
    refutation = None
    if not is_checkmate:
        prev_threat = perform_pre_move_threat_analysis(prev_board, opponent_color) if prev_board else None
        curr_threat = perform_null_move_analysis(board, player_color) if board else None
        refutation = perform_refutation_analysis(board) if (board and is_bad_move) else None

    threat_resolution = ""
    if prev_board and move_uci and prev_threat and not is_checkmate:
        try:
            player_move = chess.Move.from_uci(move_uci)
            threat_resolution = verify_threat_resolution(prev_board, player_move, prev_threat)
        except Exception:
            pass

    # 5. Compact Feature Mapping
    feature_lines = [
        f"- Player: {player_color}",
        f"- Phase: {game_phase}",
        f"- Class: {cls_label}",
        f"- Eval: {eval_str}"
    ]

    if is_checkmate:
        feature_lines = [
            f"- Player: {player_color}",
            f"- Phase: {game_phase}",
            f"- Class: checkmate",
            f"- Action: Delivered checkmate via {move_san}"
        ]
    else:
        # Include Refutation details ONLY for bad moves
        if is_bad_move and refutation:
            feature_lines.append(f"- Opponent Refutation: {refutation['refutation_description']}")
            feature_lines.append(f"- Refutation PV: {refutation['refutation_pv']}")
        
        # Include Fork / Rook activations ONLY if they exist and the move is not bad
        fork_made = ""
        new_open_rooks = []
        new_semi_rooks = []
        
        if board and prev_board and move_uci:
            try:
                player_col = chess.WHITE if player_color == "White" else chess.BLACK
                player_move = chess.Move.from_uci(move_uci)
                
                if not is_bad_move:
                    if prev_board.is_castling(player_move):
                        feature_lines.append("- Action: Castled safely")
                    if prev_board.is_capture(player_move):
                        tgt = prev_board.piece_at(player_move.to_square)
                        if tgt:
                            to_name = chess.square_name(player_move.to_square)
                            feature_lines.append(f"- Action: Captured opponent's {chess.piece_name(tgt.piece_type)} on {to_name}")
                    
                    fork_made = get_move_fork(board, player_move, player_col)
                    if fork_made:
                        feature_lines.append(f"- Fork: {fork_made}")
                        
                    rooks_open_prev, rooks_semi_prev = get_rook_files(prev_board, player_col)
                    rooks_open_curr, rooks_semi_curr = get_rook_files(board, player_col)
                    new_open_rooks = [r for r in rooks_open_curr if r not in rooks_open_prev]
                    new_semi_rooks = [r for r in rooks_semi_curr if r not in rooks_semi_prev]
                    if new_open_rooks:
                        feature_lines.append(f"- Rook: Open file ({', '.join(new_open_rooks)})")
                    elif new_semi_rooks:
                        feature_lines.append(f"- Rook: Semi-open file ({', '.join(new_semi_rooks)})")

            except Exception as e:
                logger.error("Error calculating positional deltas: %s", e)

        # Explicitly tag the best alternative's owner to prevent mismatching
        is_best_move = (best_move_san == move_san) or (best_move_san is None)
        if not is_best_move and best_move_san:
            feature_lines.append(f"- Best Alternative for {player_color}: {best_move_san}")

        if opening_name:
            feature_lines.append(f"- Opening: {opening_name}")

    eval_context = ""
    if ev is not None and not is_bad_move and cls_label == "book" and not is_checkmate:
        eval_context = f"(Matches standard theory despite engine eval of {eval_str}.)"

    is_forced_mate = False
    if ev is not None and not is_checkmate:
        try:
            if isinstance(ev, str) and "M" in ev:
                is_forced_mate = True
        except Exception:
            pass

    # Compile Benefit Detail
    benefits = []
    if fork_made and not is_checkmate:
        benefits.append(f"creating a fork ({fork_made})")
    if new_open_rooks and not is_checkmate:
        benefits.append(f"activating a rook on open file ({', '.join(new_open_rooks)})")
    elif new_semi_rooks and not is_checkmate:
        benefits.append(f"activating a rook on semi-open file ({', '.join(new_semi_rooks)})")

    benefit_detail = " and ".join(benefits) if benefits else ""

    return {
        "move_san": move_san,
        "best_move_san": best_move_san,
        "cls_label": cls_label,
        "is_bad_move": is_bad_move,
        "eval_str": eval_str,
        "eval_context": eval_context,
        "features_block": "\n".join(feature_lines),
        "player_color": player_color,
        "opening_name": opening_name,
        "game_phase": game_phase,
        "is_checkmate": is_checkmate,
        "is_forced_mate": is_forced_mate,
        "move_purpose": move_purpose,
        "benefit_detail": benefit_detail,
        "eval_desc": eval_desc,
        "refutation_pv": refutation["refutation_pv"] if refutation else ""
    }
