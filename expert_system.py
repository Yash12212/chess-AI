import re
import os
import shutil
import threading
import atexit
import logging
from pathlib import Path

import chess
import chess.engine
import chess.pgn

logger = logging.getLogger("chess_ai.expert")

# Initialize the chess-detect analyzer with arrows enabled
try:
    from chess_detect import ChessDetector
    detector = ChessDetector(lang="en", tactics=True, strategy=True, arrows=True)
except Exception as e:
    logger.error("Failed to load chess_detect: %s", e)
    detector = None

# Piece values for material checks
PVAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}

# PGN color code -> brush name, and the move-classification labels that flag errors
_BRUSH = {"G": "green", "R": "red", "B": "blue", "Y": "yellow"}
_BAD_CLASSES = ("inaccuracy", "mistake", "blunder")


# ── Thread-Safe Engine Manager ────────────────────────────────────────
class ExpertEngineManager:
    """Manages an auxiliary Stockfish instance to compute threats and refutations."""

    def __init__(self):
        self._e = None
        self.lock = threading.Lock()

    def _get_path(self) -> str:
        # Prefer a stockfish binary sitting in the current directory
        try:
            for p in Path(".").iterdir():
                if "stockfish" in p.name.lower() and p.is_file() and (
                    os.name != "nt" or p.suffix.lower() == ".exe"
                ):
                    return str(p)
        except Exception as e:
            logger.error("Error scanning local files: %s", e)
        candidates = ["stockfish", "./stockfish", "/usr/games/stockfish",
                      "/usr/bin/stockfish", "/opt/homebrew/bin/stockfish"]
        return next((c for c in candidates if shutil.which(c) or Path(c).exists()), "stockfish")

    def get(self) -> chess.engine.SimpleEngine | None:
        with self.lock:
            if not self._e:
                try:
                    self._e = chess.engine.SimpleEngine.popen_uci(self._get_path())
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
    empty = {"arrows": [], "circles": []}
    if not detector or not prev_fen or not move_uci:
        return empty
    try:
        board = chess.Board(prev_fen)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return empty

        game = chess.pgn.Game()
        game.setup(board)
        game.add_variation(move)
        annotated = detector.analyze(str(game))

        arrows, circles = [], []
        blocks = re.findall(r"\{([^}]+)\}", annotated)
        for b in blocks:
            # Clean the block to get the specific reason for this block
            block_reason = re.sub(r"\[%(cal|csl)\s+[^\]]+\]", "", b).strip()
            block_reason = re.sub(r"\s+", " ", block_reason)
            
            # Parse arrows in this block
            for m in re.findall(r"\[%cal\s+([^\]]+)\]", b):
                for item in m.split(","):
                    item = item.strip()
                    if len(item) >= 5:
                        arrow = {
                            "orig": item[1:3],
                            "dest": item[3:5],
                            "brush": _BRUSH.get(item[0], "green")
                        }
                        if block_reason:
                            arrow["reason"] = block_reason
                        arrows.append(arrow)
                        
            # Parse circles in this block
            for m in re.findall(r"\[%csl\s+([^\]]+)\]", b):
                for item in m.split(","):
                    item = item.strip()
                    if len(item) >= 3:
                        circle = {
                            "orig": item[1:3],
                            "brush": _BRUSH.get(item[0], "green")
                        }
                        if block_reason:
                            circle["reason"] = block_reason
                        circles.append(circle)
 
        return {"arrows": arrows, "circles": circles}
    except Exception as e:
        logger.error("Error running chess-detect annotation parser: %s", e)
        return empty


# ── Positional & Evaluation Extractors ──────────────────────────────────
def get_eval_description(ev, player_color: str) -> str:
    """Converts an evaluation to active-player perspective, using absolute values
    to avoid sign contradictions."""
    if ev is None:
        return "maintains a highly balanced position"

    opp_color = "Black" if player_color == "White" else "White"

    # Forced mate
    if isinstance(ev, str) and "M" in ev:
        try:
            n = int(ev.replace("M", "").replace("+", ""))
            return (f"concedes a forced mate-in-{abs(n)}" if n < 0
                    else f"sets up a forced mate-in-{n}")
        except ValueError:
            return "leads to a forced mate sequence"

    # Numeric eval
    try:
        score = float(ev) / 100
    except (ValueError, TypeError):
        return "maintains a stable position"

    if abs(score) <= 0.3:
        return "maintains a highly balanced position"

    favored = "White" if score > 0 else "Black"
    level = "decisive" if abs(score) > 1.9 else "clear" if abs(score) > 0.9 else "slight"
    amt = f"{abs(score):.2f}"
    if player_color == favored:
        return f"secures a {level} advantage of {amt} for {player_color}"
    return f"concedes a {level} advantage of {amt} to {opp_color}"


def is_tactical_threat(board: chess.Board, move: chess.Move) -> bool:
    """Filters out quiet development moves so only genuine tactical threats are logged."""
    if board.is_capture(move) or board.gives_check(move):
        return True
    piece = board.piece_at(move.from_square)
    if not piece:
        return False
    temp = board.copy()
    temp.push(move)
    pv = PVAL[piece.piece_type]
    return any(
        (ap := temp.piece_at(sq)) is not None
        and ap.color != piece.color
        and PVAL.get(ap.piece_type, 0) > pv
        for sq in temp.attacks(move.to_square)
    )


def _own_pieces(board: chess.Board, color: chess.Color):
    """Yields (square, piece) for non-king pieces of `color`."""
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == color and p.piece_type != chess.KING:
            yield sq, p


def _name(p: chess.Piece, sq: int) -> str:
    return f"{chess.piece_name(p.piece_type).capitalize()} on {chess.square_name(sq)}"


def get_pins(board: chess.Board, color: chess.Color) -> list:
    return [_name(p, sq) for sq, p in _own_pieces(board, color) if board.is_pinned(color, sq)]


def get_hanging_pieces(board: chess.Board, color: chess.Color) -> list:
    hanging = []
    for sq, p in _own_pieces(board, color):
        attackers = board.attackers(not color, sq)
        if not attackers:
            continue
        defenders = board.attackers(color, sq)
        
        is_undefended = not defenders
        is_attacked_by_cheaper = any(
            PVAL.get(board.piece_type_at(a) or 0, 0) < PVAL.get(p.piece_type, 0)
            for a in attackers if board.piece_at(a)
        )
        has_attacker_dominance = len(attackers) > len(defenders)
        
        if is_undefended or is_attacked_by_cheaper or has_attacker_dominance:
            hanging.append(_name(p, sq))
    return hanging


def get_rook_files(board: chess.Board, color: chess.Color) -> tuple:
    open_files, semi_open_files = [], []
    rook_files = {chess.square_file(sq) for sq, p in _own_pieces(board, color)
                  if p.piece_type == chess.ROOK}
    for f in rook_files:
        own = opp = False
        for r in range(8):
            pc = board.piece_at(chess.square(f, r))
            if pc and pc.piece_type == chess.PAWN:
                if pc.color == color:
                    own = True
                else:
                    opp = True
        name = f"{chess.FILE_NAMES[f]}-file"
        if not own and not opp:
            open_files.append(name)
        elif not own and opp:
            semi_open_files.append(name)
    return open_files, semi_open_files


def get_move_fork(board: chess.Board, move: chess.Move, color: chess.Color) -> str:
    p = board.piece_at(move.to_square)
    if not p or p.color != color:
        return ""
    opp = "White" if color == chess.BLACK else "Black"
    pv = PVAL[p.piece_type]
    targets = [
        f"{opp}'s {chess.piece_name(t.piece_type)} on {chess.square_name(sq)}"
        for sq in board.attacks(move.to_square)
        if (t := board.piece_at(sq))
        and t.color != color and t.piece_type != chess.KING
        and (PVAL.get(t.piece_type, 0) >= pv or not board.is_attacked_by(not color, sq))
    ]
    if len(targets) >= 2:
        head = f"{chess.piece_name(p.piece_type).capitalize()} on {chess.square_name(move.to_square)}"
        return f"{head} attacks {', and '.join(targets)}"
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


def _null_move_threat(board: chess.Board, actor_color: str, tense: str,
                      require_minor_pieces: bool = False) -> dict | None:
    """Push a null move and report the side-to-move's strongest tactical threat.

    `tense` is the verb form used in the description ("threatens" / "threatened").
    Replaces the former perform_null_move_analysis / perform_pre_move_threat_analysis.
    """
    if board is None or board.is_check() or board.is_game_over():
        return None
    if require_minor_pieces and not any(
            p.piece_type not in (chess.PAWN, chess.KING) for p in board.piece_map().values()):
        return None

    temp = board.copy()
    try:
        temp.push(chess.Move.null())
    except Exception:
        return None

    engine = expert_eng.get()
    if engine is None:
        return None

    try:
        with expert_eng.lock:
            pv = engine.analyse(temp, chess.engine.Limit(time=0.10, depth=10)).get("pv", [])
        if not pv or not is_tactical_threat(temp, pv[0]):
            return None

        threat = pv[0]
        desc = f"{actor_color} {tense} to play {temp.san(threat)}"
        victim = board.piece_at(threat.to_square)
        if victim:
            vc = "White" if victim.color == chess.WHITE else "Black"
            desc += (f" to capture {vc}'s {chess.piece_name(victim.piece_type)} "
                     f"on {chess.square_name(threat.to_square)}")
        return {"threat_move_uci": threat.uci(), "threat_description": desc}
    except Exception as e:
        logger.error("Threat analysis failed: %s", e)
        return None


def verify_threat_resolution(prev_board: chess.Board, player_move: chess.Move, prev_threat: dict) -> str:
    if not prev_threat:
        return ""
    uci = prev_threat.get("threat_move_uci")
    if not uci:
        return ""

    threat = chess.Move.from_uci(uci)
    if player_move.to_square == threat.from_square:
        return f"Captured attacking piece on {chess.square_name(threat.from_square)}."

    target_sq = threat.to_square
    after = prev_board.copy()
    after.push(player_move)

    if player_move.from_square == target_sq and not after.is_attacked_by(after.turn, player_move.to_square):
        return f"Moved threatened piece to safe square {chess.square_name(player_move.to_square)}."
    if threat not in after.legal_moves:
        return "Blocked or prevented threat."
    if player_move.from_square != target_sq and after.is_attacked_by(prev_board.turn, target_sq):
        return f"Protected threatened piece on {chess.square_name(target_sq)}."
    return "Left threat active."


def perform_refutation_analysis(board: chess.Board) -> dict | None:
    if board.is_game_over():
        return None
    engine = expert_eng.get()
    if engine is None:
        return None

    try:
        with expert_eng.lock:
            pv = engine.analyse(board, chess.engine.Limit(time=0.12, depth=11)).get("pv", [])
        if not pv:
            return None

        ref_san = board.san(pv[0])
        temp = board.copy()
        pv_san = []
        for m in pv[:4]:
            try:
                pv_san.append(temp.san(m))
                temp.push(m)
            except Exception:
                break

        # Material change from the refuter's perspective over the PV
        def material(b):
            return sum(PVAL[p.piece_type] for p in b.piece_map().values() if p.color == chess.WHITE) \
                 - sum(PVAL[p.piece_type] for p in b.piece_map().values() if p.color == chess.BLACK)

        sign = 1 if board.turn == chess.WHITE else -1
        gain = sign * (material(temp) - material(board))

        if ref_san.endswith("#") or temp.is_checkmate():
            desc = f"opponent can play {ref_san} to deliver immediate checkmate"
        elif gain > 0:
            desc = f"opponent can play {ref_san} to gain a material advantage of {gain} point{'s' if gain > 1 else ''}"
        else:
            desc = f"opponent can play {ref_san} to gain a positional advantage"

        ref_pv = ", ".join(f"{i+1}. {m}" for i, m in enumerate(pv_san))
        return {"refutation_description": desc, "refutation_pv": ref_pv}
    except Exception as e:
        logger.error("Refutation computation failed: %s", e)
        return None


# ── Failsafe Post-Processing Filter ───────────────────────────────────
_PERIOD_PHRASES = ("without repeating", "as requested", "per the provided",
                   "per the instructions", "according to the rules")


def clean_meta_text(text: str) -> str:
    """Programmatically cleans up instruction leakages."""
    phrases = "|".join(_PERIOD_PHRASES)
    # Phrases running to the end of a sentence -> collapse to "."
    text = re.sub(rf"\b(?:{phrases})\b.*?\.(?=\s|$)", ".", text, flags=re.IGNORECASE)
    # Bare "without repeating ..." with no period -> drop entirely
    text = re.sub(r"\bwithout repeating\b.*?(?=\s|$)", "", text, flags=re.IGNORECASE)
    # "using the provided <feature|line|text|box>"
    text = re.sub(r"\busing the provided\b.*?\b(?:feature|line|text|box)\b", "", text, flags=re.IGNORECASE)
    # Whitespace / punctuation cleanup
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip()


# ── Context Pipeline Helpers ───────────────────────────────────────────
def _colors(board, prev_board):
    """Returns (player_color, opponent_color) as name strings."""
    if board:
        white_to_move = board.turn == chess.WHITE
        return ("Black" if white_to_move else "White",
                "White" if white_to_move else "Black")
    if prev_board:
        white_to_move = prev_board.turn == chess.WHITE
        return ("White" if white_to_move else "Black",
                "Black" if white_to_move else "White")
    return "Unknown", "Unknown"


def _game_phase(board):
    if not board:
        return "Middlegame"
    minors = [p for p in board.piece_map().values()
              if p.piece_type not in (chess.PAWN, chess.KING)]
    if len(minors) <= 4 or board.fullmove_number >= 40:
        return "Endgame"
    return "Opening" if board.fullmove_number <= 12 else "Middlegame"


def _eval_scores(prev_board, board):
    """Returns (prev_score, curr_score) via the shared engine, or Nones on failure."""
    engine = expert_eng.get()
    if engine is None:
        return None, None
    out = []
    for b in (prev_board, board):
        if b is None:
            out.append(None)
            continue
        try:
            with expert_eng.lock:
                r = engine.analyse(b, chess.engine.Limit(time=0.10, depth=10))
            out.append(score_val(r.get("score")))
        except Exception:
            out.append(None)
    return out[0], out[1]


def _format_eval(ev):
    if ev is None:
        return "0.0"
    try:
        if isinstance(ev, str) and "M" in ev:
            return f"Forced mate ({ev})"
        return f"{float(ev) / 100:+.2f}"
    except (ValueError, TypeError):
        return str(ev)


def _king_shield_weakened(prev_board, player_move, player_color) -> bool:
    if not player_move:
        return False
    try:
        piece = prev_board.piece_at(player_move.from_square)
        if not piece or piece.piece_type != chess.PAWN:
            return False
        player_col = chess.WHITE if player_color == "White" else chess.BLACK
        king_sq = prev_board.king(player_col)
        if king_sq is None:
            return False
        ff = chess.square_file(player_move.from_square)
        fr = chess.square_rank(player_move.from_square)
        kf = chess.square_file(king_sq)
        target_rank = 1 if player_col == chess.WHITE else 6
        return fr == target_rank and ff in (5, 6, 7) and abs(ff - kf) <= 2
    except Exception:
        return False


def _move_purpose(prev_board, player_move, game_phase, opening_name, king_shield) -> str:
    if not prev_board or not player_move:
        return ""
    try:
        piece = prev_board.piece_at(player_move.from_square)
        if not piece:
            return ""
        p_name = chess.piece_name(piece.piece_type).capitalize()
        to_name = chess.square_name(player_move.to_square)

        if king_shield:
            return f"advance the {p_name} to {to_name} but weaken king safety"
        if prev_board.is_castling(player_move):
            return "secure king safety and activate the rook via castling"
        if piece.piece_type == chess.PAWN:
            if game_phase == "Opening":
                suffix = f" in the {opening_name}" if opening_name else ""
                return f"advance the pawn to {to_name} to claim space and contest the center{suffix}"
            if game_phase == "Endgame":
                return f"advance the pawn to {to_name} to threaten promotion"
            return f"advance the pawn to {to_name} to claim space"
        if game_phase == "Opening":
            return f"develop the {p_name} to {to_name} to control key squares"
        if game_phase == "Endgame":
            return f"activate the {p_name} on {to_name} to support centralization"
        return f"position the {p_name} on {to_name} to improve coordinate activity"
    except Exception:
        return ""


_DEFAULT_PURPOSE = {
    "Opening": "fight for central control in the opening",
    "Endgame": "activate pieces in the endgame",
}


# ── Context Pipeline (Optimized Context Pruning) ──────────────────────────
def prepare_coach_context(data: dict) -> dict:
    prev_fen = data.get("prev_fen")
    current_fen = data.get("fen")
    move_san = data.get("move_san") or "Unknown"
    move_uci = data.get("move_uci")
    best_move_san = data.get("best_move_san")
    ev = data.get("eval")
    opening_name = data.get("opening_name")

    cls_label = data.get("classification") or "unknown"
    prev_board = chess.Board(prev_fen) if prev_fen else None
    board = chess.Board(current_fen) if current_fen else None
    is_checkmate = board.is_checkmate() if board else False

    player_color, opponent_color = _colors(board, prev_board)
    game_phase = _game_phase(board)

    # Parse the player's move once (used in several checks)
    player_move = None
    if prev_board and move_uci:
        try:
            player_move = chess.Move.from_uci(move_uci)
        except Exception:
            player_move = None

    # 1. Engine scoring + classification sanity check
    prev_score = curr_score = None
    if not is_checkmate:
        prev_score, curr_score = _eval_scores(prev_board, board)
    if prev_score is not None and curr_score is not None:
        delta = (curr_score - prev_score) if player_color == "White" else (prev_score - curr_score)
        if cls_label == "book" and delta < -80:
            cls_label = "inaccuracy" if delta >= -150 else "mistake"
    is_bad_move = cls_label in _BAD_CLASSES

    eval_str = _format_eval(ev)
    eval_desc = get_eval_description(ev, player_color)

    # 2. King-safety shield tracking
    king_shield = _king_shield_weakened(prev_board, player_move, player_color) \
        if not is_checkmate else False

    # 3. Move purpose
    move_purpose = _move_purpose(prev_board, player_move, game_phase, opening_name, king_shield) \
        or _DEFAULT_PURPOSE.get(game_phase, "adjust piece activity in the middlegame")

    # 4. Tactical calculations
    prev_threat = curr_threat = refutation = None
    threat_resolution = ""
    if not is_checkmate:
        # Check if we can reuse current threat passed from client
        curr_threat = data.get("threat")
        if curr_threat is None and board:
            curr_threat = _null_move_threat(board, player_color, "threatens",
                                            require_minor_pieces=True)

        # Check if we can reconstruct previous threat from threat_move_uci
        prev_threat_uci = data.get("prev_threat_uci")
        if prev_threat_uci and prev_board:
            try:
                threat_move = chess.Move.from_uci(prev_threat_uci)
                desc = f"{opponent_color} threatened to play {prev_board.san(threat_move)}"
                victim = prev_board.piece_at(threat_move.to_square)
                if victim:
                    vc = "White" if victim.color == chess.WHITE else "Black"
                    desc += f" to capture {vc}'s {chess.piece_name(victim.piece_type)} on {chess.square_name(threat_move.to_square)}"
                prev_threat = {"threat_move_uci": prev_threat_uci, "threat_description": desc}
            except Exception as e:
                logger.error("Error reconstructing prev_threat: %s", e)
                prev_threat = None
        
        if prev_threat is None and prev_board:
            prev_threat = _null_move_threat(prev_board, opponent_color, "threatened")

        refutation = perform_refutation_analysis(board) if (board and is_bad_move) else None
        if prev_board and player_move and prev_threat:
            try:
                threat_resolution = verify_threat_resolution(prev_board, player_move, prev_threat)
            except Exception:
                pass

    # 5. Positional deltas (forks / newly-opened rook files) — only for good moves
    fork_made = ""
    new_open_rooks, new_semi_rooks = [], []
    if board and prev_board and player_move and not is_bad_move and not is_checkmate:
        try:
            player_col = chess.WHITE if player_color == "White" else chess.BLACK
            fork_made = get_move_fork(board, player_move, player_col)
            open_prev, semi_prev = get_rook_files(prev_board, player_col)
            open_curr, semi_curr = get_rook_files(board, player_col)
            new_open_rooks = [r for r in open_curr if r not in open_prev]
            new_semi_rooks = [r for r in semi_curr if r not in semi_prev]
        except Exception as e:
            logger.error("Error calculating positional deltas: %s", e)

    # 6. Compact feature mapping
    feature_lines = [f"- Player: {player_color}", f"- Phase: {game_phase}"]
    if is_checkmate:
        feature_lines += [f"- Class: checkmate", f"- Action: Delivered checkmate via {move_san}"]
    else:
        feature_lines += [f"- Class: {cls_label}", f"- Eval: {eval_str}"]

        if is_bad_move and refutation:
            feature_lines.append(f"- Opponent Refutation: {refutation['refutation_description']}")
            feature_lines.append(f"- Refutation PV: {refutation['refutation_pv']}")

        if board and prev_board and player_move and not is_bad_move:
            if prev_board.is_castling(player_move):
                feature_lines.append("- Action: Castled safely")
            if prev_board.is_capture(player_move):
                tgt = prev_board.piece_at(player_move.to_square)
                if tgt:
                    feature_lines.append(
                        f"- Action: Captured opponent's {chess.piece_name(tgt.piece_type)} "
                        f"on {chess.square_name(player_move.to_square)}")
            if fork_made:
                feature_lines.append(f"- Fork: {fork_made}")
            if new_open_rooks:
                feature_lines.append(f"- Rook: Open file ({', '.join(new_open_rooks)})")
            elif new_semi_rooks:
                feature_lines.append(f"- Rook: Semi-open file ({', '.join(new_semi_rooks)})")

        if best_move_san and best_move_san != move_san:
            feature_lines.append(f"- Best Alternative for {player_color}: {best_move_san}")
        if opening_name:
            feature_lines.append(f"- Opening: {opening_name}")

        # Worst-Placed Piece Info
        wpp_info = data.get("worst_placed_piece")
        if not wpp_info and board:
            try:
                wpp_info = WorstPlacedPieceAnalyzer(board).get_wpp()
            except Exception as e:
                logger.error("Error calculating WPP in prepare_coach_context: %s", e)
                wpp_info = None
        if wpp_info and wpp_info.get("wpp_name"):
            feature_lines.append(f"- Worst-Placed Piece: {wpp_info['wpp_name']} ({int(wpp_info['mobility_ratio']*100)}% active)")
            if wpp_info.get("maneuver_path") and len(wpp_info["maneuver_path"]) >= 2:
                feature_lines.append(f"- Maneuver Path: {' -> '.join(wpp_info['maneuver_path'])}")

    eval_context = ""
    if ev is not None and not is_bad_move and cls_label == "book" and not is_checkmate:
        eval_context = f"(Matches standard theory despite engine eval of {eval_str}.)"

    is_forced_mate = (not is_checkmate and ev is not None
                      and isinstance(ev, str) and "M" in ev)

    # 7. Compile benefit detail
    benefits = []
    if not is_checkmate:
        if fork_made:
            benefits.append(f"creating a fork ({fork_made})")
        if new_open_rooks:
            benefits.append(f"activating a rook on open file ({', '.join(new_open_rooks)})")
        elif new_semi_rooks:
            benefits.append(f"activating a rook on semi-open file ({', '.join(new_semi_rooks)})")
    benefit_detail = " and ".join(benefits)

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
        "refutation_pv": refutation["refutation_pv"] if refutation else "",
    }


class PositionalAggregator:
    def __init__(self, board: chess.Board):
        self.board = board

    def get_space_score(self) -> float:
        """
        Advanced Space: Evaluates weighted square control and outposts.
        """
        white_points = 0.0
        black_points = 0.0

        for square in chess.SQUARES:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            
            # Determine weight of the square (higher in center)
            weight = 1.0
            if file in [2, 3, 4, 5]:  # Files C, D, E, F
                weight = 1.5 if rank in [2, 3, 4, 5] else 1.2

            # Check attackers to implement "least-valuable-attacker" rule
            w_attackers = self.board.attackers(chess.WHITE, square)
            b_attackers = self.board.attackers(chess.BLACK, square)
            
            if w_attackers or b_attackers:
                if w_attackers and not b_attackers:
                    if rank >= 4:
                        white_points += (1.0 * weight)
                elif b_attackers and not w_attackers:
                    if rank <= 3:
                        black_points += (1.0 * weight)
                else:  # Both sides attack the square
                    w_types = [self.board.piece_type_at(sq) for sq in w_attackers]
                    b_types = [self.board.piece_type_at(sq) for sq in b_attackers]
                    min_w = min(PVAL.get(pt, 1) for pt in w_types if pt is not None)
                    min_b = min(PVAL.get(pt, 1) for pt in b_types if pt is not None)
                    
                    if min_w < min_b:
                        if rank >= 4:
                            white_points += (1.0 * weight)
                    elif min_b < min_w:
                        if rank <= 3:
                            black_points += (1.0 * weight)
                    else:  # Equal value lowest attackers, fall back to total count dominance
                        if len(w_attackers) > len(b_attackers):
                            if rank >= 4:
                                white_points += (1.0 * weight)
                        elif len(b_attackers) > len(w_attackers):
                            if rank <= 3:
                                black_points += (1.0 * weight)

            # Outpost detection: Knight or Bishop safely defended by pawn in opponent's half
            piece = self.board.piece_at(square)
            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                is_defended_by_pawn = any(
                    self.board.piece_type_at(def_sq) == chess.PAWN 
                    for def_sq in self.board.attackers(piece.color, square)
                )
                if is_defended_by_pawn:
                    if piece.color == chess.WHITE and rank >= 4:
                        white_points += 2.0  # Big bonus for active outpost
                    elif piece.color == chess.BLACK and rank <= 3:
                        black_points += 2.0

        total = white_points + black_points
        return (white_points - black_points) / total if total > 0 else 0.0

    def get_king_safety(self) -> float:
        """
        Advanced King Safety: Checks pawn shields, open files, the King Ring, and shield hooks.
        """
        def evaluate_color_safety(color) -> float:
            king_sq = self.board.king(color)
            if king_sq is None:
                return 0.0

            safety_score = 100.0  # Start with perfect safety
            opp_color = not color

            # 1. King Ring Analysis (squares around the King)
            king_file = chess.square_file(king_sq)
            king_rank = chess.square_rank(king_sq)
            king_ring_squares = []
            
            for f_offset in [-1, 0, 1]:
                for r_offset in [-1, 0, 1]:
                    if f_offset == 0 and r_offset == 0:
                        continue
                    t_file, t_rank = king_file + f_offset, king_rank + r_offset
                    if 0 <= t_file <= 7 and 0 <= t_rank <= 7:
                        king_ring_squares.append(chess.square(t_file, t_rank))

            # Penalize based on enemy attackers aiming at the King Ring
            for sq in king_ring_squares:
                attackers = self.board.attackers(opp_color, sq)
                for attacker_sq in attackers:
                    attacker_piece = self.board.piece_at(attacker_sq)
                    if attacker_piece:
                        # Heavy penalties for major pieces
                        if attacker_piece.piece_type == chess.QUEEN:
                            safety_score -= 8.0
                        elif attacker_piece.piece_type == chess.ROOK:
                            safety_score -= 5.0
                        else:
                            safety_score -= 3.0

            # 2. Open Files flank checks (adjacent files to castled/starting king)
            files_to_check = [max(0, king_file - 1), king_file, min(7, king_file + 1)]
            for file_idx in set(files_to_check):
                has_friendly_pawn = False
                has_enemy_pawn = False
                for r in range(8):
                    p = self.board.piece_at(chess.square(file_idx, r))
                    if p and p.piece_type == chess.PAWN:
                        if p.color == color:
                            has_friendly_pawn = True
                        else:
                            has_enemy_pawn = True
                
                if not has_friendly_pawn and not has_enemy_pawn:
                    safety_score -= 15.0  # Fully open file penalty
                elif not has_friendly_pawn:
                    safety_score -= 8.0   # Semi-open file penalty

            # 3. Pawn Shield Hook / Integrity check
            for file_idx in set(files_to_check):
                friendly_pawn_rank = None
                for r in range(8):
                    p = self.board.piece_at(chess.square(file_idx, r))
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        friendly_pawn_rank = r
                        break
                
                if friendly_pawn_rank is not None:
                    # Calculate rank distance relative to the King
                    rel_rank = friendly_pawn_rank - king_rank if color == chess.WHITE else king_rank - friendly_pawn_rank
                    if rel_rank == 2:  # Pushed one square (e.g. g3/f3 pawn hook)
                        safety_score -= 8.0
                    elif rel_rank >= 3:  # Pushed two or more squares -> highly compromised
                        safety_score -= 20.0

            return max(0.0, safety_score) / 100.0

        w_safety = evaluate_color_safety(chess.WHITE)
        b_safety = evaluate_color_safety(chess.BLACK)
        return w_safety - b_safety

    def get_pawn_structure(self) -> float:
        """
        Advanced Pawn Structure: Evaluates doubled, isolated, backward, and passed pawns.
        """
        def evaluate_pawns(color) -> float:
            score = 0.0
            opp_color = not color
            pawns = self.board.pieces(chess.PAWN, color)

            for sq in pawns:
                file_idx = chess.square_file(sq)
                rank_idx = chess.square_rank(sq)

                # 1. Isolated Pawn Check
                adjacent_files = [file_idx - 1, file_idx + 1]
                is_isolated = True
                for adj_f in adjacent_files:
                    if 0 <= adj_f <= 7:
                        for r in range(8):
                            p = self.board.piece_at(chess.square(adj_f, r))
                            if p and p.piece_type == chess.PAWN and p.color == color:
                                is_isolated = False
                                break
                if is_isolated:
                    score -= 0.20

                # 2. Passed Pawn Check
                is_passed = True
                files_to_check = [file_idx - 1, file_idx, file_idx + 1]
                ranks_to_check = range(rank_idx + 1, 8) if color == chess.WHITE else range(0, rank_idx)
                
                for f in files_to_check:
                    if 0 <= f <= 7:
                        for r in ranks_to_check:
                            p = self.board.piece_at(chess.square(f, r))
                            if p and p.piece_type == chess.PAWN and p.color == opp_color:
                                is_passed = False
                                break
                if is_passed:
                    advancement = rank_idx if color == chess.WHITE else (7 - rank_idx)
                    score += 0.15 + (advancement * 0.05)

                # 3. Doubled Pawn Check
                same_file_ranks = range(rank_idx + 1, 8) if color == chess.WHITE else range(0, rank_idx)
                for r in same_file_ranks:
                    p = self.board.piece_at(chess.square(file_idx, r))
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        score -= 0.15
                        break

                # 4. Backward Pawn Check
                has_pawn_behind = False
                for adj_f in adjacent_files:
                    if 0 <= adj_f <= 7:
                        behind_ranks = range(0, rank_idx) if color == chess.WHITE else range(rank_idx + 1, 8)
                        for r in behind_ranks:
                            p = self.board.piece_at(chess.square(adj_f, r))
                            if p and p.piece_type == chess.PAWN and p.color == color:
                                has_pawn_behind = True
                                break
                
                has_adj_files_pawns = False
                for adj_f in adjacent_files:
                    if 0 <= adj_f <= 7:
                        for r in range(8):
                            p = self.board.piece_at(chess.square(adj_f, r))
                            if p and p.piece_type == chess.PAWN and p.color == color:
                                has_adj_files_pawns = True
                                break
                
                if has_adj_files_pawns and not has_pawn_behind:
                    front_rank = rank_idx + (1 if color == chess.WHITE else -1)
                    if 0 <= front_rank <= 7:
                        front_sq = chess.square(file_idx, front_rank)
                        enemy_pawn_attackers = any(
                            self.board.piece_type_at(atk) == chess.PAWN and self.board.color_at(atk) == opp_color
                            for atk in self.board.attackers(opp_color, front_sq)
                        )
                        if enemy_pawn_attackers:
                            score -= 0.15

            return score

        w_structure = evaluate_pawns(chess.WHITE)
        b_structure = evaluate_pawns(chess.BLACK)
        
        diff = w_structure - b_structure
        return max(-1.0, min(1.0, diff))

    def get_space_details(self) -> list:
        highlights = []
        for square in chess.SQUARES:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            sq_name = chess.square_name(square)
            
            weight = 1.0
            if file in [2, 3, 4, 5]:
                weight = 1.5 if rank in [2, 3, 4, 5] else 1.2

            w_attackers = self.board.attackers(chess.WHITE, square)
            b_attackers = self.board.attackers(chess.BLACK, square)
            
            if w_attackers or b_attackers:
                if w_attackers and not b_attackers:
                    if rank >= 4:
                        highlights.append({
                            "orig": sq_name,
                            "brush": "green",
                            "reason": f"White controls {sq_name} exclusively (weight: {weight})"
                        })
                elif b_attackers and not w_attackers:
                    if rank <= 3:
                        highlights.append({
                            "orig": sq_name,
                            "brush": "red",
                            "reason": f"Black controls {sq_name} exclusively (weight: {weight})"
                        })
                else:
                    w_types = [self.board.piece_type_at(sq) for sq in w_attackers]
                    b_types = [self.board.piece_type_at(sq) for sq in b_attackers]
                    min_w = min(PVAL.get(pt, 1) for pt in w_types if pt is not None)
                    min_b = min(PVAL.get(pt, 1) for pt in b_types if pt is not None)
                    
                    if min_w < min_b:
                        if rank >= 4:
                            highlights.append({
                                "orig": sq_name,
                                "brush": "green",
                                "reason": f"White controls {sq_name} via cheaper attacker (weight: {weight})"
                            })
                    elif min_b < min_w:
                        if rank <= 3:
                            highlights.append({
                                "orig": sq_name,
                                "brush": "red",
                                "reason": f"Black controls {sq_name} via cheaper attacker (weight: {weight})"
                            })
                    else:
                        if len(w_attackers) > len(b_attackers):
                            if rank >= 4:
                                highlights.append({
                                    "orig": sq_name,
                                    "brush": "green",
                                    "reason": f"White controls {sq_name} via attacker count dominance (weight: {weight})"
                                })
                        elif len(b_attackers) > len(w_attackers):
                            if rank <= 3:
                                highlights.append({
                                    "orig": sq_name,
                                    "brush": "red",
                                    "reason": f"Black controls {sq_name} via attacker count dominance (weight: {weight})"
                                })

            # Outposts
            piece = self.board.piece_at(square)
            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                is_defended_by_pawn = any(
                    self.board.piece_type_at(def_sq) == chess.PAWN 
                    for def_sq in self.board.attackers(piece.color, square)
                )
                if is_defended_by_pawn:
                    if piece.color == chess.WHITE and rank >= 4:
                        highlights.append({
                            "orig": sq_name,
                            "brush": "green",
                            "reason": f"White active outpost: {piece.unicode_symbol()} on {sq_name} (defended by pawn)"
                        })
                    elif piece.color == chess.BLACK and rank <= 3:
                        highlights.append({
                            "orig": sq_name,
                            "brush": "red",
                            "reason": f"Black active outpost: {piece.unicode_symbol()} on {sq_name} (defended by pawn)"
                        })
        return highlights

    def get_king_safety_details(self) -> list:
        highlights = []
        for color in [chess.WHITE, chess.BLACK]:
            king_sq = self.board.king(color)
            if king_sq is None:
                continue
            
            brush = "green" if color == chess.WHITE else "red"
            opp_color = not color
            opp_brush = "red" if color == chess.WHITE else "green"
            
            # King square itself
            highlights.append({
                "orig": chess.square_name(king_sq),
                "brush": brush,
                "reason": f"{'White' if color == chess.WHITE else 'Black'} King position"
            })
            
            # 1. King Ring Analysis
            king_file = chess.square_file(king_sq)
            king_rank = chess.square_rank(king_sq)
            
            for f_offset in [-1, 0, 1]:
                for r_offset in [-1, 0, 1]:
                    if f_offset == 0 and r_offset == 0:
                        continue
                    t_file, t_rank = king_file + f_offset, king_rank + r_offset
                    if 0 <= t_file <= 7 and 0 <= t_rank <= 7:
                        sq = chess.square(t_file, t_rank)
                        sq_name = chess.square_name(sq)
                        attackers = self.board.attackers(opp_color, sq)
                        if attackers:
                            highlights.append({
                                "orig": sq_name,
                                "brush": opp_brush,
                                "reason": f"King Ring square {sq_name} under attack by {len(attackers)} enemy piece(s)"
                            })
                            
            # 2. Open / Semi-open files next to King
            files_to_check = [max(0, king_file - 1), king_file, min(7, king_file + 1)]
            for file_idx in set(files_to_check):
                has_friendly_pawn = False
                has_enemy_pawn = False
                for r in range(8):
                    p = self.board.piece_at(chess.square(file_idx, r))
                    if p and p.piece_type == chess.PAWN:
                        if p.color == color:
                            has_friendly_pawn = True
                        else:
                            has_enemy_pawn = True
                
                if not has_friendly_pawn:
                    file_name = chr(97 + file_idx)
                    file_desc = "Fully open file" if not has_enemy_pawn else "Semi-open file"
                    for r in range(max(0, king_rank - 2), min(8, king_rank + 3)):
                        highlights.append({
                            "orig": chess.square_name(chess.square(file_idx, r)),
                            "brush": "blue",
                            "reason": f"{file_desc} ({file_name}-file) adjacent to the King"
                        })

            # 3. Pawn Shield Hook
            for file_idx in set(files_to_check):
                friendly_pawn_rank = None
                for r in range(8):
                    p = self.board.piece_at(chess.square(file_idx, r))
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        friendly_pawn_rank = r
                        break
                
                if friendly_pawn_rank is not None:
                    rel_rank = friendly_pawn_rank - king_rank if color == chess.WHITE else king_rank - friendly_pawn_rank
                    sq_name = chess.square_name(chess.square(file_idx, friendly_pawn_rank))
                    if rel_rank == 2:
                        highlights.append({
                            "orig": sq_name,
                            "brush": "yellow",
                            "reason": f"Pushed pawn shield (hook): friendly pawn on {sq_name}"
                        })
                    elif rel_rank >= 3:
                        highlights.append({
                            "orig": sq_name,
                            "brush": "yellow",
                            "reason": f"Highly pushed/compromised pawn shield: friendly pawn on {sq_name}"
                        })
        return highlights

    def get_pawn_structure_details(self) -> list:
        highlights = []
        for color in [chess.WHITE, chess.BLACK]:
            opp_color = not color
            pawns = self.board.pieces(chess.PAWN, color)
            
            for sq in pawns:
                sq_name = chess.square_name(sq)
                file_idx = chess.square_file(sq)
                rank_idx = chess.square_rank(sq)
                
                # 1. Isolated
                adjacent_files = [file_idx - 1, file_idx + 1]
                is_isolated = True
                for adj_f in adjacent_files:
                    if 0 <= adj_f <= 7:
                        for r in range(8):
                            p = self.board.piece_at(chess.square(adj_f, r))
                            if p and p.piece_type == chess.PAWN and p.color == color:
                                is_isolated = False
                                break
                if is_isolated:
                    highlights.append({
                        "orig": sq_name,
                        "brush": "yellow",
                        "reason": f"{'White' if color == chess.WHITE else 'Black'} Isolated pawn on {sq_name}"
                    })

                # 2. Passed Pawn
                is_passed = True
                files_to_check = [file_idx - 1, file_idx, file_idx + 1]
                ranks_to_check = range(rank_idx + 1, 8) if color == chess.WHITE else range(0, rank_idx)
                
                for f in files_to_check:
                    if 0 <= f <= 7:
                        for r in ranks_to_check:
                            p = self.board.piece_at(chess.square(f, r))
                            if p and p.piece_type == chess.PAWN and p.color == opp_color:
                                is_passed = False
                                break
                if is_passed:
                    highlights.append({
                        "orig": sq_name,
                        "brush": "green",
                        "reason": f"{'White' if color == chess.WHITE else 'Black'} Passed pawn on {sq_name} (advanced to rank {rank_idx + 1})"
                    })

                # 3. Doubled Pawn
                same_file_ranks = range(rank_idx + 1, 8) if color == chess.WHITE else range(0, rank_idx)
                is_doubled = False
                for r in same_file_ranks:
                    p = self.board.piece_at(chess.square(file_idx, r))
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        is_doubled = True
                        break
                if is_doubled:
                    highlights.append({
                        "orig": sq_name,
                        "brush": "yellow",
                        "reason": f"{'White' if color == chess.WHITE else 'Black'} Doubled pawn on {sq_name}"
                    })

                # 4. Backward Pawn
                has_pawn_behind = False
                for adj_f in adjacent_files:
                    if 0 <= adj_f <= 7:
                        behind_ranks = range(0, rank_idx) if color == chess.WHITE else range(rank_idx + 1, 8)
                        for r in behind_ranks:
                            p = self.board.piece_at(chess.square(adj_f, r))
                            if p and p.piece_type == chess.PAWN and p.color == color:
                                has_pawn_behind = True
                                break
                
                has_adj_files_pawns = False
                for adj_f in adjacent_files:
                    if 0 <= adj_f <= 7:
                        for r in range(8):
                            p = self.board.piece_at(chess.square(adj_f, r))
                            if p and p.piece_type == chess.PAWN and p.color == color:
                                has_adj_files_pawns = True
                                break
                
                if has_adj_files_pawns and not has_pawn_behind:
                    front_rank = rank_idx + (1 if color == chess.WHITE else -1)
                    if 0 <= front_rank <= 7:
                        front_sq = chess.square(file_idx, front_rank)
                        enemy_pawn_attackers = any(
                            self.board.piece_type_at(atk) == chess.PAWN and self.board.color_at(atk) == opp_color
                            for atk in self.board.attackers(opp_color, front_sq)
                        )
                        if enemy_pawn_attackers:
                            highlights.append({
                                "orig": sq_name,
                                "brush": "red",
                                "reason": f"{'White' if color == chess.WHITE else 'Black'} Backward pawn on {sq_name}"
                            })
        return highlights

    def get_positional_details(self) -> dict:
        return {
            "space": self.get_space_details(),
            "king_safety": self.get_king_safety_details(),
            "structure": self.get_pawn_structure_details()
        }

    def get_positional_balance(self) -> dict:
        return {
            "space": round(self.get_space_score(), 2),
            "king_safety": round(self.get_king_safety(), 2),
            "structure": round(self.get_pawn_structure(), 2)
        }


class WorstPlacedPieceAnalyzer:
    def __init__(self, board: chess.Board):
        self.board = board
        self.turn = board.turn

    def get_piece_mobility_ratio(self, square: chess.Square) -> float:
        """
        Calculates the ratio of actual moves to maximum theoretical moves 
        for a given piece.
        """
        piece = self.board.piece_at(square)
        if not piece:
            return 1.0

        # Max theoretical mobilities
        max_mobility = {
            chess.KNIGHT: 8,
            chess.BISHOP: 13,
            chess.ROOK: 14,
        }
        
        if piece.piece_type not in max_mobility:
            return 1.0

        # Simple attack count as proxy for mobility in static position
        actual_moves = len(self.board.attacks(square))
        
        # Add extra penalty for "Bad Bishops" blocked by own pawns
        if piece.piece_type == chess.BISHOP:
            own_pawns = self.board.pieces(chess.PAWN, piece.color)
            pawn_color_match = sum(1 for p_sq in own_pawns if (chess.square_file(p_sq) + chess.square_rank(p_sq)) % 2 == (chess.square_file(square) + chess.square_rank(square)) % 2)
            if pawn_color_match >= 5: # Highly blocked by own pawns
                actual_moves = max(0, actual_moves - 2)

        return actual_moves / max_mobility[piece.piece_type]

    def get_accessible_squares_at(self, current_sq: chess.Square, start_sq: chess.Square, piece: chess.Piece) -> list[chess.Square]:
        original_piece_at_current = self.board.piece_at(current_sq)
        original_piece_at_start = self.board.piece_at(start_sq)
        
        if current_sq != start_sq:
            self.board.remove_piece_at(start_sq)
            self.board.set_piece_at(current_sq, piece)
            
        targets = []
        for sq in self.board.attacks(current_sq):
            tgt_piece = self.board.piece_at(sq)
            if tgt_piece and tgt_piece.color == piece.color:
                continue
            targets.append(sq)
            
        if current_sq != start_sq:
            if original_piece_at_start:
                self.board.set_piece_at(start_sq, original_piece_at_start)
            else:
                self.board.remove_piece_at(start_sq)
                
            if original_piece_at_current:
                self.board.set_piece_at(current_sq, original_piece_at_current)
            else:
                self.board.remove_piece_at(current_sq)
                
        return targets

    def is_square_safe_stockfish(self, square: chess.Square, start_sq: chess.Square, piece: chess.Piece, current_eval: int) -> bool:
        original_piece_at_target = self.board.piece_at(square)
        original_piece_at_start = self.board.piece_at(start_sq)
        
        self.board.remove_piece_at(start_sq)
        self.board.set_piece_at(square, piece)
        
        engine = expert_eng.get()
        safe = True
        if engine:
            try:
                with expert_eng.lock:
                    info = engine.analyse(self.board, chess.engine.Limit(time=0.03, depth=8))
                score = info.get("score")
                if score:
                    if score.is_mate():
                        mate_moves = score.relative.mate()
                        if mate_moves is not None and mate_moves < 0:
                            safe = False
                    else:
                        val = score.relative.score()
                        if val is not None and val < current_eval - 150:
                            safe = False
            except Exception as e:
                logger.error("Stockfish safety check failed: %s", e)
                
        if original_piece_at_start:
            self.board.set_piece_at(start_sq, original_piece_at_start)
        else:
            self.board.remove_piece_at(start_sq)
            
        if original_piece_at_target:
            self.board.set_piece_at(square, original_piece_at_target)
        else:
            self.board.remove_piece_at(square)
            
        return safe

    def is_target_outpost_at(self, square: chess.Square, start_sq: chess.Square, piece: chess.Piece, current_eval: int) -> bool:
        if square == start_sq:
            return False
            
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        in_opp_territory = (rank >= 4 if piece.color == chess.WHITE else rank <= 3)
        in_center = (rank in [2, 3, 4, 5] and file in [2, 3, 4, 5])
        if not (in_opp_territory or in_center):
            return False
            
        enemy_color = not piece.color
        is_attacked_by_pawn = any(
            self.board.piece_type_at(atk) == chess.PAWN and self.board.color_at(atk) == enemy_color
            for atk in self.board.attackers(enemy_color, square)
        )
        if is_attacked_by_pawn:
            return False
            
        original_piece_at_target = self.board.piece_at(square)
        original_piece_at_start = self.board.piece_at(start_sq)
        
        self.board.remove_piece_at(start_sq)
        self.board.set_piece_at(square, piece)
        
        ratio = self.get_piece_mobility_ratio(square)
        
        if original_piece_at_start:
            self.board.set_piece_at(start_sq, original_piece_at_start)
        else:
            self.board.remove_piece_at(start_sq)
            
        if original_piece_at_target:
            self.board.set_piece_at(square, original_piece_at_target)
        else:
            self.board.remove_piece_at(square)
            
        if ratio > 0.60:
            return self.is_square_safe_stockfish(square, start_sq, piece, current_eval)
        return False

    def find_maneuver_path(self, start_sq: chess.Square, piece: chess.Piece, current_eval: int) -> list[str]:
        """
        BFS pathfinder to find the quickest route to an active square.
        """
        from collections import deque
        queue = deque([(start_sq, [start_sq])])
        visited = {start_sq}
        max_depth = 4
        
        while queue:
            curr_sq, path = queue.popleft()
            
            if curr_sq != start_sq and self.is_target_outpost_at(curr_sq, start_sq, piece, current_eval):
                return [chess.square_name(sq) for sq in path]
                
            if len(path) - 1 >= max_depth:
                continue
                
            for next_sq in self.get_accessible_squares_at(curr_sq, start_sq, piece):
                if next_sq not in visited:
                    visited.add(next_sq)
                    queue.append((next_sq, path + [next_sq]))
                    
        return []

    def get_wpp(self) -> dict:
        """
        Scan all minor pieces/rooks of the side-to-move, find the worst ones (up to 3),
        and calculate their improvement paths using phase-based heuristics.
        """
        side_pieces = []
        is_opening = self.board.fullmove_number <= 12

        # Get current evaluation as baseline for Stockfish safety checks
        current_eval = 0
        engine = expert_eng.get()
        if engine:
            try:
                with expert_eng.lock:
                    info = engine.analyse(self.board, chess.engine.Limit(time=0.03, depth=8))
                score = info.get("score")
                if score:
                    if score.is_mate():
                        m = score.relative.mate()
                        current_eval = -20000 if m and m < 0 else 20000
                    else:
                        val = score.relative.score()
                        if val is not None:
                            current_eval = val
            except Exception as e:
                logger.error("Error getting current eval in get_wpp: %s", e)

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.turn and piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
                # 1. Exclude Rooks in the Opening Phase
                if piece.piece_type == chess.ROOK and is_opening:
                    continue

                ratio = self.get_piece_mobility_ratio(square)
                sort_ratio = ratio

                # 2. Differentiate "Undeveloped" vs. "Misplaced" Minor Pieces
                starting_minor_squares = {
                    (chess.WHITE, chess.KNIGHT): {chess.B1, chess.G1},
                    (chess.WHITE, chess.BISHOP): {chess.C1, chess.F1},
                    (chess.BLACK, chess.KNIGHT): {chess.B8, chess.G8},
                    (chess.BLACK, chess.BISHOP): {chess.C8, chess.F8},
                }
                if (piece.color, piece.piece_type) in starting_minor_squares and square in starting_minor_squares[(piece.color, piece.piece_type)]:
                    sort_ratio += 0.50

                # 3. Respect Castling and Rook Dormancy
                if piece.piece_type == chess.ROOK:
                    rook_start = {chess.A1, chess.H1} if piece.color == chess.WHITE else {chess.A8, chess.H8}
                    if square in rook_start:
                        king_start = chess.E1 if piece.color == chess.WHITE else chess.E8
                        has_castling_rights = self.board.has_kingside_castling_rights(piece.color) or self.board.has_queenside_castling_rights(piece.color)
                        king_on_start = self.board.king(piece.color) == king_start
                        if king_on_start or has_castling_rights:
                            sort_ratio += 0.50

                side_pieces.append((square, piece, ratio, sort_ratio))

        if not side_pieces:
            return {"worst_pieces_list": [], "wpp_square": "", "wpp_name": "", "mobility_ratio": 1.0, "maneuver_path": []}

        # Sort by lowest sort_ratio, breaking ties deterministically by mixed square index
        side_pieces.sort(key=lambda x: (x[3], (x[0] * 17) % 64))
        
        worst_pieces_list = []
        for x in side_pieces:
            w_sq, w_piece, w_ratio, w_sort_ratio = x
            w_name = f"{chess.piece_name(w_piece.piece_type).capitalize()} on {chess.square_name(w_sq)}"
            path = self.find_maneuver_path(w_sq, w_piece, current_eval)
            worst_pieces_list.append({
                "wpp_square": chess.square_name(w_sq),
                "wpp_name": w_name,
                "mobility_ratio": round(w_ratio, 2),
                "maneuver_path": path
            })
            
        primary = worst_pieces_list[0]
        return {
            "worst_pieces_list": worst_pieces_list,
            "wpp_square": primary["wpp_square"],
            "wpp_name": primary["wpp_name"],
            "mobility_ratio": primary["mobility_ratio"],
            "maneuver_path": primary["maneuver_path"]
        }

