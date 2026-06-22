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

        # Textual reason: strip visual markings [%cal ...] / [%csl ...]
        reasons = []
        for c in re.findall(r"\{([^}]+)\}", annotated):
            cleaned = re.sub(r"\[%(cal|csl)\s+[^\]]+\]", "", c).strip()
            if cleaned:
                reasons.append(cleaned)
        reason_text = "; ".join(reasons)

        def make_item(d):
            if reason_text:
                d["reason"] = reason_text
            return d

        arrows, circles = [], []
        for m in re.findall(r"\[%cal\s+([^\]]+)\]", annotated):
            for item in m.split(","):
                item = item.strip()
                if len(item) >= 5:
                    arrows.append(make_item({"orig": item[1:3], "dest": item[3:5],
                                             "brush": _BRUSH.get(item[0], "green")}))
        for m in re.findall(r"\[%csl\s+([^\]]+)\]", annotated):
            for item in m.split(","):
                item = item.strip()
                if len(item) >= 3:
                    circles.append(make_item({"orig": item[1:3],
                                              "brush": _BRUSH.get(item[0], "green")}))

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
    return [_name(p, sq) for sq, p in _own_pieces(board, color)
            if board.is_attacked_by(not color, sq) and not board.is_attacked_by(color, sq)]


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
        prev_threat = _null_move_threat(prev_board, opponent_color, "threatened") if prev_board else None
        curr_threat = _null_move_threat(board, player_color, "threatens",
                                        require_minor_pieces=True) if board else None
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
