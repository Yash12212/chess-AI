import re
import os
import json
import shutil
import threading
import atexit
import logging
from pathlib import Path
from collections import deque

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

PVAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}

_BRUSH = {"G": "green", "R": "red", "B": "blue", "Y": "yellow"}
_BAD_CLASSES = ("inaccuracy", "mistake", "blunder")


def _pawn_can_attack(board: chess.Board, sq: chess.Square, attacker_color: chess.Color) -> bool:
    """True if any pawn of attacker_color can attack sq now or by advancing (structural outpost check)."""
    r, f = chess.square_rank(sq), chess.square_file(sq)
    for df in (-1, 1):
        af = f + df
        if not (0 <= af <= 7):
            continue
        ranks = range(0, r) if attacker_color == chess.WHITE else range(r + 1, 8)
        for ar in ranks:
            p = board.piece_at(chess.square(af, ar))
            if p and p.piece_type == chess.PAWN and p.color == attacker_color:
                return True
    return False


class ExpertEngineManager:
    """Manages an auxiliary Stockfish instance to compute threats and refutations."""
    def __init__(self):
        self._e = None
        self.lock = threading.RLock()

    def _get_path(self) -> str:
        try:
            for p in Path(".").iterdir():
                if "stockfish" in p.name.lower() and p.is_file() and (os.name != 'nt' or p.suffix.lower() == '.exe'):
                    return str(p)
        except Exception:
            pass
        candidates = ["stockfish", "./stockfish", "/usr/games/stockfish", "/usr/bin/stockfish", "/opt/homebrew/bin/stockfish"]
        return next((c for c in candidates if shutil.which(c) or Path(c).exists()), "stockfish")

    def get(self) -> chess.engine.SimpleEngine | None:
        with self.lock:
            if self._e:
                try:
                    self._e.ping()
                except Exception as e:
                    logger.warning("Expert Stockfish safety check failed: %s. Restarting engine.", e)
                    try:
                        self._e.quit()
                    except Exception:
                        pass
                    self._e = None
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
        for block in re.findall(r"\{([^}]+)\}", annotated):
            reason = re.sub(r"\s+", " ", re.sub(r"\[%(cal|csl)\s+[^\]]+\]", "", block)).strip()

            for m in re.findall(r"\[%cal\s+([^\]]+)\]", block):
                for item in m.split(","):
                    item = item.strip()
                    if len(item) >= 5:
                        arr = {"orig": item[1:3].lower(), "dest": item[3:5].lower(),
                               "brush": _BRUSH.get(item[0].upper(), "green")}
                        if reason: arr["reason"] = reason
                        arrows.append(arr)

            for m in re.findall(r"\[%csl\s+([^\]]+)\]", block):
                for item in m.split(","):
                    item = item.strip()
                    if len(item) >= 3:
                        cir = {"orig": item[1:3].lower(), "brush": _BRUSH.get(item[0].upper(), "green")}
                        if reason: cir["reason"] = reason
                        circles.append(cir)
        return {"arrows": arrows, "circles": circles}
    except Exception as e:
        logger.error("Error running chess-detect annotation parser: %s", e)
        return empty


def get_eval_description(ev, player_color: str) -> str:
    """Converts an evaluation to active-player perspective."""
    if ev is None:
        return "maintains a highly balanced position"
    opp_color = "Black" if player_color == "White" else "White"

    if isinstance(ev, str) and "M" in ev:
        try:
            sign_neg = ev.strip().startswith("-")
            n = int(ev.replace("M", "").replace("+", "").replace("-", ""))
            if n == 0:
                return "delivers checkmate"
            return f"concedes a forced mate-in-{n}" if sign_neg else f"sets up a forced mate-in-{n}"
        except ValueError:
            return "leads to a forced mate sequence"

    try:
        score = float(ev) / 100
    except (ValueError, TypeError):
        return "maintains a stable position"

    if abs(score) <= 0.3:
        return "maintains a highly balanced position"

    favored = "White" if score > 0 else "Black"
    abs_s = abs(score)
    level = "decisive" if abs_s > 1.9 else "clear" if abs_s > 0.9 else "slight"

    if player_color == favored:
        return f"secures a {level} advantage of {abs_s:.2f} for {player_color}"
    return f"concedes a {level} advantage of {abs_s:.2f} to {opp_color}"


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
    if any((ap := temp.piece_at(sq)) is not None and ap.color != piece.color and PVAL.get(ap.piece_type, 0) > pv
           for sq in temp.attacks(move.to_square)):
        return True
    opp = not piece.color
    return any((ap := temp.piece_at(sq)) is not None and ap.color == opp and ap.piece_type != chess.KING
               and temp.is_pinned(opp, sq) for sq in temp.attacks(move.to_square))


def get_pins(board: chess.Board, color: chess.Color) -> list:
    if board.king(color) is None:
        return []
    return [f"{chess.piece_name(p.piece_type).capitalize()} on {chess.square_name(sq)}"
            for sq, p in board.piece_map().items()
            if p.color == color and p.piece_type != chess.KING and board.is_pinned(color, sq)]


def _see(board: chess.Board, sq: chess.Square, color: chess.Color, ptype: chess.PieceType) -> bool:
    """Static exchange evaluation. Returns True if piece on sq is not capturable at a profit."""
    opp = not color
    atk = board.attackers(opp, sq)
    if not atk:
        return True

    def _sorted(squares):
        vals = sorted(PVAL[ptype] for s in squares
                      if (ptype := board.piece_type_at(s)) is not None and ptype != chess.KING)
        if any(board.piece_type_at(s) == chess.KING for s in squares if board.piece_at(s)):
            vals.append(100)
        return vals

    a_vals, d_vals = _sorted(atk), _sorted(board.attackers(color, sq))
    gains, target, a, d = [], PVAL[ptype], 0, 0
    while a < len(a_vals):
        if a_vals[a] == 100 and d < len(d_vals):
            break
        gains.append(target)
        target = a_vals[a]; a += 1
        if target == 100 or d >= len(d_vals):
            break
        gains.append(target)
        target = d_vals[d]; d += 1
        if target == 100:
            break
    score = 0
    for g in reversed(gains):
        score = max(0, g - score)
    return score <= 0


def get_hanging_pieces(board: chess.Board, color: chess.Color) -> list:
    if board.king(color) is None:
        return []
    return [f"{chess.piece_name(p.piece_type).capitalize()} on {chess.square_name(sq)}"
            for sq, p in board.piece_map().items()
            if p.color == color and p.piece_type != chess.KING and not _see(board, sq, color, p.piece_type)]


def get_rook_files(board: chess.Board, color: chess.Color) -> tuple:
    open_files, semi_open_files = [], []
    rook_files = {chess.square_file(sq) for sq, p in board.piece_map().items() if p.color == color and p.piece_type == chess.ROOK}
    for f in rook_files:
        own = any(board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, color) for r in range(8))
        opp = any(board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, not color) for r in range(8))
        name = f"{chess.FILE_NAMES[f]}-file"
        if not own and not opp: open_files.append(name)
        elif not own and opp: semi_open_files.append(name)
    return open_files, semi_open_files


def get_move_fork(board: chess.Board, move: chess.Move, color: chess.Color) -> str:
    p = board.piece_at(move.to_square)
    if not p or p.color != color: return ""
    opp = "White" if color == chess.BLACK else "Black"
    pv = PVAL[p.piece_type]
    targets = []
    for sq in board.attacks(move.to_square):
        t = board.piece_at(sq)
        if not t or t.color == color: continue
        if t.piece_type == chess.KING:
            targets.append(f"{opp}'s King on {chess.square_name(sq)}")
        elif PVAL.get(t.piece_type, 0) >= pv or not board.is_attacked_by(not color, sq):
            targets.append(f"{opp}'s {chess.piece_name(t.piece_type)} on {chess.square_name(sq)}")
    if len(targets) >= 2:
        head = f"{chess.piece_name(p.piece_type).capitalize()} on {chess.square_name(move.to_square)}"
        return f"{head} attacks {', and '.join(targets)}"
    return ""


def score_val(s: chess.engine.PovScore | None) -> int:
    if s is None: return 0
    if s.is_mate():
        m = s.white().mate()
        return 0 if m is None else (20000 - m if m > 0 else -20000 - m)
    v = s.white().score()
    return v if v is not None else 0


def _null_move_threat(board: chess.Board, actor_color: str, tense: str, require_minor_pieces: bool = False) -> dict | None:
    if board is None or board.is_check() or board.is_game_over(): return None
    if require_minor_pieces and not any(p.piece_type not in (chess.PAWN, chess.KING) for p in board.piece_map().values()): return None

    temp = board.copy()
    try:
        temp.push(chess.Move.null())
    except Exception:
        return None

    engine = expert_eng.get()
    if engine is None: return None

    try:
        with expert_eng.lock:
            pv = engine.analyse(temp, chess.engine.Limit(time=0.10, depth=10)).get("pv", [])
        if not pv or not is_tactical_threat(temp, pv[0]): return None

        threat = pv[0]
        desc = f"{actor_color} {tense} to play {temp.san(threat)}"
        victim = board.piece_at(threat.to_square)
        if victim:
            vc = "White" if victim.color == chess.WHITE else "Black"
            desc += f" to capture {vc}'s {chess.piece_name(victim.piece_type)} on {chess.square_name(threat.to_square)}"
        return {"threat_move_uci": threat.uci(), "threat_description": desc}
    except Exception as e:
        logger.error("Threat analysis failed: %s", e)
        return None


def verify_threat_resolution(prev_board: chess.Board, player_move: chess.Move, prev_threat: dict) -> str:
    if not prev_threat or not (uci := prev_threat.get("threat_move_uci")): return ""
    try:
        threat = chess.Move.from_uci(uci)
    except ValueError:
        return ""
    after = prev_board.copy()
    after.push(player_move)
    target_sq = threat.to_square

    if player_move.to_square == threat.from_square and prev_board.piece_at(threat.from_square):
        return f"Captured attacking piece on {chess.square_name(threat.from_square)}."

    moved_piece = prev_board.piece_at(target_sq)
    if player_move.from_square == target_sq and moved_piece is not None:
        if _see(after, player_move.to_square, moved_piece.color, moved_piece.piece_type):
            return f"Moved threatened piece to safe square {chess.square_name(player_move.to_square)}."
        return "Moved threatened piece but it remains under attack."

    if threat not in after.legal_moves:
        return "Blocked or prevented threat."

    threatened = prev_board.piece_at(target_sq)
    if threatened is not None and after.is_attacked_by(prev_board.turn, target_sq):
        if _see(after, target_sq, threatened.color, threatened.piece_type):
            return f"Protected threatened piece on {chess.square_name(target_sq)}."
        return f"Reinforced threatened piece on {chess.square_name(target_sq)} but it remains hanging."

    return "Left threat active."


def perform_refutation_analysis(board: chess.Board) -> dict | None:
    if board.is_game_over(): return None
    engine = expert_eng.get()
    if engine is None: return None

    try:
        with expert_eng.lock:
            pv = engine.analyse(board, chess.engine.Limit(time=0.12, depth=11)).get("pv", [])
        if not pv: return None

        ref_san = board.san(pv[0])
        temp = board.copy()
        pv_san = []
        for m in pv[:4]:
            try:
                pv_san.append(temp.san(m))
                temp.push(m)
            except Exception: break

        def material(b):
            return sum(PVAL[p.piece_type] * (1 if p.color == chess.WHITE else -1) for p in b.piece_map().values())

        sign = 1 if board.turn == chess.WHITE else -1
        gain = sign * (material(temp) - material(board))

        if ref_san.endswith("#") or temp.is_checkmate():
            desc = f"opponent can play {ref_san} to deliver immediate checkmate"
        elif gain > 0:
            desc = f"opponent can play {ref_san} to gain a material advantage of {gain} point{'s' if gain > 1 else ''}"
        else:
            desc = f"opponent can play {ref_san} to gain a positional advantage"

        return {"refutation_description": desc, "refutation_pv": ", ".join(f"{i+1}. {m}" for i, m in enumerate(pv_san))}
    except Exception as e:
        logger.error("Refutation computation failed: %s", e)
        return None


_PERIOD_PHRASES = ("without repeating", "as requested", "per the provided", "per the instructions", "according to the rules")

def clean_meta_text(text: str) -> str:
    phrases = "|".join(_PERIOD_PHRASES)
    text = re.sub(rf"\b(?:{phrases})\b.*?\.(?=\s|$)", ".", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwithout repeating\b.*?(?=\s|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\busing the provided\b.*?\b(?:feature|line|text|box)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+\.", ".", re.sub(r"\s+", " ", text))
    return re.sub(r"\.{2,}", ".", text).strip()


def _colors(board, prev_board):
    if board:
        white_to_move = board.turn == chess.WHITE
        return ("Black" if white_to_move else "White", "White" if white_to_move else "Black")
    if prev_board:
        white_to_move = prev_board.turn == chess.WHITE
        return ("White" if white_to_move else "Black", "Black" if white_to_move else "White")
    return "Unknown", "Unknown"


def _game_phase(board):
    if not board: return "Middlegame"
    minors = [p for p in board.piece_map().values() if p.piece_type not in (chess.PAWN, chess.KING)]
    if len(minors) <= 4 or board.fullmove_number >= 40: return "Endgame"
    return "Opening" if board.fullmove_number <= 12 else "Middlegame"


def _eval_scores(prev_board, board):
    engine = expert_eng.get()
    if engine is None: return None, None
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
    if ev is None: return "0.0"
    try:
        if isinstance(ev, str) and "M" in ev: return f"Forced mate ({ev})"
        return f"{float(ev) / 100:+.2f}"
    except (ValueError, TypeError): return str(ev)


def _king_shield_weakened(prev_board, player_move, player_color) -> bool:
    if not player_move: return False
    try:
        piece = prev_board.piece_at(player_move.from_square)
        if not piece or piece.piece_type != chess.PAWN: return False
        player_col = chess.WHITE if player_color == "White" else chess.BLACK
        king_sq = prev_board.king(player_col)
        if king_sq is None: return False
        ff, fr = chess.square_file(player_move.from_square), chess.square_rank(player_move.from_square)
        kf = chess.square_file(king_sq)
        target_rank = 1 if player_col == chess.WHITE else 6
        return fr == target_rank and abs(ff - kf) <= 1
    except Exception:
        return False


def _move_purpose(prev_board, player_move, game_phase, opening_name, king_shield) -> str:
    if not prev_board or not player_move: return ""
    try:
        piece = prev_board.piece_at(player_move.from_square)
        if not piece: return ""
        p_name = chess.piece_name(piece.piece_type).capitalize()
        to_name = chess.square_name(player_move.to_square)
        is_en_passant = prev_board.is_en_passant(player_move)
        cap_tgt = prev_board.piece_at(player_move.to_square) if (prev_board.is_capture(player_move) and not is_en_passant) else None

        if king_shield: return f"advance the {p_name} to {to_name} but weaken king safety"
        if prev_board.is_castling(player_move): return "secure king safety and activate the rook via castling"
        if piece.piece_type == chess.PAWN:
            if player_move.promotion:
                promo = chess.piece_name(player_move.promotion).capitalize()
                if cap_tgt: return f"capture the opponent's {chess.piece_name(cap_tgt.piece_type)} on {to_name} and promote to {promo}"
                return f"advance the pawn to {to_name} and promote to {promo}"
            if is_en_passant: return f"capture en passant on {to_name} to undermine the opponent's pawn structure"
            if cap_tgt: return f"capture the opponent's {chess.piece_name(cap_tgt.piece_type)} on {to_name}"
            if game_phase == "Opening":
                suffix = f" in the {opening_name}" if opening_name else ""
                return f"advance the pawn to {to_name} to claim space and contest the center{suffix}"
            if game_phase == "Endgame": return f"advance the pawn to {to_name} to threaten promotion"
            return f"advance the pawn to {to_name} to claim space"
        if cap_tgt: return f"capture the opponent's {chess.piece_name(cap_tgt.piece_type)} on {to_name} and improve piece activity"
        if piece.piece_type == chess.KING:
            if game_phase == "Endgame": return f"activate the King to {to_name} to support pawns or restrict the enemy king"
            return f"reposition the King to {to_name} for safety"
        if game_phase == "Opening": return f"develop the {p_name} to {to_name} to control key squares"
        if game_phase == "Endgame": return f"activate the {p_name} on {to_name} to support centralization"
        return f"position the {p_name} on {to_name} to improve coordinate activity"
    except Exception:
        return ""

_DEFAULT_PURPOSE = {"Opening": "fight for central control in the opening", "Endgame": "activate pieces in the endgame"}


def strip_empty(x):
    """Recursively strips None, empty strings, and empty collections from dicts/lists."""
    if isinstance(x, dict):
        return {k: v for k, v in ((k, strip_empty(v)) for k, v in x.items()) if v not in (None, "", [], {})}
    if isinstance(x, list):
        return [v for v in (strip_empty(v) for v in x) if v not in (None, "", [], {})]
    return x


def prepare_coach_context(data: dict) -> dict:
    prev_fen, current_fen = data.get("prev_fen"), data.get("fen")
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

    player_move = None
    if prev_board and move_uci:
        try: player_move = chess.Move.from_uci(move_uci)
        except Exception: player_move = None

    prev_score = curr_score = None
    if not is_checkmate:
        prev_score, curr_score = _eval_scores(prev_board, board)
    if prev_score is not None and curr_score is not None:
        delta = (curr_score - prev_score) if player_color == "White" else (prev_score - curr_score)
        if cls_label == "book" and delta < -80:
            cls_label = "inaccuracy" if delta >= -150 else "mistake" if delta >= -300 else "blunder"
    is_bad_move = cls_label in _BAD_CLASSES

    eval_str = _format_eval(ev)
    king_shield = _king_shield_weakened(prev_board, player_move, player_color) if not is_checkmate else False
    move_purpose = _move_purpose(prev_board, player_move, game_phase, opening_name, king_shield) or _DEFAULT_PURPOSE.get(game_phase, "adjust piece activity in the middlegame")

    prev_threat = curr_threat = refutation = None
    threat_resolution = ""
    if not is_checkmate:
        curr_threat = data.get("threat") or (_null_move_threat(board, player_color, "threatens", require_minor_pieces=True) if board else None)

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
            except Exception:
                prev_threat = None

        if prev_threat is None and prev_board:
            prev_threat = _null_move_threat(prev_board, opponent_color, "threatened")

        refutation = perform_refutation_analysis(board) if (board and is_bad_move) else None
        if prev_board and player_move and prev_threat:
            try: threat_resolution = verify_threat_resolution(prev_board, player_move, prev_threat)
            except Exception: pass

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

    wpp_info = data.get("worst_placed_piece")
    if not wpp_info and board:
        try: wpp_info = WorstPlacedPieceAnalyzer(board).get_wpp()
        except Exception as e: logger.error("Error calculating WPP: %s", e); wpp_info = None

    eval_context = f"(Matches standard theory despite engine eval of {eval_str}.)" if ev is not None and not is_bad_move and cls_label == "book" and not is_checkmate else ""
    is_forced_mate = not is_checkmate and ev is not None and isinstance(ev, str) and "M" in ev

    benefits = []
    if not is_checkmate:
        if fork_made: benefits.append(f"creating a fork ({fork_made})")
        if new_open_rooks: benefits.append(f"activating a rook on open file ({', '.join(new_open_rooks)})")
        elif new_semi_rooks: benefits.append(f"activating a rook on semi-open file ({', '.join(new_semi_rooks)})")

    # Structured JSON features for LLM clarity
    features = {
        "context": {
            "player_color": player_color,
            "game_phase": game_phase,
            "opening_name": opening_name
        },
        "move": {
            "san": move_san,
            "uci": move_uci,
            "classification": cls_label,
            "eval": eval_str,
            "is_bad_move": is_bad_move,
            "is_checkmate": is_checkmate,
            "is_forced_mate": is_forced_mate,
            "purpose": move_purpose
        },
        "tactics": {
            "refutation": refutation["refutation_description"] if refutation else None,
            "refutation_pv": refutation["refutation_pv"] if refutation else None,
            "threat_resolution": threat_resolution,
            "fork": fork_made if fork_made else None,
            "benefit": " and ".join(benefits) if benefits else None
        },
        "strategy": {
            "worst_placed_piece": wpp_info.get("wpp_name") if wpp_info else None,
            "maneuver_path": wpp_info.get("maneuver_path") if wpp_info else None,
            "rook_files": new_open_rooks + new_semi_rooks
        }
    }
    features_block = json.dumps(strip_empty(features), indent=2)

    return {
        "move_san": move_san, "best_move_san": best_move_san, "cls_label": cls_label, "is_bad_move": is_bad_move,
        "eval_str": eval_str, "eval_context": eval_context, "features_block": features_block,
        "player_color": player_color, "opening_name": opening_name, "game_phase": game_phase,
        "is_checkmate": is_checkmate, "is_forced_mate": is_forced_mate, "move_purpose": move_purpose,
        "benefit_detail": " and ".join(benefits), "eval_desc": get_eval_description(ev, player_color),
        "refutation_pv": refutation["refutation_pv"] if refutation else "",
    }


class PositionalAggregator:
    def __init__(self, board: chess.Board):
        self.board = board

    def _get_analyses(self):
        if not hasattr(self, '_cache'):
            self._cache = {
                "space": self.analyze_space(),
                "king_safety": self.analyze_king_safety(),
                "structure": self.analyze_pawn_structure()
            }
        return self._cache

    def analyze_space(self):
        white_pts = black_pts = 0.0
        highlights = []
        for sq in chess.SQUARES:
            rank, file = chess.square_rank(sq), chess.square_file(sq)
            sq_name = chess.square_name(sq)
            weight = 1.5 if file in (2,3,4,5) and rank in (2,3,4,5) else 1.2 if file in (2,3,4,5) else 1.0

            w_atk = self.board.attackers(chess.WHITE, sq)
            b_atk = self.board.attackers(chess.BLACK, sq)

            if w_atk or b_atk:
                min_w = min((PVAL[ptype] for s in w_atk if (ptype := self.board.piece_type_at(s)) is not None), default=999)
                min_b = min((PVAL[ptype] for s in b_atk if (ptype := self.board.piece_type_at(s)) is not None), default=999)

                for color, atk, opp_atk, min_c, min_o, rank_cond, color_name, brush in [
                    (chess.WHITE, w_atk, b_atk, min_w, min_b, rank >= 4, "White", "green"),
                    (chess.BLACK, b_atk, w_atk, min_b, min_w, rank <= 3, "Black", "red")
                ]:
                    control = reason = None
                    if atk and not opp_atk and rank_cond:
                        control = True; reason = f"{color_name} controls {sq_name} exclusively (weight: {weight})"
                    elif atk and opp_atk:
                        if min_c < min_o and rank_cond:
                            control = True; reason = f"{color_name} controls {sq_name} via cheaper attacker (weight: {weight})"
                        elif min_c == min_o and len(atk) > len(opp_atk) and rank_cond:
                            control = True; reason = f"{color_name} controls {sq_name} via attacker count dominance (weight: {weight})"

                    if control:
                        if color == chess.WHITE: white_pts += 1.0 * weight
                        else: black_pts += 1.0 * weight
                        highlights.append({"orig": sq_name, "brush": brush, "reason": reason})

            piece = self.board.piece_at(sq)
            if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                is_defended_by_pawn = any(self.board.piece_type_at(s) == chess.PAWN for s in self.board.attackers(piece.color, sq))
                no_opp_pawn = not _pawn_can_attack(self.board, sq, not piece.color)
                if is_defended_by_pawn and no_opp_pawn:
                    color_name = "White" if piece.color == chess.WHITE else "Black"
                    brush = "green" if piece.color == chess.WHITE else "red"
                    if piece.color == chess.WHITE and rank >= 4:
                        white_pts += 2.0
                        highlights.append({"orig": sq_name, "brush": brush, "reason": f"White active outpost: {piece.unicode_symbol()} on {sq_name} (defended by pawn)"})
                    elif piece.color == chess.BLACK and rank <= 3:
                        black_pts += 2.0
                        highlights.append({"orig": sq_name, "brush": brush, "reason": f"Black active outpost: {piece.unicode_symbol()} on {sq_name} (defended by pawn)"})

        total = white_pts + black_pts
        score = (white_pts - black_pts) / total if total > 0 else 0.0
        return score, highlights

    def analyze_king_safety(self):
        highlights = []
        w_safe, b_safe = 100.0, 100.0

        for color in (chess.WHITE, chess.BLACK):
            king_sq = self.board.king(color)
            if king_sq is None: continue
            opp = not color
            color_name = "White" if color == chess.WHITE else "Black"
            brush = "green" if color == chess.WHITE else "red"
            opp_brush = "red" if color == chess.WHITE else "green"

            highlights.append({"orig": chess.square_name(king_sq), "brush": brush, "reason": f"{color_name} King position"})
            safety = 100.0
            kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)

            for df in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if df == 0 and dr == 0: continue
                    nf, nr = kf + df, kr + dr
                    if 0 <= nf <= 7 and 0 <= nr <= 7:
                        sq = chess.square(nf, nr)
                        atk = self.board.attackers(opp, sq)
                        if atk:
                            safety -= sum(8.0 if self.board.piece_type_at(s) == chess.QUEEN else 5.0 if self.board.piece_type_at(s) == chess.ROOK else 3.0 for s in atk if self.board.piece_at(s))
                            highlights.append({"orig": chess.square_name(sq), "brush": opp_brush, "reason": f"King Ring square {chess.square_name(sq)} under attack by {len(atk)} enemy piece(s)"})

            files = {max(0, kf-1), kf, min(7, kf+1)}
            for f in files:
                has_own = has_opp = False
                own_pawn_rank = None
                for r in range(8):
                    p = self.board.piece_at(chess.square(f, r))
                    if p and p.piece_type == chess.PAWN:
                        if p.color == color:
                            has_own = True
                            in_front = (r > kr) if color == chess.WHITE else (r < kr)
                            if in_front and (own_pawn_rank is None or
                                             (color == chess.WHITE and r < own_pawn_rank) or
                                             (color == chess.BLACK and r > own_pawn_rank)):
                                own_pawn_rank = r
                        else:
                            has_opp = True

                file_name = chr(97 + f)
                if not has_own:
                    file_desc = "Fully open file" if not has_opp else "Semi-open file"
                    safety -= 15.0 if not has_opp else 8.0
                    for r in range(max(0, kr-2), min(8, kr+3)):
                        highlights.append({"orig": chess.square_name(chess.square(f, r)), "brush": "blue", "reason": f"{file_desc} ({file_name}-file) adjacent to the King"})

                if own_pawn_rank is not None:
                    rel_rank = own_pawn_rank - kr if color == chess.WHITE else kr - own_pawn_rank
                    sq_name = chess.square_name(chess.square(f, own_pawn_rank))
                    if rel_rank == 2:
                        safety -= 8.0
                        highlights.append({"orig": sq_name, "brush": "yellow", "reason": f"Pushed pawn shield (hook): friendly pawn on {sq_name}"})
                    elif rel_rank >= 3:
                        safety -= 20.0
                        highlights.append({"orig": sq_name, "brush": "yellow", "reason": f"Highly pushed/compromised pawn shield: friendly pawn on {sq_name}"})

            safe_val = max(0.0, safety) / 100.0
            if color == chess.WHITE: w_safe = safe_val
            else: b_safe = safe_val

        return w_safe - b_safe, highlights

    def analyze_pawn_structure(self):
        highlights = []
        w_score, b_score = 0.0, 0.0

        for color in (chess.WHITE, chess.BLACK):
            opp = not color
            color_name = "White" if color == chess.WHITE else "Black"
            score = 0.0

            for sq in self.board.pieces(chess.PAWN, color):
                sq_name = chess.square_name(sq)
                f, r = chess.square_file(sq), chess.square_rank(sq)
                adj_files = [x for x in (f-1, f+1) if 0 <= x <= 7]

                is_iso = not any(self.board.piece_at(chess.square(af, ar)) == chess.Piece(chess.PAWN, color) for af in adj_files for ar in range(8))
                if is_iso:
                    score -= 0.20
                    highlights.append({"orig": sq_name, "brush": "yellow", "reason": f"{color_name} Isolated pawn on {sq_name}"})

                ranks_ahead = range(r+1, 8) if color == chess.WHITE else range(0, r)
                is_passed = not any(self.board.piece_at(chess.square(af, ar)) == chess.Piece(chess.PAWN, opp) for af in [f-1, f, f+1] if 0 <= af <= 7 for ar in ranks_ahead)
                if is_passed:
                    adv = r if color == chess.WHITE else 7 - r
                    score += 0.15 + adv * 0.05
                    highlights.append({"orig": sq_name, "brush": "green", "reason": f"{color_name} Passed pawn on {sq_name} (advanced to rank {r+1})"})

                ranks_doubled = range(r+1, 8) if color == chess.WHITE else range(0, r)
                if any(self.board.piece_at(chess.square(f, ar)) == chess.Piece(chess.PAWN, color) for ar in ranks_doubled):
                    score -= 0.15
                    highlights.append({"orig": sq_name, "brush": "yellow", "reason": f"{color_name} Doubled pawn on {sq_name}"})

                ranks_behind = range(0, r) if color == chess.WHITE else range(r+1, 8)
                has_behind = any(self.board.piece_at(chess.square(af, ar)) == chess.Piece(chess.PAWN, color) for af in adj_files for ar in ranks_behind)
                ranks_fwd = range(r+1, 8) if color == chess.WHITE else range(0, r)
                has_fwd = any(self.board.piece_at(chess.square(af, ar)) == chess.Piece(chess.PAWN, color) for af in adj_files for ar in ranks_fwd)

                if has_fwd and not has_behind:
                    front_r = r + (1 if color == chess.WHITE else -1)
                    if 0 <= front_r <= 7:
                        front_sq = chess.square(f, front_r)
                        if any(self.board.piece_type_at(a) == chess.PAWN and self.board.color_at(a) == opp for a in self.board.attackers(opp, front_sq)):
                            score -= 0.15
                            highlights.append({"orig": sq_name, "brush": "red", "reason": f"{color_name} Backward pawn on {sq_name}"})

            if color == chess.WHITE: w_score = score
            else: b_score = score

        score_diff = max(-1.0, min(1.0, w_score - b_score))
        return score_diff, highlights

    def get_positional_details(self) -> dict:
        c = self._get_analyses()
        return {"space": c["space"][1], "king_safety": c["king_safety"][1], "structure": c["structure"][1]}

    def get_positional_balance(self) -> dict:
        c = self._get_analyses()
        return {"space": round(c["space"][0], 2), "king_safety": round(c["king_safety"][0], 2), "structure": round(c["structure"][0], 2)}


class WorstPlacedPieceAnalyzer:
    def __init__(self, board: chess.Board):
        self.board = board
        self.turn = board.turn

    def _temp_move_piece(self, start, target, piece):
        original_at_target = self.board.piece_at(target)
        original_at_start = self.board.piece_at(start)
        self.board.remove_piece_at(start)
        self.board.set_piece_at(target, piece)
        def restore():
            if original_at_start: self.board.set_piece_at(start, original_at_start)
            else: self.board.remove_piece_at(start)
            if original_at_target: self.board.set_piece_at(target, original_at_target)
            else: self.board.remove_piece_at(target)
        return restore

    def _get_piece_destinations(self, sq, piece):
        """Static move generator for BFS pathing. Includes 1 and 2 square pushes for pawns."""
        if piece.piece_type == chess.PAWN:
            dests = set()
            direction = 8 if piece.color == chess.WHITE else -8
            start_rank = 1 if piece.color == chess.WHITE else 6
            
            # 1-square push
            push1 = sq + direction
            if 0 <= push1 <= 63 and not self.board.piece_at(push1):
                dests.add(push1)
                
                # 2-square push from starting rank
                if chess.square_rank(sq) == start_rank:
                    push2 = sq + (direction * 2)
                    if 0 <= push2 <= 63 and not self.board.piece_at(push2):
                        dests.add(push2)
                        
            # Captures
            for atk in self.board.attacks(sq):
                target = self.board.piece_at(atk)
                if target and target.color != piece.color:
                    dests.add(atk)
            return list(dests)
        else:
            return [sq2 for sq2 in self.board.attacks(sq)
                    if (p := self.board.piece_at(sq2)) is None or p.color != piece.color]

    def _is_piece_safe_on(self, sq, piece) -> bool:
        return _see(self.board, sq, piece.color, piece.piece_type)

    def get_piece_mobility_ratio(self, square: chess.Square) -> float:
        piece = self.board.piece_at(square)
        if not piece: return 1.0
        max_mobility = {chess.PAWN: 4, chess.KNIGHT: 8, chess.BISHOP: 13, chess.ROOK: 14, chess.QUEEN: 27}
        if piece.piece_type not in max_mobility: return 1.0

        actual_moves = len(self._get_piece_destinations(square, piece))
        if piece.piece_type == chess.BISHOP:
            own_pawns = self.board.pieces(chess.PAWN, piece.color)
            same_color = sum(1 for p_sq in own_pawns if (chess.square_file(p_sq) + chess.square_rank(p_sq)) % 2 == (chess.square_file(square) + chess.square_rank(square)) % 2)
            if same_color >= 5: actual_moves = max(0, actual_moves - 2)
        return actual_moves / max_mobility[piece.piece_type]

    def get_accessible_squares_at(self, curr_sq, start_sq, piece) -> list:
        """Returns safe squares reachable from curr_sq. Uses pseudo-legal checks to avoid turn-alternation hacks."""
        on_start = curr_sq == start_sq
        restore_outer = None if on_start else self._temp_move_piece(start_sq, curr_sq, piece)
        try:
            safe = []
            for dest in self._get_piece_destinations(curr_sq, piece):
                move = chess.Move(curr_sq, dest)
                # Check legality without pushing to avoid moving pinned pieces or exposing the king
                if move not in self.board.legal_moves:
                    continue
                r = self._temp_move_piece(curr_sq, dest, piece)
                if self._is_piece_safe_on(dest, piece):
                    safe.append(dest)
                r()
            return safe
        finally:
            if restore_outer:
                restore_outer()

    def is_target_good_square_at(self, square, start_sq, piece, current_eval) -> bool:
        """Evaluates if the target square is an ideal post for the specific piece type."""
        if square == start_sq: return False
        
        # 1. Evaluate strategic value first (fast Python logic)
        rank, file = chess.square_rank(square), chess.square_file(square)
        opp = not piece.color
        is_strategic = False
        
        if piece.piece_type == chess.PAWN:
            is_passed = not any(self.board.piece_at(chess.square(af, ar)) == chess.Piece(chess.PAWN, opp) for af in [file-1, file, file+1] if 0 <= af <= 7 for ar in (range(rank+1, 8) if piece.color == chess.WHITE else range(0, rank)))
            in_center = file in (2,3,4,5) and rank in (2,3,4,5)
            is_strategic = is_passed or in_center
            
        elif piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            in_opp = rank >= 4 if piece.color == chess.WHITE else rank <= 3
            in_center = rank in (2,3,4,5) and file in (2,3,4,5)
            if (in_opp or in_center) and not _pawn_can_attack(self.board, square, opp):
                is_strategic = True
            
        elif piece.piece_type == chess.ROOK:
            is_seventh = (piece.color == chess.WHITE and rank == 6) or (piece.color == chess.BLACK and rank == 1)
            file_pawns = [p for r in range(8) if (p := self.board.piece_at(chess.square(file, r))) and p.piece_type == chess.PAWN]
            is_open = not file_pawns
            is_semi = not any(p.color == piece.color for p in file_pawns) and not is_open
            is_strategic = is_seventh or is_open or is_semi
            
        elif piece.piece_type == chess.QUEEN:
            in_center = rank in (2,3,4,5) and file in (2,3,4,5)
            if in_center:
                is_strategic = True
            else:
                ratio = self.get_piece_mobility_ratio(square)
                is_strategic = ratio > 0.5
                
        if not is_strategic:
            return False

        # 2. Only if the square is strategically desirable, run the heavy engine and safety checks
        restore = self._temp_move_piece(start_sq, square, piece)
        is_safe = False
        if self._is_piece_safe_on(square, piece):
            is_safe = True
            engine = expert_eng.get()
            if engine:
                try:
                    with expert_eng.lock:
                        info = engine.analyse(self.board, chess.engine.Limit(time=0.03, depth=8))
                    score = info.get("score")
                    if score:
                        if score.is_mate():
                            m = score.relative.mate()
                            if m is not None and m < 0: is_safe = False
                        else:
                            val = score.relative.score()
                            if val is not None and val < current_eval - 100: is_safe = False
                except Exception as e:
                    logger.error("Stockfish safety check failed: %s", e)
        restore()
        return is_safe

    def find_maneuver_path(self, start_sq, piece, current_eval) -> list:
        queue = deque([(start_sq, [start_sq])])
        visited = {start_sq}
        max_depth = 4
        while queue:
            curr_sq, path = queue.popleft()
            if curr_sq != start_sq and self.is_target_good_square_at(curr_sq, start_sq, piece, current_eval):
                return [chess.square_name(sq) for sq in path]
            if len(path) - 1 >= max_depth: continue
            for next_sq in self.get_accessible_squares_at(curr_sq, start_sq, piece):
                if next_sq not in visited:
                    visited.add(next_sq)
                    queue.append((next_sq, path + [next_sq]))
        return []

    def get_wpp(self) -> dict:
        side_pieces = []
        is_opening = self.board.fullmove_number <= 12
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
                        current_eval = val if val is not None else 0
            except Exception as e:
                logger.error("Error getting current eval in get_wpp: %s", e)

        start_squares = {
            (chess.WHITE, chess.KNIGHT): {chess.B1, chess.G1}, (chess.WHITE, chess.BISHOP): {chess.C1, chess.F1},
            (chess.BLACK, chess.KNIGHT): {chess.B8, chess.G8}, (chess.BLACK, chess.BISHOP): {chess.C8, chess.F8},
            (chess.WHITE, chess.QUEEN): {chess.D1}, (chess.BLACK, chess.QUEEN): {chess.D8}
        }
        rook_start = {chess.WHITE: {chess.A1, chess.H1}, chess.BLACK: {chess.A8, chess.H8}}
        king_start = {chess.WHITE: chess.E1, chess.BLACK: chess.E8}

        for square, piece in self.board.piece_map().items():
            if piece.color != self.turn or piece.piece_type == chess.KING: continue
            if piece.piece_type == chess.ROOK and is_opening: continue

            ratio = self.get_piece_mobility_ratio(square)
            sort_ratio = ratio

            if piece.piece_type in (chess.KNIGHT, chess.BISHOP) and square in start_squares.get((piece.color, piece.piece_type), set()):
                sort_ratio += 0.50
            if piece.piece_type == chess.ROOK and square in rook_start[piece.color]:
                has_castling = self.board.has_kingside_castling_rights(piece.color) or self.board.has_queenside_castling_rights(piece.color)
                king_on_start = self.board.king(piece.color) == king_start[piece.color]
                if king_on_start or has_castling: sort_ratio += 0.50
            if piece.piece_type == chess.QUEEN and square in start_squares.get((piece.color, chess.QUEEN), set()):
                sort_ratio += 0.30

            side_pieces.append((square, piece, ratio, sort_ratio))

        if not side_pieces:
            return {"worst_pieces_list": [], "wpp_square": "", "wpp_name": "", "mobility_ratio": 1.0, "maneuver_path": []}

        side_pieces.sort(key=lambda x: (x[3], (x[0] * 17) % 64))
        worst_pieces_list = []
        for sq, piece, ratio, _ in side_pieces:
            name = f"{chess.piece_name(piece.piece_type).capitalize()} on {chess.square_name(sq)}"
            path = self.find_maneuver_path(sq, piece, current_eval)
            # Only include in list if a valid maneuver path exists
            if not path:
                continue
            worst_pieces_list.append({
                "wpp_square": chess.square_name(sq), "wpp_name": name,
                "mobility_ratio": round(ratio, 2), "maneuver_path": path
            })

        if not worst_pieces_list:
            return {"worst_pieces_list": [], "wpp_square": "", "wpp_name": "", "mobility_ratio": 1.0, "maneuver_path": []}

        primary = worst_pieces_list[0]
        return {
            "worst_pieces_list": worst_pieces_list, "wpp_square": primary["wpp_square"],
            "wpp_name": primary["wpp_name"], "mobility_ratio": primary["mobility_ratio"],
            "maneuver_path": primary["maneuver_path"]
        }