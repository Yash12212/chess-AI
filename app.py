# app.py
import os, re, json, math, shutil, sys, threading, traceback, logging, warnings
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string, Response
import chess
import chess.engine
import ollama

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = types = None

from expert_system import (
    prepare_coach_context,
    clean_meta_text,
    get_annotations,
    PositionalAggregator,
    get_hanging_pieces,
    _null_move_threat,
    WorstPlacedPieceAnalyzer
)

warnings.filterwarnings("ignore", message=".*null moves.*")

# ── Logging ───────────────────────────────────────────────────────────
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LEVEL_COLORS = {logging.DEBUG: "38;20", logging.INFO: "32;20", logging.WARNING: "33;20",
                logging.ERROR: "31;20", logging.CRITICAL: "31;1"}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = LEVEL_COLORS.get(record.levelno, "38;20")
        fmt = f"\x1b[{color}m{LOG_FMT}\x1b[0m"
        return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S").format(record)

root = logging.getLogger("chess_ai")
root.setLevel(logging.DEBUG)
root.propagate = False
_h = logging.StreamHandler()
_h.setLevel(logging.DEBUG)
_h.setFormatter(ColoredFormatter())
root.addHandler(_h)
logging.getLogger("chess.engine").setLevel(logging.ERROR)

logger = logging.getLogger("chess_ai.app")
if genai is None:
    logger.error("Failed to import google-genai library.")


def log_box(title, message, color="36"):
    b, r = f"\x1b[{color};1m", "\x1b[0m"
    lines = str(message).strip().split('\n')
    parts = [f"\n{b}┌── {title} " + "─" * max(0, 60 - len(title)) + r]
    parts += [f"{b}│{r} {ln}" for ln in lines]
    parts.append(f"{b}└" + "─" * 64 + r)
    return "\n".join(parts)


# ── Env & Config ──────────────────────────────────────────────────────
def load_env():
    p = Path(".env")
    if not p.exists():
        return
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip("'\"")
    except Exception as e:
        logger.error("Error loading .env file: %s", e)

load_env()

LLM_PROVIDER = (os.environ.get("LLM_PROVIDER") or "none").lower().strip()
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL")

SF_DEPTH, SF_TIME, SF_THREADS, SF_HASH = 20, 0.20, 2, 256
SF_MULTIPV_DEPTH, SF_MULTIPV_TIME = 14, 0.10
WIN_PROB_K = 0.00368208
EP_THRESH = [(0.02, "excellent"), (0.05, "good"), (0.10, "inaccuracy"), (0.20, "mistake")]
MISS_WIN_MIN, MISS_PLAY_MAX = 0.70, 0.55
PVAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
openings_db = {}


def die(msg):
    print(f"\nCONFIGURATION ERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def validate_config():
    if not Path(".env").exists() and not os.environ.get("LLM_PROVIDER"):
        die("'.env' file not found. Copy .env.example to .env and configure it.")
    provider = (os.environ.get("LLM_PROVIDER") or "").lower().strip()
    if provider in ("none", "disabled", "null", "false", ""):
        return
    if provider not in ("local", "google"):
        die(f"Invalid LLM_PROVIDER '{provider}'. Must be 'local', 'google', or 'none'.")
    if provider == "local" and not OLLAMA_MODEL:
        die("OLLAMA_MODEL is not set. Define it in your .env file (e.g. OLLAMA_MODEL=qwen3.5:0.8b).")
    if provider == "google":
        if not GEMINI_API_KEY:
            die("GEMINI_API_KEY is not set. Define it in your .env file.")
        if not GEMINI_MODEL:
            die("GEMINI_MODEL is not set. Define it in your .env file (e.g. GEMINI_MODEL=gemini-3.1-flash-lite).")

validate_config()


def sf_path():
    try:
        for p in Path(".").iterdir():
            if "stockfish" in p.name.lower() and p.is_file() and (os.name != 'nt' or p.suffix.lower() == '.exe'):
                return str(p)
    except Exception as e:
        logger.error("Error scanning local files: %s", e)
    candidates = ["stockfish", "./stockfish", "/usr/games/stockfish",
                  "/usr/bin/stockfish", "/opt/homebrew/bin/stockfish"]
    return next((p for p in candidates if shutil.which(p) or Path(p).exists()), "stockfish")


class EngineManager:
    def __init__(self):
        self._e = None
        self.lock = threading.Lock()

    def get(self):
        if not self._e:
            p = sf_path()
            logger.info("Starting Stockfish: %s", p)
            self._e = chess.engine.SimpleEngine.popen_uci(p)
            try:
                self._e.configure({"Threads": SF_THREADS, "Hash": SF_HASH})
            except Exception as e:
                logger.error("Config failed: %s", e)
        return self._e

    def close(self):
        if self._e:
            try: self._e.quit()
            except: pass

eng = EngineManager()


# ── Openings DB ───────────────────────────────────────────────────────
def norm_fen(fen, n=4):
    parts = fen.split()
    return " ".join(parts[:n]) if len(parts) >= n else parts[0]


def load_openings():
    global openings_db
    p = Path("openings.json")
    if not p.exists():
        return
    try:
        for item in json.loads(p.read_text(encoding="utf-8")):
            name, fen = item.get("name"), item.get("fen")
            if name and fen:
                openings_db[norm_fen(fen, 4)] = name
                openings_db[norm_fen(fen, 1)] = name
        logger.info("Loaded %d opening entries.", len(openings_db))
    except Exception as e:
        logger.error("Error loading openings.json: %s", e)


# ── Eval Helpers ──────────────────────────────────────────────────────
def win_prob(v):
    try:
        e = -WIN_PROB_K * v
        if e > 700:  return 0.0
        if e < -700: return 1.0
        return 1.0 / (1.0 + math.exp(e))
    except OverflowError:
        return 0.0 if v < 0 else 1.0


def score_val(s):
    if s.is_mate():
        m = s.white().mate()
        return 0 if m is None else (20000 - m if m > 0 else -20000 - m)
    return v if (v := s.white().score()) is not None else 0


def parse_eval(v):
    if isinstance(v, str) and v.startswith("M"):
        m = int(re.sub(r"[M+-]", "", v) or 0)
        return -20000 + m if "-" in v else 20000 - m
    return int(v)


def analyze(board, multipv=1, t=SF_TIME):
    if board.is_game_over():
        return []
    with eng.lock:
        r = eng.get().analyse(board, chess.engine.Limit(time=t, depth=SF_DEPTH), multipv=multipv)
        r = r if isinstance(r, list) else [r]
    return [{"move": i.get("pv", [None])[0], "score": score_val(i["score"])}
            for i in r if i.get("pv") and i.get("score")]


def eval_score(board):
    if board.is_game_over():
        o = board.outcome()
        return 0 if o.winner is None else ("M+0" if o.winner == chess.WHITE else "M-0")
    with eng.lock:
        s = eng.get().analyse(board, chess.engine.Limit(time=SF_TIME, depth=SF_DEPTH)).get("score")
    if not s:
        return 0
    if s.is_mate():
        m = s.white().mate()
        return 0 if m is None else f"M{'+' if m > 0 else ''}{m}"
    return v if (v := s.white().score()) is not None else 0


def _get_safe_cp_value(pov_score: chess.engine.PovScore) -> float:
    """Safely extracts raw centipawn float scores or maps mate-in-N to a ceiling of +-3000.0 centipawns."""
    if pov_score.is_mate():
        m = pov_score.relative.mate()
        mate_moves = max(1, abs(m)) if m else 1
        sign = 1 if m and m > 0 else -1
        return sign * (3000.0 - (mate_moves * 10))
    val = pov_score.relative.score()
    return float(val if val is not None else 0.0)


def calculate_stdev(scores):
    """Computes population standard deviation on raw centipawns."""
    n = len(scores)
    if n <= 1:
        return 0.0
    mean = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
    return math.sqrt(variance)


def evaluate_and_calculate_pci(board: chess.Board):
    """Evaluates position and returns (eval_score, pci_score, pci_tier, threat_dict)."""
    if board.is_game_over():
        o = board.outcome()
        winner = o.winner if o is not None else None
        ev = 0 if winner is None else ("M+0" if winner == chess.WHITE else "M-0")
        return ev, 0, "Quiet", None

    opp_color = "Black" if board.turn == chess.WHITE else "White"
    threat = _null_move_threat(board, opp_color, "threatens", require_minor_pieces=True)

    check_score = 20.0 if board.is_check() else 0.0
    threat_score = 20.0 if threat is not None else 0.0
    
    hanging_w = len(get_hanging_pieces(board, chess.WHITE))
    hanging_b = len(get_hanging_pieces(board, chess.BLACK))
    hanging_count = hanging_w + hanging_b
    hanging_score = min(hanging_count * 10.0, 30.0)
    
    tactical_density = min(check_score + threat_score + hanging_score, 50.0)

    with eng.lock:
        r = eng.get().analyse(board, chess.engine.Limit(time=SF_MULTIPV_TIME, depth=SF_MULTIPV_DEPTH), multipv=3)
        r = r if isinstance(r, list) else [r]

    if not r:
        return 0, 0, "Quiet", threat

    primary_score = r[0].get("score")
    if not primary_score:
        ev = 0
    elif primary_score.is_mate():
        m = primary_score.white().mate()
        ev = 0 if m is None else f"M{'+' if m > 0 else ''}{m}"
    else:
        ev = v if (v := primary_score.white().score()) is not None else 0

    scores = []
    has_mate = False
    for entry in r:
        score_obj = entry.get("score")
        if not score_obj:
            continue
        if score_obj.is_mate():
            has_mate = True
        scores.append(_get_safe_cp_value(score_obj))

    if has_mate:
        urgency_score = 50.0
    elif len(scores) < 2:
        urgency_score = 0.0
    else:
        stdev = calculate_stdev(scores)
        urgency_score = min((stdev / 100.0) * 50.0, 50.0)

    pci_score = round(urgency_score + tactical_density)
    pci_score = max(0, min(100, pci_score))

    if pci_score <= 25:
        pci_tier = "Quiet"
    elif pci_score <= 50:
        pci_tier = "Strategic"
    elif pci_score <= 75:
        pci_tier = "Sharp"
    else:
        pci_tier = "Volatile"

    return ev, pci_score, pci_tier, threat


def lpdo_circles(board):
    """LPDO tracker: red circles on loose pieces (undefended, or attacked by a cheaper piece,
    or with attacker count dominance) of the side to move."""
    color, opp = board.turn, not board.turn
    out = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece or piece.color != color or piece.piece_type == chess.KING:
            continue
        attackers = board.attackers(opp, sq)
        if not attackers:
            continue
        defenders = board.attackers(color, sq)
        
        undefended = not defenders
        cheaper = any(
            PVAL.get(board.piece_at(a).piece_type, 0) < PVAL.get(piece.piece_type, 0)
            for a in attackers if board.piece_at(a)
        )
        attacker_dominance = len(attackers) > len(defenders)
        
        if undefended:
            out.append({"orig": chess.square_name(sq), "brush": "red", "reason": "Undefended piece under attack"})
        elif cheaper:
            out.append({"orig": chess.square_name(sq), "brush": "red", "reason": "Piece attacked by lower-value piece"})
        elif attacker_dominance:
            out.append({"orig": chess.square_name(sq), "brush": "red", "reason": f"Under-defended piece ({len(attackers)} attacker(s) vs {len(defenders)} defender(s))"})
    return out


# ── Flask App ─────────────────────────────────────────────────────────
app = Flask(__name__)


def _classify_response(cls, ev, best, acc, arrows, circles, pos_bal, opening=None, pci_score=0, pci_tier="Maneuvering", threat=None, positional_details=None, worst_placed_piece=None):
    return jsonify({"classification": cls, "opening_name": opening, "eval": ev,
                    "best_move": best, "move_accuracy": acc, "arrows": arrows, "circles": circles,
                    "positional_balance": pos_bal, "positional_details": positional_details,
                    "pci_score": pci_score, "pci_tier": pci_tier, "threat": threat,
                    "worst_placed_piece": worst_placed_piece})


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/evaluate", methods=["POST"])
def api_eval():
    if not (fen := request.json.get("fen")):
        return jsonify({"error": "Missing FEN"}), 400
    try:
        board = chess.Board(fen)
        agg = PositionalAggregator(board)
        pos_bal = agg.get_positional_balance()
        pos_det = agg.get_positional_details()
        ev, pci_score, pci_tier, threat = evaluate_and_calculate_pci(board)
        wpp_data = WorstPlacedPieceAnalyzer(board).get_wpp()
        return jsonify({
            "eval": ev,
            "circles": lpdo_circles(board),
            "positional_balance": pos_bal,
            "positional_details": pos_det,
            "pci_score": pci_score,
            "pci_tier": pci_tier,
            "threat": threat,
            "worst_placed_piece": wpp_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/classify", methods=["POST"])
def api_classify():
    d = request.json
    prev_fen, uci = d.get("prev_fen"), d.get("move_uci")
    if not prev_fen or not uci:
        return jsonify({"error": "Missing parameters"}), 400
    try:
        bb = chess.Board(prev_fen)
        mv = chess.Move.from_uci(uci)
        if mv not in bb.legal_moves:
            return jsonify({"error": "Illegal move"}), 400

        ba = bb.copy()
        ba.push(mv)
        ev, pci_score, pci_tier, threat = evaluate_and_calculate_pci(ba)
        agg = PositionalAggregator(ba)
        pos_bal = agg.get_positional_balance()
        pos_det = agg.get_positional_details()
        wpp_data = WorstPlacedPieceAnalyzer(ba).get_wpp()

        ann = get_annotations(prev_fen, uci)
        arrows = ann.get("arrows", [])
        circles = ann.get("circles", []) + lpdo_circles(ba)

        opening = openings_db.get(norm_fen(ba.fen(), 4)) or openings_db.get(norm_fen(ba.fen(), 1))
        if opening:
            return _classify_response("book", ev, uci, 100.0, arrows, circles, pos_bal, opening, pci_score=pci_score, pci_tier=pci_tier, threat=threat, positional_details=pos_det, worst_placed_piece=wpp_data)
        if bb.legal_moves.count() == 1:
            return _classify_response("forced", ev, uci, 100.0, arrows, circles, pos_bal, pci_score=pci_score, pci_tier=pci_tier, threat=threat, positional_details=pos_det, worst_placed_piece=wpp_data)

        ab = analyze(bb, multipv=2)
        if not ab:
            return jsonify({"error": "Engine failed"}), 500
        best = ab[0]["move"]
        if not isinstance(best, chess.Move):
            return jsonify({"error": "Engine failed to return valid move"}), 500
        sb = ab[0]["score"]
        if not isinstance(sb, int):
            return jsonify({"error": "Engine failed to return valid score"}), 500

        turn = bb.turn
        nps = parse_eval(ev)
        sa = nps if turn == chess.WHITE else -nps
        wp_b = win_prob(sb if turn == chess.WHITE else -sb)
        wp_a = win_prob(sa)
        xpl = max(0.0, wp_b - wp_a)

        acc = 100.0 if mv == best else max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * xpl * 100) - 3.1669))
        sb1 = ab[1]["score"] if len(ab) > 1 and isinstance(ab[1]["score"], int) else 0
        great = mv == best and len(ab) > 1 and ((sb - sb1) if turn == chess.WHITE else (sb1 - sb)) >= 150
        brilliant = _is_brilliant(bb, ba, mv, best, turn, sa)
        miss = mv != best and wp_b >= MISS_WIN_MIN and wp_a < MISS_PLAY_MAX

        if brilliant:                       cls = "brilliant"
        elif great:                         cls = "great"
        elif miss:                          cls = "miss"
        elif mv == best or xpl <= 0.0001:   cls = "best"
        else:                               cls = next((c for t, c in EP_THRESH if xpl <= t), "blunder")

        return _classify_response(cls, ev, best.uci(), acc, arrows, circles, pos_bal, pci_score=pci_score, pci_tier=pci_tier, threat=threat, positional_details=pos_det, worst_placed_piece=wpp_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _is_brilliant(bb, ba, mv, best, turn, sa):
    """A best move that sacrifices material into a winning tactical sequence."""
    if mv != best:
        return False
    pt = bb.piece_type_at(mv.from_square)
    if bb.is_castling(mv) or pt is None or pt == chess.KING:
        return False
    opp, to = not turn, mv.to_square
    if not ba.is_attacked_by(opp, to):
        return False
    atk_vals = [PVAL.get(ba.piece_at(s).piece_type, 1) for s in ba.attackers(opp, to) if ba.piece_at(s)]
    cap = bb.piece_at(to)
    cap_val = 1 if bb.is_en_passant(mv) else (PVAL.get(cap.piece_type, 0) if cap else 0)
    return (PVAL[pt] - cap_val > 0
            and (not ba.is_attacked_by(turn, to) or min(atk_vals, default=999) < PVAL[pt])
            and sa >= -150)


# ── Coach (LLM) ───────────────────────────────────────────────────────
SYS_INTRO = "You are an objective chess analyst. Output exactly two sentences."
SYS_OUTRO = "Output only the commentary."


def build_coach_prompts(ctx):
    p, m = ctx["player_color"], ctx["move_san"]
    cls, ev, ev_desc = ctx["cls_label"], ctx["eval_str"], ctx["eval_desc"]
    purpose, ref, best = ctx["move_purpose"], ctx["refutation_pv"], ctx["best_move_san"]
    feats = ctx["features_block"]

    def ev_phrase():
        raw = ev.replace("+", "").replace("-", "").strip()
        return "" if (raw and raw in ev_desc) else f" with an evaluation of {ev}"

    sys_first = sys_second = ""
    if ctx.get("is_checkmate"):
        sys_first = "State that checkmate has been delivered."
        sys_second = "Confirm that this concludes the game."
        instr = f"1. State that {p} delivered checkmate with {m}.\n2. Confirm that this concludes the game."
    elif ctx.get("is_forced_mate") and not ctx["is_bad_move"]:
        sys_first = "Explain the forced checkmate sequence setup."
        sys_second = "Note the forced checkmate valuation."
        instr = f"1. State that {p} played {m} to set up a forced mate.\n2. Note that the evaluation is a forced checkmate ({ev})."
    elif ctx["is_bad_move"]:
        sys_first = "Explain why the move is bad using the provided evaluation."
        sys_second = "State the refutation sequence exactly as provided."
        phrase = {"inaccuracy": "an inaccuracy", "mistake": "a mistake"}.get(cls, "a blunder")
        lead = f"1. Explain that {p} played {m}, which is {phrase} because it {ev_desc}{ev_phrase()}.\n"
        instr = lead + (f"2. Conclude by writing: The refutation is exactly: {ref}." if ref else f"2. State that '{best}' was the best alternative.")
    else:
        sys_first = "Explain the move's purpose."
        sys_second = "State the evaluation or immediate tactical benefit."
        if ctx["benefit_detail"]:
            instr = f"1. Explain that {p} played {m} to {purpose}.\n2. Describe the benefit of {m} using the Fork or Rook details from the provided JSON."
        else:
            instr = f"1. Explain that {p} played {m} to {purpose}.\n2. State that this move {ev_desc}{ev_phrase()}."

    system_prompt = f"{SYS_INTRO} First sentence: {sys_first} Second sentence: {sys_second} {SYS_OUTRO}"
    user_prompt = f"Data:\n{feats}\n{ctx['eval_context']}\n\nInstructions:\n{instr}\n\nRule: Write exactly two sentences. Never repeat the evaluation or its description. No meta-text."
    return system_prompt, user_prompt


def _stream(iterable, log_title):
    """Accumulate streamed chunks, yield them, and log the full response (or an error)."""
    full = ""
    try:
        for chunk in iterable:
            if chunk:
                full += chunk
                yield chunk
        logger.info(log_box(log_title, full, "32"))
    except Exception as e:
        logger.error("Generator failed: %s", e)
        yield f"\n[Coach Error: {e}]"


def stream_google(system_prompt, prompt):
    if not GEMINI_API_KEY or not GEMINI_MODEL:
        yield "[Coach Error: Gemini API Key or Model is not configured.]"; return
    if genai is None or types is None:
        yield "[Coach Error: google-genai library not installed.]"; return
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content_stream(
        model=GEMINI_MODEL, contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt,
                                           temperature=0.1, max_output_tokens=150))
    yield from _stream((chunk.text for chunk in resp if chunk.text), "GOOGLE AI STUDIO OUTPUT")


def stream_ollama(system_prompt, prompt):
    client = ollama.Client(host=OLLAMA_HOST)
    kwargs = dict(model=OLLAMA_MODEL or "",
                  messages=[{"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}],
                  options={"temperature": 0.1, "num_predict": 150})
    try:
        stream = client.chat(think=False, stream=True, **kwargs)
    except TypeError:                       # older ollama has no `think` kwarg
        stream = client.chat(stream=True, **kwargs)

    def chunks():
        for chunk in stream:
            msg = getattr(chunk, 'message', None) or chunk.get('message', {})
            yield getattr(msg, 'content', None) or msg.get('content', '')
    yield from _stream(chunks(), "LLM OUTPUT")


@app.route("/api/coach", methods=["POST"])
def api_coach():
    if LLM_PROVIDER in ("none", "disabled", "null", "false"):
        return Response(
            "AI Coach is currently disabled. Set LLM_PROVIDER to 'google' or 'local' in your .env file to enable it.",
            mimetype="text/plain")

    ctx = prepare_coach_context(request.json)
    system_prompt, prompt = build_coach_prompts(ctx)
    logger.debug(log_box(f"SYSTEM PROMPT SENT TO {LLM_PROVIDER.upper()}", system_prompt, "36"))
    logger.debug(log_box(f"USER PROMPT SENT TO {LLM_PROVIDER.upper()}", prompt, "33"))

    gen = stream_google(system_prompt, prompt) if LLM_PROVIDER == "google" else stream_ollama(system_prompt, prompt)
    return Response(gen, mimetype="text/plain")


# ── HTML / Frontend ───────────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="Chess Analysis Hub - Analyze your chess games, evaluate positions with Stockfish, track position types, and get AI coach insights.">
  <title>Chess Analysis Hub</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@lichess-org/chessground@10.1.1/assets/chessground.base.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@lichess-org/chessground@10.1.1/assets/chessground.brown.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@lichess-org/chessground@10.1.1/assets/chessground.cburnett.css">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700;800&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <style type="text/tailwindcss">
    @layer components {
      .card   { @apply bg-[#151519]/80 backdrop-blur-md border border-neutral-800/80 rounded-2xl shadow-xl transition-all duration-300; }
      .panel  { @apply bg-[#0c0c0e]/90 border border-neutral-900/90 rounded-xl shadow-inner; }
      .btn    { @apply flex items-center justify-center gap-1.5 bg-[#202226] border border-neutral-800/60 py-2.5 rounded-lg transition active:translate-y-0.5 text-neutral-300 hover:bg-[#282a2f] hover:text-white; }
      .btn-em { @apply bg-emerald-950/30 border-emerald-500/40 text-emerald-400 hover:bg-emerald-900/50 hover:text-emerald-300; }
      .btn-red{ @apply text-red-400 hover:bg-red-950/40 hover:text-red-300; }
      .input  { @apply w-full bg-[#111113] border border-neutral-800/60 text-neutral-100 text-xs px-3 py-2.5 rounded-lg focus:outline-none focus:ring-1 focus:ring-emerald-500/50 font-mono; }
      .kbd    { @apply bg-[#202226] px-1.5 py-0.5 rounded text-neutral-300 text-[10px] font-mono; }
    }
  </style>
  <style>
    * {
      font-family: 'Outfit', sans-serif;
    }
    .font-mono {
      font-family: 'JetBrains Mono', monospace !important;
    }
    ::-webkit-scrollbar{width:6px;height:6px}
    ::-webkit-scrollbar-track{background:#0c0c0e}
    ::-webkit-scrollbar-thumb{background:#2a2a2e;border-radius:99px}
    ::-webkit-scrollbar-thumb:hover{background:#3a3a3e}
    #board .last-move{background:var(--last-move-bg, rgba(255,255,255,0.12))!important}
  </style>
</head>
<body class="bg-[#0b0b0d] bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-[#141418] via-[#0b0b0d] to-[#060608] text-neutral-200 min-h-screen flex flex-col items-center p-4 sm:p-6 font-sans select-none antialiased justify-center">
  <div id="main-grid" class="w-full max-w-[1180px] mx-auto grid grid-cols-1 lg:grid-cols-[540px_1fr] gap-6 items-start justify-center">
    
    <!-- LEFT COLUMN: BOARD AND CONTROLS -->
    <div class="flex flex-col gap-4 items-center w-full max-w-[540px] lg:sticky lg:top-6 lg:self-start z-20">
      <header class="relative z-40 w-full bg-[#151519]/90 backdrop-blur-md border border-neutral-800/80 px-4 py-3 rounded-2xl shadow-md flex flex-col gap-2.5">
        <div class="flex items-center justify-between w-full">
          <div class="flex items-center gap-3">
            <div class="p-2 rounded-xl bg-emerald-950/40 border border-emerald-500/30 text-emerald-400"><i data-lucide="crown" class="w-5 h-5"></i></div>
            <h1 class="text-base sm:text-lg font-extrabold text-white tracking-tight">Chess AI</h1>
          </div>
          <div id="turn-indicator" class="px-3 py-1.5 rounded-xl bg-[#202226] border border-neutral-800 text-xs font-black uppercase tracking-wider text-neutral-300">
            White to move
          </div>
        </div>
        <div class="flex items-center justify-between w-full border-t border-neutral-800/40 pt-2.5">
          <div id="opening-badge" class="px-3 py-1.5 rounded-xl bg-[#1c1c21] border border-neutral-800 text-xs font-semibold text-neutral-300 flex items-center gap-1.5 truncate max-w-[240px] sm:max-w-[280px]" title="Opening">
            <i data-lucide="book-open" class="w-3.5 h-3.5 text-emerald-400 shrink-0"></i>
            <span id="opening-name" class="truncate">Starting Position</span>
          </div>
          <div class="flex items-center gap-2">
            <button id="mute-btn" onclick="toggleMute()" class="p-2 py-1.5 rounded-xl bg-[#202226] border border-neutral-800 hover:bg-[#282a2f] text-neutral-300 hover:text-white transition-all select-none flex items-center justify-center hover:scale-105 w-9 h-9" title="Mute sounds">
              <i id="mute-icon" data-lucide="volume-2" class="w-4 h-4 text-emerald-400"></i>
            </button>
            <button id="toggle-analysis-btn" onclick="toggleAnalysisColumn()" class="px-3 py-1.5 rounded-xl bg-[#202226] border border-neutral-800 hover:bg-[#282a2f] text-neutral-300 hover:text-white transition-all select-none flex items-center gap-1.5 hover:scale-105" title="Toggle metrics panel">
              <i data-lucide="bar-chart-2" class="w-4 h-4 text-emerald-400"></i>
              <span>Metrics</span>
            </button>
          </div>
        </div>
      </header>
      
      <div class="flex gap-3 sm:gap-5 items-stretch justify-center w-full z-10">
        <div class="relative w-8 sm:w-10 shrink-0">
          <div class="relative w-full h-full bg-black border border-neutral-800 rounded-lg overflow-hidden flex flex-col shadow-[inset_0_2px_6px_rgba(0,0,0,0.7)]">
            <div id="eval-bar-black" class="w-full bg-gradient-to-b from-[#3a3a3a] via-[#1a1a1a] to-[#050505] transition-[height] duration-500 ease-out" style="height:50%"></div>
            <div class="w-full bg-gradient-to-b from-neutral-100 via-neutral-200 to-neutral-300 flex-1"></div>
            <div id="eval-bar-text-container" class="absolute left-1/2 transition-all duration-500 ease-out pointer-events-none z-10" style="top:50%">
              <span id="eval-bar-text" class="block text-[10px] sm:text-xs font-black font-mono leading-none whitespace-nowrap transition-opacity duration-300">0.0</span>
            </div>
          </div>
        </div>
        <div class="relative flex-1 min-w-0 aspect-square">
          <div id="external-ranks" class="absolute -left-3 sm:-left-4 top-0 bottom-0 flex flex-col justify-between text-center text-[9px] sm:text-[11px] font-black text-neutral-500 w-3 py-[6.25%] z-10 opacity-80"></div>
          <div id="external-files" class="absolute -bottom-3 sm:-bottom-4 left-0 right-0 flex justify-between text-center text-[9px] sm:text-[11px] font-black text-neutral-500 h-3 px-[6.25%] z-10 opacity-80"></div>
          <div id="board" class="chessground w-full h-full relative rounded-xl shadow-[0_8px_30px_rgba(0,0,0,0.5)] overflow-hidden border-2 border-neutral-800"></div>
        </div>
      </div>
      
      <div class="flex justify-between items-center gap-2 w-full p-2 card shadow-md">
        <div class="flex gap-2 flex-1">
          <button onclick="navigate('start')" class="btn flex-1 hover:bg-emerald-950/20 hover:text-emerald-400 group"><i data-lucide="chevrons-left" class="w-5 h-5 transition group-hover:-translate-x-1 duration-200"></i></button>
          <button onclick="navigate('back')" class="btn flex-1 hover:bg-emerald-950/20 hover:text-emerald-400 group"><i data-lucide="chevron-left" class="w-5 h-5 transition group-hover:-translate-x-0.5 duration-200"></i></button>
          <button onclick="navigate('forward')" class="btn flex-1 hover:bg-emerald-950/20 hover:text-emerald-400 group"><i data-lucide="chevron-right" class="w-5 h-5 transition group-hover:translate-x-0.5 duration-200"></i></button>
          <button onclick="navigate('end')" class="btn flex-1 hover:bg-emerald-950/20 hover:text-emerald-400 group"><i data-lucide="chevrons-right" class="w-5 h-5 transition group-hover:translate-x-1 duration-200"></i></button>
        </div>
        <div class="flex gap-2 shrink-0 ml-2">
          <button onclick="toggleBoardOrientation()" class="btn btn-em px-2 sm:px-4 group"><i data-lucide="refresh-cw" class="w-5 h-5 transition group-hover:rotate-180 duration-300"></i></button>
          <button onclick="confirmReset()" class="btn btn-red px-2 sm:px-4 group"><i data-lucide="trash-2" class="w-5 h-5 transition group-hover:scale-110 duration-200"></i></button>
        </div>
      </div>
    </div>
    
    <!-- RIGHT COLUMN: MOVES, COACH & ANALYSIS -->
    <div class="w-full max-w-[540px] lg:max-w-[600px] lg:h-[691px] flex flex-col gap-4 lg:col-start-2 lg:row-start-1">
      <!-- MOVES & COACH CARD -->
      <div class="w-full card flex flex-col min-h-[350px] flex-1 overflow-hidden shadow-2xl">
        <div class="px-4 py-3 bg-[#0c0c0e] border-b border-neutral-800/80">
          <div class="flex bg-[#202226] p-1 rounded-xl border border-neutral-800 gap-1" id="tab-bar"></div>
        </div>
        <div id="game-tab" class="tab-content p-4 flex-1 flex flex-col overflow-hidden">
          <div class="flex-1 flex flex-col panel p-4 overflow-y-auto overscroll-contain" id="move-list"></div>
          
          <!-- LLM Coach Panel -->
          <div class="panel p-4 mt-3 flex flex-col h-[180px] bg-[#0e0e11]/40 border border-neutral-800/80 shrink-0 shadow-inner" id="coach-panel">
            <div class="flex items-center gap-2 mb-2 pb-2 border-b border-neutral-900 shrink-0">
              <i data-lucide="sparkles" class="w-4 h-4 text-emerald-400 animate-pulse"></i>
              <span class="text-xs font-extrabold uppercase tracking-wider text-neutral-300">AI Coach Insights</span>
              <div id="coach-loading" class="hidden ml-auto flex items-center gap-1 text-[10px] text-emerald-400 font-bold animate-pulse">
                <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-bounce"></span>
                <span>Streaming...</span>
              </div>
            </div>
            <div id="coach-content" class="flex-1 overflow-y-auto text-lg text-neutral-300 font-sans leading-relaxed select-text pr-1">
              Select a move to receive grandmaster level coaching and strategic tips from AI!
            </div>
          </div>
        </div>
        <div id="stats-tab" class="tab-content p-4 flex-1 flex flex-col overflow-hidden hidden">
          <div class="flex flex-col gap-3 flex-1 min-h-0 pr-1" id="stats-grid"></div>
        </div>
        <div id="io-tab" class="tab-content p-4 flex-1 flex-col justify-between overflow-hidden hidden">
          <div class="flex-1 flex flex-col gap-4 min-h-0">
            <div class="panel p-3.5 flex flex-col gap-3 shrink-0">
              <div class="flex items-center gap-2 text-emerald-400">
                <i data-lucide="hash" class="w-4 h-4"></i>
                <span class="text-xs font-bold uppercase tracking-wider text-neutral-300">FEN - Single Position</span>
              </div>
              <input type="text" id="fen-input" class="input" placeholder="FEN">
              <div class="grid grid-cols-2 gap-2">
                <button onclick="loadCustomFen()" class="btn btn-em text-xs font-bold"><i data-lucide="upload" class="w-3.5 h-3.5"></i> Load FEN</button>
                <button onclick="copyCurrentFen()" class="btn text-xs font-bold"><i data-lucide="copy" class="w-3.5 h-3.5"></i> Copy FEN</button>
              </div>
            </div>
            <div class="panel p-3.5 flex-1 flex flex-col gap-3 min-h-[160px]">
              <div class="flex items-center gap-2 text-emerald-400 shrink-0">
                <i data-lucide="file-text" class="w-4 h-4"></i>
                <span class="text-xs font-bold uppercase tracking-wider text-neutral-300">PGN - Game History</span>
              </div>
              <textarea id="pgn-input" class="input flex-1 resize-none" placeholder="Paste PGN here..."></textarea>
              <div class="grid grid-cols-2 gap-2 shrink-0">
                <button onclick="importPgn()" class="btn btn-em text-xs font-bold"><i data-lucide="download" class="w-3.5 h-3.5"></i> Import PGN</button>
                <button onclick="copyCurrentPgn()" class="btn text-xs font-bold"><i data-lucide="copy" class="w-3.5 h-3.5"></i> Copy PGN</button>
              </div>
            </div>
          </div>
          <details class="border-t border-neutral-800 pt-3 group mt-4 shrink-0">
            <summary class="text-xs font-bold text-neutral-400 uppercase tracking-wider flex items-center gap-1.5 cursor-pointer hover:text-white select-none">
              <i data-lucide="keyboard" class="w-4 h-4 text-emerald-400"></i> Keyboard Shortcuts
              <i data-lucide="chevron-down" class="w-3.5 h-3.5 ml-auto transition-transform group-open:rotate-180"></i>
            </summary>
            <div class="grid grid-cols-2 gap-x-4 gap-y-2 text-neutral-400 mt-3 border-t border-neutral-800/30 pt-3" id="kbd-help"></div>
          </details>
          <div id="io-status" class="mt-4 w-max max-w-full bg-emerald-500 text-[#0f0f11] text-[11px] font-extrabold px-3 py-1.5 rounded-full shadow-lg hidden text-center animate-pulse mx-auto"></div>
        </div>
      </div>
      
      <!-- COLLAPSIBLE ANALYSIS COLUMN -->
      <div id="analysis-col" class="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full shrink-0">
        <div class="w-full px-3.5 py-2.5 card shadow-md">
          <h3 class="text-[11px] font-black uppercase tracking-wider text-neutral-300 mb-2 flex items-center gap-2 select-none">
            <i data-lucide="bar-chart-2" class="w-3.5 h-3.5 text-emerald-400 shrink-0"></i>
            <span>Position Type</span>
            <span id="position-complexity-badge" class="hidden px-1.5 py-0.5 rounded bg-[#202226] border border-neutral-800 text-[8px] font-bold uppercase tracking-wider text-neutral-400 transition-all duration-300 shrink-0 whitespace-nowrap"></span>
          </h3>
          <div class="space-y-2 select-none">
            <div id="space-row" onclick="selectMetric('space')" class="flex items-center gap-2 py-1 px-2 rounded-xl border border-transparent hover:bg-neutral-800/35 hover:border-neutral-700/30 transition-all cursor-pointer group">
              <span class="text-[10px] font-bold uppercase tracking-wider text-neutral-400 group-hover:text-white transition-colors shrink-0 w-20">Space</span>
              <div class="relative h-1.5 flex-1 bg-[#0c0c0e] rounded-full overflow-hidden flex shadow-inner">
                <div class="w-1/2 flex justify-end bg-transparent border-r border-neutral-900/40">
                  <div id="space-bar-black" class="h-full bg-[#f43f5e] shadow-[0_0_8px_rgba(244,63,94,0.4)] transition-all duration-300" style="width: 0%"></div>
                </div>
                <div class="w-1/2 bg-transparent">
                  <div id="space-bar-white" class="h-full bg-[#10b981] shadow-[0_0_8px_rgba(16,185,129,0.4)] transition-all duration-300" style="width: 0%"></div>
                </div>
              </div>
              <span id="space-val-text" class="text-neutral-400 font-mono text-xs w-10 text-right shrink-0">0.00</span>
            </div>
            <div id="king_safety-row" onclick="selectMetric('king_safety')" class="flex items-center gap-2 py-1 px-2 rounded-xl border border-transparent hover:bg-neutral-800/35 hover:border-neutral-700/30 transition-all cursor-pointer group">
              <span class="text-[10px] font-bold uppercase tracking-wider text-neutral-400 group-hover:text-white transition-colors shrink-0 w-20">King Safety</span>
              <div class="relative h-1.5 flex-1 bg-[#0c0c0e] rounded-full overflow-hidden flex shadow-inner">
                <div class="w-1/2 flex justify-end bg-transparent border-r border-neutral-900/40">
                  <div id="king-safety-bar-black" class="h-full bg-[#f43f5e] shadow-[0_0_8px_rgba(244,63,94,0.4)] transition-all duration-300" style="width: 0%"></div>
                </div>
                <div class="w-1/2 bg-transparent">
                  <div id="king-safety-bar-white" class="h-full bg-[#10b981] shadow-[0_0_8px_rgba(16,185,129,0.4)] transition-all duration-300" style="width: 0%"></div>
                </div>
              </div>
              <span id="king-safety-val-text" class="text-neutral-400 font-mono text-xs w-10 text-right shrink-0">0.00</span>
            </div>
            <div id="structure-row" onclick="selectMetric('structure')" class="flex items-center gap-2 py-1 px-2 rounded-xl border border-transparent hover:bg-neutral-800/35 hover:border-neutral-700/30 transition-all cursor-pointer group">
              <span class="text-[10px] font-bold uppercase tracking-wider text-neutral-400 group-hover:text-white transition-colors shrink-0 w-20">Structure</span>
              <div class="relative h-1.5 flex-1 bg-[#0c0c0e] rounded-full overflow-hidden flex shadow-inner">
                <div class="w-1/2 flex justify-end bg-transparent border-r border-neutral-900/40">
                  <div id="structure-bar-black" class="h-full bg-[#f43f5e] shadow-[0_0_8px_rgba(244,63,94,0.4)] transition-all duration-300" style="width: 0%"></div>
                </div>
                <div class="w-1/2 bg-transparent">
                  <div id="structure-bar-white" class="h-full bg-[#10b981] shadow-[0_0_8px_rgba(16,185,129,0.4)] transition-all duration-300" style="width: 0%"></div>
                </div>
              </div>
              <span id="structure-val-text" class="text-neutral-400 font-mono text-xs w-10 text-right shrink-0">0.00</span>
            </div>
          </div>
        </div>
        
        <div class="w-full px-3.5 py-2.5 card shadow-md flex flex-col gap-2" id="wpp-card">
          <h3 class="text-[11px] font-black uppercase tracking-wider text-neutral-300 flex flex-wrap items-center justify-between gap-1 select-none">
            <span class="flex items-center gap-1.5">
              <i data-lucide="crosshair" class="w-3.5 h-3.5 text-blue-400"></i> Worst Pieces
            </span>
            <button id="toggle-all-maneuvers-btn" onclick="toggleAllManeuvers()" class="px-2 py-0.5 rounded-md bg-[#202226] border border-neutral-800/80 text-[9px] font-bold uppercase tracking-wider text-neutral-300 hover:bg-[#282a2f] hover:text-white transition-all flex items-center gap-1 shrink-0" title="Show maneuvering ideas">
              <i data-lucide="eye" class="w-3 h-3 text-blue-400"></i> Maneuvers
            </button>
          </h3>
          <div id="wpp-list" class="flex flex-col gap-1.5 h-[110px] overflow-y-auto pr-1">
            <div class="flex flex-col items-center justify-center h-full text-neutral-500 text-xs font-sans animate-pulse"><i data-lucide="loader-2" class="w-4 h-4 text-neutral-500 animate-spin mb-1"></i><span>Analyzing position...</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/lucide@0.454.0/dist/umd/lucide.min.js"></script>
  <script type="module">
    import { Chessground } from "https://cdn.jsdelivr.net/npm/@lichess-org/chessground@10.1.1/dist/chessground.min.js";

    const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    const CLS = {
      brilliant:  { label:"Brilliant",  bg:"bg-[#1baca6]", text:"!!" },
      great:      { label:"Great Move",  bg:"bg-[#5c8bb0]", text:"!"  },
      best:       { label:"Best Move",   bg:"bg-[#429443]", icon:"star", fill:true },
      excellent:  { label:"Excellent",   bg:"bg-[#73a342]", icon:"thumbs-up", fill:true },
      good:       { label:"Good",        bg:"bg-[#528aae]", icon:"check", fill:false },
      book:       { label:"Book",        bg:"bg-[#d09140]", icon:"book-open", fill:true },
      forced:     { label:"Forced",      bg:"bg-[#8cae8c]", icon:"custom-arrow", fill:true },
      inaccuracy: { label:"Inaccuracy",  bg:"bg-[#f4bf23]", text:"?!" },
      mistake:    { label:"Mistake",     bg:"bg-[#e58f2a]", text:"?"  },
      miss:       { label:"Miss",        bg:"bg-[#e53e3e]", icon:"x", fill:false },
      blunder:    { label:"Blunder",     bg:"bg-[#b33430]", text:"??" }
    };

    const CLS_COLORS = {
      brilliant:  "#1baca6", great:      "#5c8bb0", best:       "#429443",
      excellent:  "#73a342", good:       "#528aae", book:       "#d09140",
      forced:     "#8cae8c", inaccuracy: "#f4bf23", mistake:    "#e58f2a",
      miss:       "#e53e3e", blunder:    "#b33430"
    };

    const TABS = [{id:"game",label:"Moves",active:true},{id:"stats",label:"Stats"},{id:"io",label:"Import/Export"}];
    const SHORTCUTS = [["← / PgUp","Back"],["→ / PgDn","Forward"],["↑ / Home","Start"],["↓ / End","End"],
      ["F","Flip"],["R","Reset"],["M","Moves"],["S","Stats"],["I","Import/Export"],["C","Copy FEN"],["P","Copy PGN"]];

    let states, currentIndex, chess, ground;
    let showAllManeuvers = false;
    let analysisExpanded = true;
    let activeTab = 'game';
    const $ = id => document.getElementById(id);
    const $$ = sel => document.querySelectorAll(sel);

    const audio = {
      ctx: null,
      buffers: {},
      urls: {
        move: 'https://lichess1.org/assets/sound/standard/Move.mp3',
        capture: 'https://lichess1.org/assets/sound/standard/Capture.mp3',
        check: 'https://lichess1.org/assets/sound/standard/Check.mp3',
        gameover: 'https://lichess1.org/assets/sound/standard/GenericNotify.mp3'
      },
      fetchPromises: {},

      prefetch() {
        for (const [key, url] of Object.entries(this.urls)) {
          this.fetchPromises[key] = fetch(url)
            .then(r => {
              if (!r.ok) throw new Error(r.statusText);
              return r.arrayBuffer();
            })
            .catch(e => {
              console.warn(`Failed to prefetch audio for ${key}:`, e);
              return null;
            });
        }
      },

      init() {
        if (!this.ctx) {
          try {
            this.ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
            this.decodeAll();
          } catch (e) {
            console.error("Failed to initialize AudioContext:", e);
          }
        }
        if (this.ctx && this.ctx.state === 'suspended') {
          this.ctx.resume().catch(() => {});
        }
      },

      async decodeAll() {
        for (const [key, promise] of Object.entries(this.fetchPromises)) {
          try {
            const arrayBuffer = await promise;
            if (arrayBuffer && !this.buffers[key]) {
              this.buffers[key] = await this.ctx.decodeAudioData(arrayBuffer.slice(0));
            }
          } catch (e) {
            console.error(`Failed to decode audio for ${key}:`, e);
          }
        }
      },

      playBuffer(key) {
        this.init();
        const buffer = this.buffers[key];
        if (!buffer) {
          // If buffer is not loaded yet (or fetch failed), fall back to standard HTML5 Audio
          try {
            const tempAudio = new Audio(this.urls[key]);
            tempAudio.play().catch(() => {});
          } catch {}
          return;
        }
        try {
          const source = this.ctx.createBufferSource();
          source.buffer = buffer;
          source.connect(this.ctx.destination);
          source.start(0);
        } catch (e) {
          console.warn("Buffer play failed:", e);
        }
      },

      playMove() { this.playBuffer('move'); },
      playCapture() { this.playBuffer('capture'); },
      playCheck() { this.playBuffer('check'); },
      playGameOver() { this.playBuffer('gameover'); }
    };

    // Prefetch all audio files immediately on load
    audio.prefetch();

    // Auto-resume AudioContext on first user interaction for zero latency
    const unlockAudio = () => {
      audio.init();
      ['click', 'mousedown', 'keydown', 'touchstart'].forEach(e => {
        document.removeEventListener(e, unlockAudio);
      });
    };
    ['click', 'mousedown', 'keydown', 'touchstart'].forEach(e => {
      document.addEventListener(e, unlockAudio);
    });

    let isMuted = false;
    const toggleMute = () => {
      isMuted = !isMuted;
      const icon = $('mute-icon');
      const btn = $('mute-btn');
      if (icon && btn) {
        if (isMuted) {
          icon.setAttribute('data-lucide', 'volume-x');
          icon.className = 'w-4 h-4 text-neutral-500';
          btn.title = 'Unmute sounds';
        } else {
          icon.setAttribute('data-lucide', 'volume-2');
          icon.className = 'w-4 h-4 text-emerald-400';
          btn.title = 'Mute sounds';
        }
        lucide.createIcons();
      }
    };

    const playSoundForMove = (san) => {
      if (isMuted || !san) return;
      try {
        if (san.includes('#')) {
          audio.playGameOver();
        } else if (san.includes('+')) {
          audio.playCheck();
        } else if (san.includes('x')) {
          audio.playCapture();
        } else {
          audio.playMove();
        }
      } catch (e) {
        console.error("Audio error:", e);
      }
    };

    const apiCall = async (url, body) => {
      const r = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });
      if (!r.ok) throw new Error(r.statusText);
      return r;
    };

    let activeWppPaths = new Set();
    let showWppPath = false;

    const newState = (fen, opening, extra={}) => ({
      fen, san:null, color:null, moveNumber:1, classification:null, eval:null,
      opening_name:opening, move_uci:null, best_move:null, move_accuracy:null,
      insight:null, insightLoading:false, arrows:[], circles:[], positional_balance:null,
      positional_details:null, pci_score:null, pci_tier:null, threat:null, worst_placed_piece:null, ...extra
    });

    const reset = () => {
      states = [newState(START_FEN, "Starting Position", {
        positional_balance: { space: 0.0, king_safety: 0.0, structure: 0.0 },
        positional_details: { space: [], king_safety: [], structure: [] },
        pci_score: 0,
        pci_tier: "Quiet",
        threat: null,
        worst_placed_piece: null
      })];
      currentIndex = 0;
      showWppPath = false;
      showAllManeuvers = false;
      activeWppPaths.clear();
      const btn = $('toggle-all-maneuvers-btn');
      if (btn) {
        btn.className = 'px-2 py-0.5 rounded-md bg-[#202226] border border-neutral-800/80 text-[9px] font-bold uppercase tracking-wider text-neutral-300 hover:bg-[#282a2f] hover:text-white transition-all flex items-center gap-1 shrink-0';
        btn.innerHTML = `<i data-lucide="eye" class="w-3 h-3 text-blue-400"></i> Maneuvers`;
      }
      const dropdown = $('wpp-dropdown');
      if (dropdown) dropdown.classList.add('hidden');
    };

    const clipboard = async text => {
      if (navigator.clipboard) return navigator.clipboard.writeText(text);
      const el = Object.assign(document.createElement('textarea'), {value:text});
      document.body.appendChild(el); el.select(); document.execCommand('copy'); document.body.removeChild(el);
    };

    const badgeHtml = (cls, sz="w-8 h-8") => {
      const c = CLS[cls]; if (!c) return '';
      const base = `inline-flex items-center justify-center rounded-full shrink-0 select-none ${c.bg} text-white ${sz} shadow-sm relative overflow-hidden`;
      let fs = sz.includes("w-[26px]") ? "16px" : sz.includes("w-7") ? "17px" : "19px";

      if (c.text) {
        const style = c.text.length === 2 ? 'font-size: 0.9em; letter-spacing: -0.06em;' : 'font-size: 1.25em; letter-spacing: -0.02em;';
        return `<span class="${base}" title="${c.label}" style="font-size: ${fs};"><span class="${c.text.includes('!') || c.text.includes('?') ? 'font-black italic' : 'font-extrabold'} leading-none flex items-center justify-center w-full h-full text-center" style="font-family: system-ui, -apple-system, sans-serif; ${style} transform: translateY(-0.5px);">${c.text}</span></span>`;
      }

      if (cls === 'forced') {
        return `<span class="${base}" title="${c.label}"><svg viewBox="0 0 24 24" fill="currentColor" class="text-white" style="width: 72%; height: 72%;"><path d="M14 4.5L21.5 12L14 19.5V15H3V9H14V4.5Z" /></svg></span>`;
      }
      
      const bgCol = CLS_COLORS[cls] || "transparent";
      const fillAttr = c.fill 
        ? `fill="currentColor" stroke="${bgCol}" stroke-width="1.2" data-lucide-stroke="${bgCol}" data-lucide-stroke-width="1.2"` 
        : `fill="none" stroke="currentColor" stroke-width="2.5" data-lucide-stroke="currentColor" data-lucide-stroke-width="2.5"`;

      return `<span class="${base}" title="${c.label}"><i data-lucide="${c.icon}" ${fillAttr} class="flex items-center justify-center" style="width: 74%; height: 74%;"></i></span>`;
    };

    const getDests = c => {
      const d = new Map();
      [..."abcdefgh"].flatMap(f => [1,2,3,4,5,6,7,8].map(r => f+r)).forEach(s => {
        const m = c.moves({square:s, verbose:true});
        m.length && d.set(s, m.map(x => x.to));
      });
      return d;
    };

    const initBoard = () => {
      ground = Chessground($('board'), {
        fen: chess.fen(), orientation:'white', coordinates:false,
        movable:{ color:'white', free:false, dests:getDests(chess), events:{ after:onUserMove } },
        animation: {
          enabled: true,
          duration: 120 // Snappy transition speed
        }
      });
      renderCoords();
    };

    const renderCoords = () => {
      const o = ground.state.orientation === 'white';
      $('external-ranks').innerHTML = (o ? [8,7,6,5,4,3,2,1] : [1,2,3,4,5,6,7,8]).map(r=>`<span>${r}</span>`).join('');
      $('external-files').innerHTML = (o ? [..."ABCDEFGH"] : [..."HGFEDCBA"]).map(f=>`<span>${f}</span>`).join('');
    };

    const onUserMove = async (orig, dest) => {
      chess.load(states[currentIndex].fen);
      const p = chess.get(orig);
      let pr;
      if (p && p.type === 'p' && (dest[1] === '8' || dest[1] === '1')) {
        if (!(pr = await promoPicker(dest, p.color))) return updateBoard();
      }
      const mv = chess.move({from:orig, to:dest, ...(pr ? {promotion:pr} : {})});
      if (mv) {
        states = states.slice(0, currentIndex+1);
        states.push(newState(chess.fen(), null, {
          san:mv.san, color:mv.color, moveNumber:Math.floor((states.length-1)/2)+1,
          move_uci:mv.from+mv.to+(mv.promotion||'')
        }));
        currentIndex = states.length-1;
      }
      updateBoard();
    };

    const promoPicker = (dest, color) => new Promise(res => {
      const o = ground.state.orientation === 'white', file = dest.charCodeAt(0)-97, rank = parseInt(dest[1])-1;
      const col = o?file:7-file, row = o?7-rank:rank, top = (o&&dest[1]==='8')||(!o&&dest[1]==='1');
      const ov = Object.assign(document.createElement('div'), {className:'absolute inset-0 z-50 bg-black/40 backdrop-blur-sm', onclick: () => { ov.remove(); pn.remove(); res(); }});
      const pn = Object.assign(document.createElement('div'), {className:'absolute z-[60] flex flex-col w-[12.5%] border border-neutral-400/40 rounded-lg overflow-hidden shadow-2xl'});
      Object.assign(pn.style, {top:`${(top?row:row-3)*12.5}%`, left:`${col*12.5}%`});
      const n = {q:'queen',r:'rook',b:'bishop',n:'knight'};
      ['q','r','b','n'].forEach(p => {
        const bt = Object.assign(document.createElement('button'), {
          className:'relative w-full aspect-square bg-stone-100/95 hover:bg-emerald-100/90 flex items-center justify-center border-b border-stone-200 last:border-none',
          innerHTML:`<piece class="${color==='w'?'white':'black'} ${n[p]}" style="position:static!important;width:82%!important;height:82%!important;transform:none!important;display:block;background-size:contain;background-position:center;background-repeat:no-repeat;"></piece>`,
          onclick: e => { e.stopPropagation(); ov.remove(); pn.remove(); res(p); }
        });
        pn.appendChild(bt);
      });
      const b = $('board').querySelector('.cg-wrap') || $('board');
      b.append(ov, pn);
    });

    const syncInputs = () => {
      const cur = states[currentIndex]; $('fen-input').value = cur.fen;
      try {
        const tc = new Chess(states[0].fen);
        if (states[0].fen !== START_FEN) { tc.header('FEN', states[0].fen); tc.header('SetUp','1'); }
        for (let i=1; i<=currentIndex; i++) states[i].san && tc.move(states[i].san);
        $('pgn-input').value = tc.pgn() || "";
      } catch { $('pgn-input').value = ""; }
    };

    const setBoardClassification = cls => {
      const b = $('board'); if (!b) return;
      b.className = b.className.replace(/\bcls-\S+/g, '').trim();
      if (cls && !['loading','unknown'].includes(cls)) b.classList.add(`cls-${cls}`);
      
      const col = CLS_COLORS[cls];
      if (col) {
        const opacity = ['inaccuracy', 'mistake', 'miss'].includes(cls) ? '4d' : '59';
        b.style.setProperty('--last-move-bg', `${col}${opacity}`);
      } else {
        b.style.removeProperty('--last-move-bg');
      }
    };

    const updateBoard = () => {
      const cur = states[currentIndex];
      if (currentIndex > 0) {
        playSoundForMove(cur.san);
      }
      chess.load(cur.fen);
      const act = chess.turn()==='w'?'white':'black';
      const lm = cur.move_uci ? [cur.move_uci.slice(0,2), cur.move_uci.slice(2,4)] : undefined;
      ground.set({ fen:cur.fen, lastMove:lm, turnColor:act, movable:{color:act, dests:getDests(chess)} });
      syncInputs(); renderMoveList();
      if ($('opening-name')) $('opening-name').innerText = cur.opening_name || lastOpening(currentIndex) || "Analyzing...";
      
      const turnInd = $('turn-indicator');
      if (turnInd) {
        if (act === 'white') {
          turnInd.innerText = 'White to move';
          turnInd.className = 'px-3 py-1.5 rounded-xl bg-neutral-100 border border-neutral-300 text-neutral-900 text-xs font-black uppercase tracking-wider';
        } else {
          turnInd.innerText = 'Black to move';
          turnInd.className = 'px-3 py-1.5 rounded-xl bg-neutral-950 border border-neutral-700 text-neutral-300 text-xs font-black uppercase tracking-wider';
        }
      }

      const isCalculating = (cur.eval === null && (currentIndex > 0 || cur.evalLoading));
      updateEvalMeter(cur.eval, isCalculating);
      
      activeWppPaths.clear();
      setBoardClassification(cur.classification); updateMarkings(cur);
      
      displayCoachInsight(cur);
      updatePositionalBalanceUI(cur.positional_balance, cur.pci_score, cur.pci_tier);
      updateWppUI(cur.worst_placed_piece, (currentIndex > 0 && cur.worst_placed_piece === null));

      if (currentIndex > 0 && cur.classification === null) classifyMove(currentIndex);
      else if (cur.eval === null || cur.positional_balance === null) evalPosition(currentIndex);
      else if (currentIndex > 0 && cur.classification && !cur.insight && !cur.insightLoading) {
        fetchCoachInsight(currentIndex);
      }
    };

    const evalPosition = async i => {
      const s = states[i]; if (!s || s.evalLoading) return;
      s.evalLoading = true;
      try {
        const r = await apiCall('/api/evaluate', {fen:s.fen});
        const d = await r.json();
        s.eval = d.eval || "0";
        s.circles = d.circles || [];
        s.positional_balance = d.positional_balance;
        s.positional_details = d.positional_details;
        s.pci_score = d.pci_score;
        s.pci_tier = d.pci_tier;
        s.threat = d.threat;
        s.worst_placed_piece = d.worst_placed_piece;
        if (currentIndex===i) {
          updateEvalMeter(s.eval);
          updateMarkings(s);
          updatePositionalBalanceUI(s.positional_balance, s.pci_score, s.pci_tier);
          updateWppUI(s.worst_placed_piece, (currentIndex > 0 && s.worst_placed_piece === null));
        }
      } catch { if (currentIndex===i) { updateEvalMeter(0); updatePositionalBalanceUI(null, null, null); updateWppUI(null, false); } }
      s.evalLoading = false;
    };

    const classifyMove = async i => {
      const s = states[i]; if (!s || s.classification==='loading') return;
      s.classification = 'loading'; renderMoveList();
      try {
        const r = await apiCall('/api/classify', {prev_fen:states[i-1].fen, move_uci:s.move_uci});
        const d = await r.json();
        Object.assign(s, {classification:d.classification, eval:d.eval, best_move:d.best_move,
          move_accuracy:d.move_accuracy, opening_name:d.opening_name||lastOpening(i),
          arrows: d.arrows || [], circles: d.circles || [],
          positional_balance: d.positional_balance,
          positional_details: d.positional_details,
          pci_score: d.pci_score,
          pci_tier: d.pci_tier,
          threat: d.threat,
          worst_placed_piece: d.worst_placed_piece});
        if (currentIndex===i) {
          updateEvalMeter(s.eval);
          if ($('opening-name')) $('opening-name').innerText = s.opening_name;
          updateMarkings(s); setBoardClassification(s.classification);
          displayCoachInsight(s);
          fetchCoachInsight(i);
          updatePositionalBalanceUI(s.positional_balance, s.pci_score, s.pci_tier);
          updateWppUI(s.worst_placed_piece, (currentIndex > 0 && s.worst_placed_piece === null));
        }
      } catch { Object.assign(s, {classification:'unknown', eval:0, move_accuracy:100}); }
      renderMoveList();
    };

    const fetchCoachInsight = async i => {
      const s = states[i];
      if (!s || i === 0 || s.insight || s.insightLoading || !s.classification || s.classification === 'loading') return;

      s.insightLoading = true;
      if (currentIndex === i) {
        $('coach-loading').classList.remove('hidden');
        $('coach-content').innerHTML = `<p class="animate-pulse text-emerald-400">Grandmaster Coach is analyzing your move...</p>`;
      }

      try {
        let bestMoveSan = s.best_move || "";
        if (s.best_move) {
          try {
            const temp = new Chess(states[i-1].fen);
            const mv = temp.move({
              from: s.best_move.slice(0,2),
              to: s.best_move.slice(2,4),
              promotion: s.best_move[4]
            });
            if (mv) bestMoveSan = mv.san;
          } catch {}
        }

        const r = await apiCall('/api/coach', {
          fen: s.fen,
          prev_fen: states[i-1].fen,
          move_san: s.san,
          move_uci: s.move_uci,
          classification: s.classification,
          best_move_san: bestMoveSan,
          eval: s.eval,
          opening_name: s.opening_name,
          threat: s.threat,
          prev_threat_uci: states[i-1].threat ? states[i-1].threat.threat_move_uci : null,
          worst_placed_piece: s.worst_placed_piece
        });

        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        s.insight = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          s.insight += decoder.decode(value, { stream: true });
          if (currentIndex === i) {
            $('coach-content').innerHTML = `<div class="prose prose-invert text-lg leading-relaxed font-sans">${s.insight}</div>`;
          }
        }
      } catch {
        s.insight = `<span class="text-red-400 font-semibold">Coach Error:</span> Connection to model stream failed.`;
      } finally {
        s.insightLoading = false;
        if (currentIndex === i) {
          $('coach-loading').classList.add('hidden');
          displayCoachInsight(s);
        }
      }
    };

    const displayCoachInsight = s => {
      const el = $('coach-content');
      if (!el) return;
      if (!s || currentIndex === 0) {
        el.innerHTML = `<span class="text-neutral-400 italic">Select a move to receive grandmaster level coaching and strategic tips from AI!</span>`;
        return;
      }
      if (s.insightLoading && !s.insight) {
        el.innerHTML = `<p class="animate-pulse text-emerald-400">Grandmaster Coach is analyzing your move...</p>`;
        return;
      }
      if (s.insight) {
        el.innerHTML = `<div class="prose prose-invert text-lg leading-relaxed font-sans">${s.insight}</div>`;
      } else if (s.classification === 'loading') {
        el.innerHTML = `<p class="text-neutral-500 italic animate-pulse">Waiting for engine classification...</p>`;
      } else {
        el.innerHTML = `<p class="text-neutral-400 italic">No insights loaded yet. Thinking...</p>`;
        if (s.classification && s.classification !== 'unknown') {
          fetchCoachInsight(currentIndex);
        }
      }
      lucide.createIcons();
    };

    let selectedMetric = null;
    const selectMetric = name => {
      selectedMetric = selectedMetric === name ? null : name;
      ['space', 'king_safety', 'structure'].forEach(m => {
        const row = $(`${m}-row`);
        if (row) {
          if (m === selectedMetric) {
            row.classList.add('bg-neutral-800/40', 'border-neutral-700/50', 'ring-1', 'ring-emerald-500/20');
          } else {
            row.classList.remove('bg-neutral-800/40', 'border-neutral-700/50', 'ring-1', 'ring-emerald-500/20');
          }
        }
      });
      updateMarkings(states[currentIndex]);
    };

    const updateMarkings = s => {
      $$('.classification-badge-wrapper').forEach(e=>e.remove());
      if (!s) {
        window.currentBoardMarkings = [];
        return ground.setShapes([]);
      }
      const sh = [], to = s.move_uci?.slice(2,4);
      if (to && s.classification && !['loading','unknown'].includes(s.classification)) drawBadge(to, s.classification);
      
      if (s.best_move && s.move_uci && s.classification) {
        if (!['best','brilliant','great','forced','book'].includes(s.classification) && s.best_move !== s.move_uci) {
          let bestMoveSan = s.best_move;
          try {
            const prevState = states[currentIndex - 1];
            if (prevState) {
              const temp = new Chess(prevState.fen);
              const mv = temp.move({
                from: s.best_move.slice(0,2),
                to: s.best_move.slice(2,4),
                promotion: s.best_move[4]
              });
              if (mv) bestMoveSan = mv.san;
            }
          } catch (e) { console.error(e); }
          sh.push({ orig: s.best_move.slice(0,2), dest: s.best_move.slice(2,4), brush: 'green', modifiers: { lineWidth: 10 }, reason: `Best alternative move: ${bestMoveSan}` });
        }
      }

      if (s.arrows) {
        s.arrows.forEach(a => sh.push({ orig: a.orig, dest: a.dest, brush: a.brush, modifiers: { lineWidth: 8 }, reason: a.reason }));
      }
      if (s.circles) {
        s.circles.forEach(c => sh.push({ orig: c.orig, brush: c.brush, reason: c.reason }));
      }

      if (selectedMetric && s.positional_details && s.positional_details[selectedMetric]) {
        s.positional_details[selectedMetric].forEach(h => sh.push({ orig: h.orig, dest: h.dest, brush: h.brush, modifiers: h.modifiers || {}, reason: h.reason }));
      }

      if (s.worst_placed_piece && s.worst_placed_piece.worst_pieces_list) {
        s.worst_placed_piece.worst_pieces_list.forEach(wpp => {
          if ((showAllManeuvers || activeWppPaths.has(wpp.wpp_square)) && wpp.maneuver_path && wpp.maneuver_path.length >= 2) {
            const path = wpp.maneuver_path;
            for (let i = 0; i < path.length - 1; i++) {
              sh.push({ orig: path[i], dest: path[i+1], brush: 'blue', modifiers: { lineWidth: 8 }, reason: `Maneuver path for worst-placed piece: ${wpp.wpp_name} (${path[i]} -> ${path[i+1]})` });
            }
          }
        });
      } else if ((showAllManeuvers || showWppPath) && s.worst_placed_piece && s.worst_placed_piece.maneuver_path && s.worst_placed_piece.maneuver_path.length >= 2) {
        const path = s.worst_placed_piece.maneuver_path;
        for (let i = 0; i < path.length - 1; i++) {
          sh.push({ orig: path[i], dest: path[i+1], brush: 'blue', modifiers: { lineWidth: 8 }, reason: `Maneuver path for worst-placed piece: ${s.worst_placed_piece.wpp_name} (${path[i]} -> ${path[i+1]})` });
        }
      }

      window.currentBoardMarkings = sh;
      ground.setShapes(sh);
    };

    const drawBadge = (key, cls) => {
      const o = ground.state.orientation === 'white';
      const col = o ? key.charCodeAt(0)-97 : 7-(key.charCodeAt(0)-97), row = o ? 8-parseInt(key[1]) : parseInt(key[1])-1;
      const w = Object.assign(document.createElement('div'), {className:'classification-badge-wrapper absolute z-30'});
      const n = 2.0;
      Object.assign(w.style, {
        left: `${col === 7 ? col*12.5+n : (col+1)*12.5-n}%`,
        top: `${row === 7 ? row*12.5+n : (row+1)*12.5-n}%`,
        transform: 'translate(-50%,-50%)'
      });
      w.innerHTML = badgeHtml(cls, 'w-[26px] h-[26px] sm:w-[32px] sm:h-[32px] border-2 border-[#0f0f11] ring-1 ring-white/20 shadow-[0_4px_12px_rgba(0,0,0,0.8)] scale-100 hover:scale-110 transition-transform duration-200');
      $('board').appendChild(w); lucide.createIcons();
    };

    const lastOpening = i => {
      for (let j=i; j>=0; j--) if (states[j]?.opening_name && states[j].opening_name!=="Analyzing position...") return states[j].opening_name;
      return "Custom Position";
    };

    const fmtEval = s => s == null ? "..." : (typeof s === 'string' ? s : `${s >= 0 ? '+' : ''}${(s/100).toFixed(1)}`);

    const updateEvalMeter = (score, isCalculating = false) => {
      const bb = $('eval-bar-black'), tc = $('eval-bar-text-container'), tx = $('eval-bar-text');
      if (!bb || !tc || !tx) return;
      
      if (score == null) {
        if (isCalculating) { tx.style.opacity = '0.4'; return; }
        Object.assign(bb.style, {height:'50%'}); Object.assign(tc.style, {top:'50%', transform:'translate(-50%,6px)'});
        tx.innerText = '0.0'; tx.style.color = '#0a0a0a'; tx.style.opacity = '1';
        return;
      }
      
      tx.style.opacity = '1';
      const isMate = typeof score === 'string' && score.startsWith('M');
      const pct = isMate ? (score.includes('-') ? 0 : 100) : 100 / (1 + Math.exp(-0.32 * (parseFloat(score) / 100)));
      const bdy = 100 - pct;
      bb.style.height = `${bdy}%`; tc.style.top = `${bdy}%`;
      tx.innerText = isMate ? score : fmtEval(score);
      Object.assign(tc.style, {transform: pct >= 50 ? 'translate(-50%,6px)' : 'translate(-50%,calc(-100% - 6px))'});
      tx.style.color = pct >= 50 ? '#0a0a0a' : '#f5f5f5';
    };

    const updatePositionalBalanceUI = (balance, pciScore, pciTier) => {
      const badge = $('position-complexity-badge');
      if (badge) {
        if (pciScore === null || pciScore === undefined || pciTier === null || pciTier === undefined) {
          badge.classList.add('hidden');
        } else {
          badge.classList.remove('hidden');
          badge.innerText = `${pciTier} (${pciScore}%)`;
          const cls = 'px-2 py-0.5 rounded-full text-[10px] font-black uppercase tracking-wider whitespace-nowrap ';
          if (pciTier === "Quiet") {
            badge.className = cls + 'bg-neutral-900 border border-neutral-700 text-neutral-400';
          } else if (pciTier === "Strategic") {
            badge.className = cls + 'bg-emerald-950/40 border border-emerald-500/30 text-emerald-400';
          } else if (pciTier === "Sharp") {
            badge.className = cls + 'bg-amber-950/40 border border-amber-500/30 text-amber-400';
          } else {
            badge.className = cls + 'bg-red-950/40 border border-red-500/30 text-red-400';
          }
        }
      }

      if (!balance) {
        ['space', 'king_safety', 'structure'].forEach(metric => {
          const idName = metric.replace('_', '-');
          const txt = $(`${idName}-val-text`);
          if (txt) { txt.innerText = '0.00'; txt.className = 'text-neutral-400 font-mono'; }
          const wb = $(`${idName}-bar-white`);
          const bb = $(`${idName}-bar-black`);
          if (wb) wb.style.width = '0%';
          if (bb) bb.style.width = '0%';
        });
        return;
      }

      ['space', 'king_safety', 'structure'].forEach(metric => {
        const val = balance[metric] || 0;
        const idName = metric.replace('_', '-');
        const txt = $(`${idName}-val-text`);
        if (txt) {
          txt.innerText = val > 0 ? `+${val.toFixed(2)}` : val.toFixed(2);
          if (val > 0) txt.className = 'text-emerald-400 font-bold font-mono';
          else if (val < 0) txt.className = 'text-red-400 font-bold font-mono';
          else txt.className = 'text-neutral-400 font-mono';
        }
        const pct = Math.min(Math.abs(val) * 100, 100);
        const wb = $(`${idName}-bar-white`);
        const bb = $(`${idName}-bar-black`);
        if (wb && bb) {
          if (val >= 0) { wb.style.width = `${pct}%`; bb.style.width = '0%'; } 
          else { bb.style.width = `${pct}%`; wb.style.width = '0%'; }
        }
      });
    };

    const updateWppUI = (wpp, isLoading = false) => {
      const listEl = $('wpp-list');
      const cardEl = $('wpp-card');
      if (!listEl || !cardEl) return;
      
      if (wpp && wpp.worst_pieces_list && wpp.worst_pieces_list.length > 0) {
        cardEl.classList.remove('hidden');
        
        listEl.innerHTML = wpp.worst_pieces_list.map(item => {
          const pct = Math.round(item.mobility_ratio * 100);
          const isActive = showAllManeuvers || activeWppPaths.has(item.wpp_square);
          
          return `
            <div class="py-1.5 px-2.5 rounded-xl bg-[#1d1d21]/50 border ${isActive ? 'border-blue-500/40 bg-blue-950/15' : 'border-neutral-800/40'} hover:bg-neutral-800/40 transition-all cursor-pointer flex flex-col gap-0.5" onclick="toggleSingleWppPath('${item.wpp_square}')">
              <div class="flex items-center justify-between">
                <div class="flex items-center gap-1.5 font-sans">
                  <span class="w-2 h-2 rounded-full ${isActive ? 'bg-blue-400 animate-pulse' : 'bg-neutral-600'}"></span>
                  <span class="text-xs font-bold text-neutral-200">${item.wpp_name}</span>
                </div>
                <span class="text-[10px] px-1.5 py-0.5 rounded bg-[#131316] text-neutral-400 font-semibold">${pct}% active</span>
              </div>
              <div class="text-[10px] text-neutral-400 flex items-center gap-1 mt-0.5 font-mono">
                <span class="text-blue-400 font-semibold">Path:</span>
                <span>${item.maneuver_path.join(' ➔ ')}</span>
              </div>
            </div>
          `;
        }).join('');
      } else {
        cardEl.classList.remove('hidden');
        if (isLoading) {
          listEl.innerHTML = `<div class="flex flex-col items-center justify-center h-full text-neutral-500 text-xs font-sans animate-pulse"><i data-lucide="loader-2" class="w-4 h-4 text-neutral-500 animate-spin mb-1"></i><span>Analyzing position...</span></div>`;
        } else {
          listEl.innerHTML = `<div class="flex flex-col items-center justify-center h-full text-neutral-500 text-xs font-sans text-center px-2">No worst-placed pieces detected. Position is balanced.</div>`;
        }
        lucide.createIcons();
      }
    };

    const toggleWppDropdown = () => {};

    const toggleSingleWppPath = sq => {
      if (activeWppPaths.has(sq)) activeWppPaths.delete(sq);
      else activeWppPaths.add(sq);
      
      if (states[currentIndex]) {
        updateWppUI(states[currentIndex].worst_placed_piece);
        updateMarkings(states[currentIndex]);
      }
    };

    const toggleAllManeuvers = () => {
      showAllManeuvers = !showAllManeuvers;
      const btn = $('toggle-all-maneuvers-btn');
      if (btn) {
        if (showAllManeuvers) {
          btn.className = 'px-2 py-0.5 rounded-md bg-blue-950/40 text-blue-400 border border-blue-500/30 text-[9px] font-bold uppercase tracking-wider hover:bg-blue-900/40 transition-all flex items-center gap-1 shrink-0';
          btn.innerHTML = `<i data-lucide="eye-off" class="w-3 h-3"></i> Maneuvers`;
        } else {
          btn.className = 'px-2 py-0.5 rounded-md bg-[#202226] border border-neutral-800/80 text-[9px] font-bold uppercase tracking-wider text-neutral-300 hover:bg-[#282a2f] hover:text-white transition-all flex items-center gap-1 shrink-0';
          btn.innerHTML = `<i data-lucide="eye" class="w-3 h-3 text-blue-400"></i> Maneuvers`;
        }
        lucide.createIcons();
      }
      if (states[currentIndex]) {
        updateWppUI(states[currentIndex].worst_placed_piece);
        updateMarkings(states[currentIndex]);
      }
    };

    const toggleAnalysisColumn = () => {
      analysisExpanded = !analysisExpanded;
      const col = $('analysis-col');
      const btn = $('toggle-analysis-btn');
      if (!col || !btn) return;

      if (analysisExpanded && activeTab === 'game') {
        col.classList.remove('hidden');
      } else {
        col.classList.add('hidden');
      }

      if (analysisExpanded) {
        btn.classList.remove('opacity-60');
        btn.innerHTML = `<i data-lucide="bar-chart-2" class="w-4 h-4 text-emerald-400"></i> <span>Metrics</span>`;
      } else {
        btn.classList.add('opacity-60');
        btn.innerHTML = `<i data-lucide="bar-chart-2" class="w-4 h-4 text-neutral-500"></i> <span>Metrics</span>`;
      }
      lucide.createIcons();
      ground && ground.redrawAll();
    };

    const navigate = d => {
      currentIndex = typeof d === 'number' ? Math.max(0, Math.min(states.length - 1, d)) :
        ({ start: 0, end: states.length - 1 }[d] ?? Math.max(0, Math.min(states.length - 1, currentIndex + (d === 'forward' ? 1 : d === 'back' ? -1 : 0))));
      updateBoard();
    };

    const confirmReset = () => { if (confirm("Reset current board and history back to start?")) { reset(); updateBoard(); } };

    const toggleBoardOrientation = () => {
      ground.set({orientation: ground.state.orientation==='white'?'black':'white'});
      renderCoords(); updateMarkings(states[currentIndex]);
    };

    const tabClass = active => `tab-btn flex-1 py-2.5 text-center cursor-pointer text-xs rounded-lg transition-all uppercase ${
      active 
        ? 'text-white bg-[#10b981] border border-emerald-500/30 shadow-md font-extrabold' 
        : 'text-neutral-400 hover:text-white bg-transparent border-transparent font-semibold'
    }`;

    const switchTab = id => {
      activeTab = id;
      $$('.tab-content').forEach(c => { const on = c.id===`${id}-tab`; c.classList.toggle('hidden', !on); c.classList.toggle('flex', on); });
      $$('.tab-btn').forEach(b => { b.className = tabClass(b.dataset.tab === id); });

      const col = $('analysis-col');
      if (col) {
        if (activeTab === 'game' && analysisExpanded) {
          col.classList.remove('hidden');
        } else {
          col.classList.add('hidden');
        }
      }
    };

    const moveEl = (s, idx, active) => {
      if (!s) return '<div class="w-full"></div>';
      const act = active ? 'bg-emerald-950/40 border-emerald-500/50 text-white font-extrabold shadow-md ring-1 ring-emerald-500/20' : 'bg-[#2b2d31]/60 hover:bg-[#35373c] border-neutral-700/80 text-neutral-300 hover:text-white';
      const badge = s.classification === 'loading' ? `<span class="inline-flex items-center text-[9px] text-neutral-500 animate-pulse font-bold shrink-0 w-8 h-8 justify-center">...</span>` : (s.classification && s.classification !== 'unknown' ? badgeHtml(s.classification, 'w-8 h-8') : `<span class="w-8 h-8 shrink-0 block"></span>`);
      return `<button onclick="navigate(${idx})" data-state-index="${idx}" ${active?'data-active="true"':''} class="flex items-center justify-between pl-3 pr-1 py-1 rounded-xl font-mono transition-all border ${act} w-full min-w-0"><span class="font-bold text-xs sm:text-sm truncate mr-1.5">${s.san}</span>${badge}</button>`;
    };

    const renderMoveList = () => {
      const c = $('move-list'); if (!c) return;

      if (states.length <= 1) {
        c.innerHTML = `<div class="flex flex-col items-center justify-center text-center py-12 px-6 h-full text-neutral-400 my-auto"><i data-lucide="git-commit" class="w-16 h-16 mb-4 text-neutral-500 shrink-0"></i><h3 class="text-sm font-bold text-white mb-1">No moves played yet</h3><p class="text-xs max-w-[250px] leading-relaxed">Make moves on the board or import a PGN game.</p></div>`;
        lucide.createIcons(); renderStats(); return;
      }

      let items = '';
      const turns = Math.ceil((states.length - 1) / 2);
      for (let i = 1; i <= turns; i++) {
        const w = 2*i-1, b = 2*i;
        items += `<div class="text-neutral-500 font-mono text-xs sm:text-sm font-black text-right self-center pr-1.5 select-none">${i}.</div><div class="min-w-0">${moveEl(states[w], w, currentIndex===w)}</div><div class="min-w-0">${states[b] ? moveEl(states[b], b, currentIndex===b) : '<div class="w-full"></div>'}</div>`;
      }

      c.innerHTML = `<div class="overflow-y-auto flex-1 pr-1.5"><div class="grid grid-cols-[24px_1fr_1fr] gap-x-2.5 gap-y-2 items-center">${items}</div></div>`;
      lucide.createIcons(); renderStats();
      setTimeout(() => { const a=c.querySelector('[data-active="true"]'); a && a.scrollIntoView({behavior:'smooth',block:'nearest'}); }, 50);
    };

    const renderStats = () => {
      const stats = Object.fromEntries(Object.keys(CLS).map(k => [k, { w: 0, b: 0 }]));
      const tot = { w: 0, b: 0 }, accs = { w: [], b: [] };
      states.slice(1).forEach(s => {
        tot[s.color]++;
        if (s.classification && stats[s.classification]) stats[s.classification][s.color]++;
        if (s.move_accuracy != null) accs[s.color].push(s.move_accuracy);
      });

      const calcAcc = col => {
        const a = accs[col]; if (!a.length) return 100;
        const harm = a.length / a.map(x => 1 / Math.max(x, 5)).reduce((s, v) => s + v, 0);
        return Math.round((a.reduce((x, y) => x + y, 0) / a.length + harm) / 2);
      };

      const wAcc = tot.w > 0 ? calcAcc('w') : 100, bAcc = tot.b > 0 ? calcAcc('b') : 100;
      let rowsHtml = '', hasCls = false;

      Object.entries(CLS).forEach(([key, cfg]) => {
        const wc = stats[key]?.w || 0, bc = stats[key]?.b || 0;
        if (wc || bc) {
          hasCls = true;
          const borderW = wc ? 'bg-emerald-950/35 text-emerald-400 border-emerald-500/20 font-black' : 'bg-neutral-900/40 text-neutral-600 border-neutral-800/40';
          const borderB = bc ? 'bg-emerald-950/35 text-emerald-400 border-emerald-500/20 font-black' : 'bg-neutral-900/40 text-neutral-600 border-neutral-800/40';
          rowsHtml += `
            <div class="grid grid-cols-[1fr_32px_110px_1fr] sm:grid-cols-[1fr_36px_130px_1fr] gap-x-2.5 items-center py-1.5 px-2 rounded-xl bg-neutral-900/20 border border-neutral-800/40 hover:bg-neutral-800/30 hover:border-neutral-700/30 transition-all duration-150">
              <div class="text-right"><span class="font-mono text-xs px-2.5 py-0.5 rounded-md border ${borderW}">${wc}</span></div>
              <div class="flex justify-center">${badgeHtml(key, 'w-7 h-7')}</div>
              <div class="text-left"><span class="text-xs font-bold text-neutral-300 tracking-wide block truncate">${cfg.label}</span></div>
              <div class="text-left"><span class="font-mono text-xs px-2.5 py-0.5 rounded-md border ${borderB}">${bc}</span></div>
            </div>`;
        }
      });

      const content = hasCls ? `<div class="flex flex-col gap-1.5">${rowsHtml}</div>` : 
        `<div class="flex flex-col items-center justify-center text-center py-12 px-6 flex-1 text-neutral-500">
          <i data-lucide="git-commit" class="w-12 h-12 mb-3 text-neutral-600 shrink-0"></i>
          <h4 class="text-xs font-bold text-neutral-300 mb-1">No classifications recorded</h4>
          <p class="text-[11px] max-w-[220px] leading-relaxed">Make some moves to generate move quality statistics.</p>
        </div>`;

      $('stats-grid').innerHTML = `
        <div class="grid grid-cols-2 gap-2.5 shrink-0">
          <div class="panel px-3 py-2 flex items-center justify-between border-neutral-700/50 shadow-sm bg-[#131316]/60">
            <div class="flex items-center gap-2 min-w-0">
              <span class="w-2.5 h-2.5 rounded-full bg-neutral-100 border border-neutral-300 shrink-0"></span>
              <span class="text-xs font-black text-neutral-300 uppercase tracking-wider truncate">White</span>
            </div>
            <div class="flex items-baseline shrink-0">
              <span class="text-base font-black text-white font-mono">${wAcc}%</span>
            </div>
          </div>
          <div class="panel px-3 py-2 flex items-center justify-between border-neutral-700/50 shadow-sm bg-[#131316]/60">
            <div class="flex items-center gap-2 min-w-0">
              <span class="w-2.5 h-2.5 rounded-full bg-neutral-950 border border-neutral-700 shrink-0"></span>
              <span class="text-xs font-black text-neutral-300 uppercase tracking-wider truncate">Black</span>
            </div>
            <div class="flex items-baseline shrink-0">
              <span class="text-base font-black text-white font-mono">${bAcc}%</span>
            </div>
          </div>
        </div>
        <div class="panel p-3.5 flex-1 flex flex-col min-h-0 border-neutral-700/50 shadow-sm">
          <div class="flex items-center justify-between border-b border-neutral-800 pb-1.5 mb-2 shrink-0">
            <span class="text-[11px] font-black text-neutral-400 uppercase tracking-wider">Move Quality</span>
            <div class="flex gap-1 items-center text-[9px] font-bold text-neutral-500 uppercase tracking-widest">
              <span>W</span>
              <span class="text-neutral-700 mx-1">•</span>
              <span>B</span>
            </div>
          </div>
          <div class="flex-1 overflow-y-auto pr-0.5 min-h-0">${content}</div>
        </div>
      `;
      lucide.createIcons();
    };

    const importPgn = () => {
      const p = $('pgn-input').value.trim(); if (!p) return alert("Please paste PGN contents to import.");
      const tc = new Chess();
      if (tc.load_pgn(p)) {
        const fm = p.match(/\[FEN\s+"([^"]+)"\]/i), sf = fm ? fm[1] : START_FEN;
        states = [newState(sf, "Starting Position")];
        const b = new Chess(sf);
        tc.history({verbose:true}).forEach((m, idx) => {
          b.move({from:m.from, to:m.to, ...(m.promotion?{promotion:m.promotion}:{})});
          states.push(newState(b.fen(), null, {san:m.san, color:m.color, moveNumber:Math.floor(idx/2)+1, move_uci:m.from+m.to+(m.promotion||'')}));
        });
        currentIndex = states.length-1; updateBoard(); switchTab('game'); showStatus("PGN imported successfully!");
      } else alert("Incorrect or invalid PGN format.");
    };

    const loadCustomFen = () => {
      const f = $('fen-input').value.trim(), tc = new Chess();
      if (tc.load(f)) {
        const p = f.split(' '), mn = parseInt(p[p.length-1]);
        states = [newState(f, "Custom Position", {moveNumber: isNaN(mn)?1:mn})];
        currentIndex = 0; updateBoard(); switchTab('game'); showStatus("FEN loaded successfully!");
      } else alert("Invalid FEN string.");
    };

    const showStatus = msg => {
      const el = $('io-status'); if (!el) return;
      el.innerText=msg; el.classList.remove('hidden'); setTimeout(()=>el.classList.add('hidden'),2000);
    };

    const copyCurrentFen = () => clipboard(states[currentIndex].fen).then(()=>showStatus("FEN copied!")).catch(()=>alert("Copy failed."));
    const copyCurrentPgn = () => {
      const p = $('pgn-input')?.value; if (!p) return alert("No moves played yet.");
      clipboard(p).then(()=>showStatus("PGN copied!")).catch(()=>alert("Copy failed."));
    };

    $('tab-bar').innerHTML = TABS.map(t => `<button class="${tabClass(t.active)}" data-tab="${t.id}" onclick="switchTab('${t.id}')">${t.label}</button>`).join('');
    $('kbd-help').innerHTML = SHORTCUTS.map(([k,v]) => `<div class="flex justify-between items-center border-b border-neutral-800/20 pb-1.5"><span class="kbd">${k}</span><span class="text-[10px]">${v}</span></div>`).join('');

    Object.assign(window, {navigate, toggleBoardOrientation, confirmReset, switchTab, loadCustomFen, copyCurrentFen, copyCurrentPgn, importPgn, selectMetric, toggleWppDropdown, toggleSingleWppPath, toggleAllManeuvers, toggleAnalysisColumn, toggleMute});

    const keyActions = {
      arrowleft:()=>navigate('back'), pageup:()=>navigate('back'), backspace:()=>navigate('back'),
      arrowright:()=>navigate('forward'), pagedown:()=>navigate('forward'),
      arrowup:()=>navigate('start'), home:()=>navigate('start'),
      arrowdown:()=>navigate('end'), end:()=>navigate('end'),
      f:toggleBoardOrientation, m:()=>switchTab('game'), s:()=>switchTab('stats'), i:()=>switchTab('io'),
      c:copyCurrentFen, p:copyCurrentPgn, r:confirmReset
    };
    document.addEventListener('keydown', e => {
      if (['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) return;
      const k = e.key.toLowerCase(); if (keyActions[k]) { e.preventDefault(); keyActions[k](); }
    });
    $('fen-input').addEventListener('keydown', e => e.key==='Enter' && loadCustomFen());
    window.addEventListener('resize', () => { ground && ground.redrawAll(); updateMarkings(states[currentIndex]); });

    const getSquareCenter = (square, rect, o) => {
      const col = square.charCodeAt(0) - 97;
      const row = 8 - parseInt(square[1]);
      const c = o ? col : 7 - col;
      const r = o ? row : 7 - row;
      const sqSize = rect.width / 8;
      return {
        x: rect.left + (c + 0.5) * sqSize,
        y: rect.top + (r + 0.5) * sqSize
      };
    };

    const distToSegment = (px, py, ax, ay, bx, by) => {
      const dx = bx - ax;
      const dy = by - ay;
      const l2 = dx * dx + dy * dy;
      if (l2 === 0) return Math.hypot(px - ax, py - ay);
      let t = ((px - ax) * dx + (py - ay) * dy) / l2;
      t = Math.max(0, Math.min(1, t));
      return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
    };

    const showTooltip = (text, x, y) => {
      let el = $('shapes-tooltip');
      if (!el) {
        el = Object.assign(document.createElement('div'), {
          id: 'shapes-tooltip',
          className: 'fixed z-[100] pointer-events-none bg-[#18181c]/95 border border-neutral-700/80 text-white text-xs px-2.5 py-1.5 rounded-lg shadow-xl font-sans font-semibold tracking-wide transition-opacity duration-150'
        });
        document.body.appendChild(el);
      }
      el.innerText = text;
      el.style.opacity = '1';
      
      const rect = el.getBoundingClientRect();
      let left = x + 15;
      let top = y + 15;
      if (left + rect.width > window.innerWidth) left = x - rect.width - 15;
      if (top + rect.height > window.innerHeight) top = y - rect.height - 15;
      el.style.left = `${left}px`;
      el.style.top = `${top}px`;
    };

    const hideTooltip = () => {
      const el = $('shapes-tooltip');
      if (el) el.style.opacity = '0';
    };

    $('board').addEventListener('mousemove', e => {
      const markings = window.currentBoardMarkings || [];
      const rect = $('board').getBoundingClientRect();
      if (!maxDistSquare(e, rect)) { hideTooltip(); return; }
      
      const o = ground.state.orientation === 'white';
      const mx = e.clientX, my = e.clientY;
      let closestMarking = null, minDist = Infinity;
      
      markings.forEach(m => {
        if (!m.reason) return;
        const centerOrig = getSquareCenter(m.orig, rect, o);
        if (m.dest) {
          const centerDest = getSquareCenter(m.dest, rect, o);
          const d = distToSegment(mx, my, centerOrig.x, centerOrig.y, centerDest.x, centerDest.y);
          if (d < minDist) { minDist = d; closestMarking = m; }
        } else {
          const d = Math.hypot(mx - centerOrig.x, my - centerOrig.y);
          if (d < minDist) { minDist = d; closestMarking = m; }
        }
      });
      
      if (closestMarking && minDist < 24) showTooltip(closestMarking.reason, mx, my);
      else hideTooltip();
    });

    function maxDistSquare(e, rect) {
      return (e.clientX >= rect.left && e.clientX <= rect.right &&
              e.clientY >= rect.top && e.clientY <= rect.bottom);
    }

    $('board').addEventListener('mouseleave', hideTooltip);

    reset(); chess = new Chess(); initBoard(); updateBoard(); lucide.createIcons(); audio.init();
  </script>
</body>
</html>
"""

# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        load_openings()
        try:
            logger.info("Pre-initializing Stockfish engine...")
            eng.get()
        except Exception as e:
            logger.error("Failed to pre-initialize Stockfish engine: %s", e)
    try:
        app.run(debug=True, port=5000)
    finally:
        eng.close()