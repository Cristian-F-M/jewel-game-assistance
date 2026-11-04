# /// script
# requires-python = ">=3.10"
# dependencies = ["opencv-python", "numpy", "pillow"]
# ///
"""
Jewel Block King Assistant v5 (tkinter + ADB)

    uv run jewel_block_king_gui.py

Flujo recomendado:
  1) Probar ADB  2) Calibrar (tablero casi vacío)  3) Calibrar arrastre
  4) Capturar y sugerir (con «Mostrar detección»)  5) Probar 1ª pieza
  6) Asistente en vivo / Jugar automáticamente
"""

import itertools
import json
import math
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

GRAB_MODES = (
    "centro de la pieza",
    "celda más cercana al centro",
    "centro de la ranura",
)
DEFAULT_OPTS = dict(
    detect=False,
    step=False,
    interval=1.5,
    recalc=True,
    reuse=False,
    drag="motionevent",
    grab=GRAB_MODES[0],
)


def config_path():
    """La configuración vive junto al script: no depende del directorio desde el que lo lances."""
    for p in (
        Path(__file__).resolve().parent / "config.json",
        Path.home() / ".jewel_block_king.json",
    ):
        try:
            if p.exists():
                return p
            p.touch()  # comprueba que se puede escribir
            p.unlink()
            return p
        except OSError:
            continue
    return Path("config.json")


CFG_PATH = config_path()
LEGACY_CFG = Path.cwd() / "config.json"  # versiones anteriores lo guardaban aquí
COLORS = [(0, 200, 255), (0, 255, 0), (255, 120, 0)]  # BGR por pieza
DEFAULT_CFG: dict[str, int | None | float | str] = {
    "n": 8,
    "grid": None,
    "slots": None,
    "piece_cell": None,
    "empty": None,
    "thr": 45,
    "thr_piece": 30,
    "drag_dx": 0.0,
    "drag_dy": 0.0,
    "drag_ax": "center",
    "drag_ay": "center",
    "grab": GRAB_MODES[0],
    "ms": 400,
    "pause": 0.9,
    "wait": 1.8,
}


# ============================================================ ADB
class Adb:
    def __init__(self):
        self.serial = ""

    def _base(self):
        return ["adb"] + (["-s", self.serial] if self.serial else [])

    def run(self, *args, timeout=20):
        return subprocess.run(
            self._base() + list(args), capture_output=True, check=True, timeout=timeout
        ).stdout

    def devices(self):
        return subprocess.run(
            ["adb", "devices", "-l"], capture_output=True, check=True, timeout=20
        ).stdout.decode(errors="ignore")

    def screenshot(self):
        data = self.run("exec-out", "screencap", "-p")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(
                "No pude decodificar la captura (¿móvil bloqueado o sin autorizar?)."
            )
        return img

    def swipe(self, x1, y1, x2, y2, ms):
        self.run(
            "shell", "input", "swipe", *[str(int(v)) for v in (x1, y1, x2, y2, ms)]
        )

    def motion(self, events, timeout=40):
        """events: ('DOWN'|'MOVE'|'UP', x, y) o ('sleep', seg). Todo en una sola llamada adb."""
        parts = []
        for ev in events:
            if ev[0] == "sleep":
                parts.append(f"sleep {ev[1]}")
            else:
                parts.append(f"input motionevent {ev[0]} {int(ev[1])} {int(ev[2])}")
        out = (
            self.run("shell", "; ".join(parts), timeout=timeout)
            .decode(errors="ignore")
            .lower()
        )
        if "usage" in out or "error" in out or "exception" in out:
            raise RuntimeError("motionevent no soportado")

    @staticmethod
    def path(sx: int, sy: int, tx: int, ty: int, steps: int = 6):
        ev: list[tuple[str, int | float] | tuple[str, int | float, int | float]] = [
            ("DOWN", sx, sy),
            ("sleep", 0.15),
        ]
        for k in range(1, steps + 1):
            ev += [
                ("MOVE", sx + (tx - sx) * k / steps, sy + (ty - sy) * k / steps),
                ("sleep", 0.03),
            ]
        return ev

    def drag_motion(self, sx, sy, tx, ty, hold=0.3):
        self.motion(self.path(sx, sy, tx, ty) + [("sleep", hold), ("UP", tx, ty)])


# ============================================================ geometría / visión
def cell_xy(cfg, r, c):
    gx, gy, gw, gh = cfg["grid"]
    n = cfg["n"]
    return gx + c * gw / n, gy + r * gh / n, gw / n, gh / n


def target_point(cfg, p, r, c):
    """Devuelve (dedo_x, dedo_y, destino_x, destino_y).
    destino = centro de la huella de la pieza sobre el tablero (hacia donde apuntan las flechas).
    dedo    = punto donde hay que soltar el dedo: el ancla elegida (borde/centro) de esa huella
              menos el offset medido entre la pieza flotante y el dedo."""
    x, y, w, h = cell_xy(cfg, r, c)
    bx0, by0, bx1, by1 = x, y, x + p["cols"] * w, y + p["rows"] * h
    ax = {"left": bx0, "center": (bx0 + bx1) / 2, "right": bx1}[
        cfg.get("drag_ax", "center")
    ]
    ay = {"top": by0, "center": (by0 + by1) / 2, "bottom": by1}[
        cfg.get("drag_ay", "center")
    ]
    return ax - cfg["drag_dx"], ay - cfg["drag_dy"], (bx0 + bx1) / 2, (by0 + by1) / 2


def grab_point(cfg, p, slot):
    """Punto de la bandeja donde el dedo toca la pieza (NO depende del tamaño del recuadro
    de la ranura, salvo en el modo «centro de la ranura»)."""
    cx, cy = p["center"]
    mode = cfg.get("grab", GRAB_MODES[0])
    if mode == GRAB_MODES[2]:
        x, y, w, h = slot
        return x + w / 2, y + h / 2
    if mode == GRAB_MODES[1] and p["rects"]:
        r = min(
            p["rects"],
            key=lambda q: (q[0] + q[2] / 2 - cx) ** 2 + (q[1] + q[3] / 2 - cy) ** 2,
        )
        return r[0] + r[2] / 2, r[1] + r[3] / 2
    return cx, cy


def choose_anchor(d, cell):
    """d: {ancla: [offset por muestra]}. Elige el ancla cuyo offset es más constante entre piezas
    de distinto tamaño; prefiere 'center' salvo que otra sea claramente mejor."""
    spread = {k: max(v) - min(v) for k, v in d.items()}
    best = min(spread, key=spread.get)
    if spread["center"] <= spread[best] + 0.15 * cell:
        best = "center"
    return best, sum(d[best]) / len(d[best]), spread


def cell_patch(img, cfg, r, c):
    x, y, w, h = cell_xy(cfg, r, c)
    p = img[int(y + 0.3 * h) : int(y + 0.7 * h), int(x + 0.3 * w) : int(x + 0.7 * w)]
    return p.reshape(-1, 3).astype(np.float32).mean(axis=0)


def read_board(img, cfg):
    n, rows = cfg["n"], []
    for r in range(n):
        m = 0
        for c in range(n):
            d = np.linalg.norm(
                cell_patch(img, cfg, r, c) - np.array(cfg["empty"][(r + c) % 2])
            )
            if d > cfg["thr"]:
                m |= 1 << c
        rows.append(m)
    return tuple(rows)


def segment_slot(crop, cfg):
    """Máscara binaria (0/1) de la pieza dentro de una ranura de la bandeja.
    - Fondo = color más frecuente de la ranura (en Lab, tolera degradados suaves).
    - Píxel de pieza = distancia Lab al fondo > thr_piece.
    - Se cierran los huecos entre cuadritos / brillos internos y se descartan manchas pequeñas."""
    pc = cfg["piece_cell"]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    q = (lab >> 3).astype(np.int32)
    key = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    vals, counts = np.unique(key, return_counts=True)
    bg = lab[key == vals[counts.argmax()]].astype(np.float32).mean(axis=0)
    dist = np.linalg.norm(lab.astype(np.float32) - bg, axis=2)
    mask = (dist > cfg.get("thr_piece", 30)).astype(np.uint8)
    k = max(3, int(pc * 0.2)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    # silueta sólida: se rellenan los huecos internos (centros oscuros de las gemas, brillos)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= 0.15 * pc * pc]
    out = np.zeros_like(mask)
    if not cnts:
        return out
    # Pieza principal = mayor mancha cuyo centro cae en la zona central del recuadro (así un recuadro
    # grande que roce a otra pieza o a un adorno no altera la forma). Solo se le unen manchas pegadas.
    H, W = mask.shape
    boxes = [cv2.boundingRect(c) for c in cnts]
    central = [
        i
        for i, (x, y, w, h) in enumerate(boxes)
        if 0.15 * W <= x + w / 2 <= 0.85 * W and 0.15 * H <= y + h / 2 <= 0.85 * H
    ] or list(range(len(cnts)))
    m = max(central, key=lambda i: cv2.contourArea(cnts[i]))
    keep, grew = {m}, True
    while grew:  # se van uniendo manchas pegadas a las ya incluidas (cuadritos sueltos de una misma pieza)
        grew = False
        for i, (x, y, w, h) in enumerate(boxes):
            if i in keep:
                continue
            for j in keep:
                mx, my, mw, mh = boxes[j]
                gap = max(mx - (x + w), x - (mx + mw), my - (y + h), y - (my + mh), 0)
                if gap <= 0.6 * pc:
                    keep.add(i)
                    grew = True
                    break
    for i in keep:
        cv2.drawContours(out, [cnts[i]], -1, 1, thickness=-1)
    return out


def mask_bbox(mask, pc):
    """Caja de la pieza usando proyecciones: ignora brillos/sombras finos (< medio cuadrito)."""
    xs = np.where(mask.sum(axis=0) > 0.35 * pc)[0]
    ys = np.where(mask.sum(axis=1) > 0.35 * pc)[0]
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs[0]), int(xs[-1]) + 1, int(ys[0]), int(ys[-1]) + 1


def estimate_cell(dims, prior):
    """Tamaño de cuadrito en la bandeja que mejor explica TODAS las piezas a la vez
    (cada ancho/alto debe ser ≈ múltiplo entero), con preferencia por el valor calibrado."""
    best_s, best_cost = prior, 1e9
    for s in np.arange(prior * 0.7, prior * 1.3, 0.5):
        err = 0.0
        for d in dims:
            k = min(6, max(1, int(d / s + 0.5)))
            err += abs(d - k * s) / s
        cost = err / len(dims) + 0.15 * abs(math.log(s / prior))
        if cost < best_cost:
            best_cost, best_s = cost, float(s)
    return best_s


def shape_ok(masks, rows, cols, cw, ch):
    """La forma es válida si: celdas ≈ cuadradas, no hay filas/columnas vacías y es conexa."""
    if not (0.8 <= cw / ch <= 1.25):
        return False
    acc = 0
    for m in masks:
        if m == 0:
            return False
        acc |= m
    if acc != (1 << cols) - 1:
        return False
    cells = {(r, c) for r in range(rows) for c in range(cols) if (masks[r] >> c) & 1}
    seen, stack = set(), [next(iter(cells))]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in cells:
                stack.append((r + dr, c + dc))
    return len(seen) == len(cells)


def apply_override(auto, ov, s):
    """Sustituye la forma detectada por la dibujada a mano (misma posición/centro en pantalla)."""
    cells = sorted(ov["cells"])
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    cells = [(r - r0, c - c0) for r, c in cells]
    rows, cols = max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1
    masks = [0] * rows
    for r, c in cells:
        masks[r] |= 1 << c
    cx, cy = auto["center"]
    x0, y0 = cx - cols * s / 2, cy - rows * s / 2
    rects = [(x0 + c * s, y0 + r * s, s, s) for r, c in cells]
    return dict(
        auto,
        rows=rows,
        cols=cols,
        masks=masks,
        cells=cells,
        rects=rects,
        ok=shape_ok(masks, rows, cols, s, s),
        manual=True,
    )


def read_pieces(img, cfg, overrides=None, cleared=None):
    """overrides: {ranura: {'cells': set((r,c)), 'sig': firma de la detección automática}}.
    Un override se descarta solo si la pieza de esa ranura cambia (firma distinta o ranura vacía)."""
    pc = cfg["piece_cell"]
    raws = []
    for slot in cfg["slots"]:
        x, y, w, h = slot
        mask = segment_slot(img[y : y + h, x : x + w], cfg)
        bb = mask_bbox(mask, pc) if mask.sum() >= 0.25 * pc * pc else None
        raws.append((slot, mask, bb))
    dims = [d for (_, _, bb) in raws if bb for d in (bb[1] - bb[0], bb[3] - bb[2])]
    s = estimate_cell(dims, pc) if dims else pc
    pcs = []
    for idx, (slot, mask, bb) in enumerate(raws):
        auto = None
        if bb:
            x, y, w, h = slot
            x0, x1, y0, y1 = bb
            cols = max(1, int((x1 - x0) / s + 0.5))
            rows = max(1, int((y1 - y0) / s + 0.5))
            cw, ch = (x1 - x0) / cols, (y1 - y0) / rows
            masks, cells, rects = [0] * rows, [], []
            for r in range(rows):
                for c in range(cols):
                    ax0, ax1 = int(x0 + (c + 0.2) * cw), int(x0 + (c + 0.8) * cw)
                    ay0, ay1 = int(y0 + (r + 0.2) * ch), int(y0 + (r + 0.8) * ch)
                    if mask[ay0 : ay1 + 1, ax0 : ax1 + 1].mean() > 0.5:
                        masks[r] |= 1 << c
                        cells.append((r, c))
                        rects.append((x + x0 + c * cw, y + y0 + r * ch, cw, ch))
            ok = bool(cells) and shape_ok(masks, rows, cols, cw, ch)
            auto = dict(
                rows=rows,
                cols=cols,
                masks=masks,
                cells=cells,
                rects=rects,
                ok=ok,
                mask=mask * 255,
                cell=s,
                center=(x + (x0 + x1) / 2, y + (y0 + y1) / 2),
                manual=False,
                auto=dict(rows=rows, cols=cols, cells=list(cells)),
                auto_sig=(rows, cols, tuple(masks)),
            )
        ov = overrides.get(idx) if overrides else None
        if ov:
            if auto is None or ov["sig"] != auto["auto_sig"]:
                overrides.pop(
                    idx, None
                )  # la pieza cambió: la forma manual ya no aplica
                if cleared is not None:
                    cleared.append(idx)
            elif ov["cells"]:
                auto = apply_override(auto, {"cells": set(ov["cells"])}, s)
        pcs.append(auto)
    return pcs


def piece_preview(img, cfg, pieces, H=120):
    """Mosaico con lo que la app 've' en cada ranura: contorno (verde) y cuadritos detectados."""
    tiles = []
    for k, (slot, p) in enumerate(zip(cfg["slots"], pieces)):
        x, y, w, h = slot
        crop = img[y : y + h, x : x + w].copy()
        if p:
            col = (255, 255, 0) if p["ok"] else (0, 0, 255)
            cnts, _ = cv2.findContours(
                p["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(crop, cnts, -1, (0, 255, 0), 1)
            for rx, ry, rw, rh in p["rects"]:
                cv2.rectangle(
                    crop,
                    (int(rx - x), int(ry - y)),
                    (int(rx - x + rw), int(ry - y + rh)),
                    col,
                    2,
                )
        s = H / h
        tile = cv2.resize(crop, (max(1, int(w * s)), H), interpolation=cv2.INTER_AREA)
        label = (
            f"{k + 1}: vacia"
            if not p
            else (
                f"{k + 1}: {p['rows']}x{p['cols']}"
                + (" MANUAL" if p.get("manual") else "")
                + ("" if p["ok"] else " ?!")
            )
        )
        cv2.putText(
            tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, H - 1), (90, 90, 90), 1)
        tiles.append(tile)
    return np.hstack(tiles)


# ============================================================ solver
def try_place(board, p, r, c, n, used=None):
    """Devuelve (nuevo_tablero, líneas, filas_limpiadas, mascara_columnas) o None.
    `used`: casillas ya ocupadas por piezas de esta ronda (no se pueden reutilizar)."""
    if r + p["rows"] > n or c + p["cols"] > n:
        return None
    full = (1 << n) - 1
    b = list(board)
    for i, m in enumerate(p["masks"]):
        m <<= c
        if b[r + i] & m:
            return None
        if used and used[r + i] & m:
            return None
        b[r + i] |= m
    full_rows = [i for i in range(n) if b[i] == full]
    col_full = full
    for row in b:
        col_full &= row
    cleared = len(full_rows) + bin(col_full).count("1")
    for i in full_rows:
        b[i] = 0
    if col_full:
        b = [row & ~col_full for row in b]
    return tuple(b), cleared, full_rows, col_full


def evaluate(board, n):
    filled = sum(bin(r).count("1") for r in board)
    iso = 0
    for r in range(n):
        for c in range(n):
            if (board[r] >> c) & 1:
                continue
            free = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n and not (board[rr] >> cc) & 1:
                    free += 1
            iso += 2 if free == 0 else (1 if free == 1 else 0)
    return -filled - 6 * iso


def plan(board, pieces, n, reuse=False, beam=12):
    """Prueba todos los órdenes. Si reuse=False, ninguna pieza usa casillas ya usadas
    por otra pieza de la misma ronda (aunque una línea completada las haya liberado)."""
    idx = [i for i, p in enumerate(pieces) if p]
    best_key, best_moves = (0, -1e9), []
    for perm in itertools.permutations(idx):
        states = [(0, board, [], 0, (0,) * n)]
        for i in perm:
            nxt, p = [], pieces[i]
            for _, b, mv, cl, used in states:
                for r in range(n - p["rows"] + 1):
                    for c in range(n - p["cols"] + 1):
                        res = try_place(b, p, r, c, n, None if reuse else used)
                        if not res:
                            continue
                        nb, k = res[0], res[1]
                        if reuse:
                            nu = used
                        else:
                            nu = tuple(
                                used[j]
                                | (
                                    (p["masks"][j - r] << c)
                                    if r <= j < r + p["rows"]
                                    else 0
                                )
                                for j in range(n)
                            )
                        nxt.append(
                            (
                                evaluate(nb, n) + 50 * (cl + k),
                                nb,
                                mv + [(i, r, c)],
                                cl + k,
                                nu,
                            )
                        )
            if not nxt:
                break
            nxt.sort(key=lambda t: -t[0])
            key = (len(nxt[0][2]), nxt[0][0])
            if key > best_key:
                best_key, best_moves = key, nxt[0][2]
            states = nxt[:beam]
    return best_moves


def plan_best(board, pieces, n, allow_reuse):
    """Hay que colocar SIEMPRE las 3 piezas para que salgan otras 3: prioriza jugadas que las
    coloquen todas. Si el modo estricto no lo logra, prueba reutilizando casillas liberadas."""
    cnt = sum(1 for p in pieces if p)
    mv = plan(board, pieces, n, reuse=allow_reuse)
    if not allow_reuse and len(mv) < cnt:
        alt = plan(board, pieces, n, reuse=True)
        if len(alt) > len(mv):
            mv = alt
    return mv


def replay(board, pieces, moves, n):
    out, b = [], board
    for i, r, c in moves:
        res = try_place(b, pieces[i], r, c, n)
        if not res:
            break
        b = res[0]
        out.append((res[2], res[3]))
    return out


# ============================================================ dibujo
def draw_arrow(img, p0, p1, color, thick=6):
    p0, p1 = (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1]))
    tip = min(0.35, 60.0 / max(1.0, math.hypot(p1[0] - p0[0], p1[1] - p0[1])))
    cv2.arrowedLine(img, p0, p1, (0, 0, 0), thick + 5, cv2.LINE_AA, tipLength=tip)
    cv2.arrowedLine(img, p0, p1, color, thick, cv2.LINE_AA, tipLength=tip)


def render(img, cfg, board, pieces, moves, detect):
    out = img.copy()
    n = cfg["n"]
    gx, gy, gw, gh = cfg["grid"]
    if detect:
        for r in range(n):
            for c in range(n):
                x, y, w, h = cell_xy(cfg, r, c)
                cv2.rectangle(
                    out, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1
                )
                if (board[r] >> c) & 1:
                    cv2.circle(
                        out,
                        (int(x + w / 2), int(y + h / 2)),
                        max(3, int(w / 6)),
                        (0, 0, 255),
                        -1,
                    )
        for sx, sy, sw, sh in cfg["slots"]:
            cv2.rectangle(out, (sx, sy), (sx + sw, sy + sh), (200, 200, 200), 2)
        for p in pieces:  # forma detectada de cada pieza en la zona de espera
            if p:
                col = (255, 255, 0) if p["ok"] else (0, 0, 255)
                for x, y, w, h in p["rects"]:
                    cv2.rectangle(
                        out, (int(x), int(y)), (int(x + w), int(y + h)), col, 2
                    )
    if moves:
        layer = out.copy()
        for i, r, c in moves:
            for pr, pc in pieces[i]["cells"]:
                x, y, w, h = cell_xy(cfg, r + pr, c + pc)
                cv2.rectangle(
                    layer, (int(x), int(y)), (int(x + w), int(y + h)), COLORS[i % 3], -1
                )
        out = cv2.addWeighted(layer, 0.55, out, 0.45, 0)
        cleared = replay(board, pieces, moves, n)
        for k, (i, r, c) in enumerate(moves):
            col, p = COLORS[i % 3], pieces[i]
            x, y, w, h = cell_xy(cfg, r, c)
            cv2.putText(
                out,
                str(k + 1),
                (int(x + 6), int(y + h * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.8, w / 40),
                (255, 255, 255),
                4,
            )
            sx, sy, sw, sh = cfg["slots"][i]
            cv2.rectangle(out, (sx, sy), (sx + sw, sy + sh), col, 5)
            cv2.putText(
                out,
                f"#{k + 1}",
                (sx + 8, sy + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                col,
                4,
            )
            if k < len(cleared):  # líneas que se completan con esta pieza
                fr, cf = cleared[k]
                for rr in fr:
                    yy = int(gy + rr * gh / n)
                    cv2.rectangle(
                        out, (int(gx), yy), (int(gx + gw), int(yy + gh / n)), col, 4
                    )
                for cc in range(n):
                    if (cf >> cc) & 1:
                        xx = int(gx + cc * gw / n)
                        cv2.rectangle(
                            out, (xx, int(gy)), (int(xx + gw / n), int(gy + gh)), col, 4
                        )
            # flecha: de la pieza en la bandeja al DESTINO de la ficha en el tablero
            fx, fy, dcx, dcy = target_point(cfg, p, r, c)
            draw_arrow(out, p["center"], (dcx, dcy), col)
            if (
                detect
            ):  # depuración del dedo: verde = dónde toca, magenta = dónde suelta
                gx0, gy0 = grab_point(cfg, p, cfg["slots"][i])
                cv2.circle(out, (int(gx0), int(gy0)), 10, (0, 255, 0), -1)
                cv2.line(
                    out, (int(gx0), int(gy0)), (int(fx), int(fy)), (255, 0, 255), 2
                )
                cv2.circle(out, (int(fx), int(fy)), 14, (255, 0, 255), 3)
    return out


# ============================================================ app
class App:
    BOARD_STEPS = [
        (
            "grid",
            "Arrastra un rectángulo sobre el TABLERO (solo las casillas n×n) y pulsa «Aceptar».",
        ),
        (
            "slot0",
            "Rodea la zona de la PIEZA 1 (izquierda), algo más grande que la pieza.",
        ),
        ("slot1", "Rodea la zona de la PIEZA 2 (centro)."),
        ("slot2", "Rodea la zona de la PIEZA 3 (derecha)."),
        (
            "cell",
            "Rodea UN solo cuadrito de una pieza TAL COMO SE VE EN LA ZONA DE ESPERA "
            "(es más pequeño que en el tablero).",
        ),
    ]
    DRAG_STEPS = [
        (
            "float",
            "Tu dedo (cruz roja) está sosteniendo una pieza. Rodea SOLO la pieza flotante "
            "(sus cuadritos) y pulsa «Aceptar».",
        ),
    ]

    def __init__(self, root):
        self.root = root
        root.title("Jewel Block King Assistant")
        self.adb = Adb()
        self.cfg: dict[str, int | None | float | str] = dict(DEFAULT_CFG)
        self.cfg_msg = self._load_cfg()
        self.opts = dict(DEFAULT_OPTS)
        self.opts.update(
            {k: v for k, v in self.cfg.get("opts", {}).items() if k in DEFAULT_OPTS}
        )
        self._save_job = None
        self.dc = None
        self.no_motion = False
        self.overrides = {}  # formas dibujadas a mano por ranura
        self.last_res = None
        self.editor = None
        self.raw = None
        self.annot = None
        self.photo = None
        self.scale = 1.0
        self.calib = None
        self.drag_state = None
        self.busy = False
        self.uiq = queue.Queue()
        self.stop_ev = threading.Event()
        self.step_ev = threading.Event()
        self._build()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(80, self._poll)
        self.log(self.cfg_msg)
        if self.configured():
            self.log(
                "✔ Calibración del tablero y las piezas recuperada. Ya puedes capturar y sugerir."
            )
        else:
            self.log(
                "1) Probar ADB  2) Calibrar  3) Calibrar arrastre  4) Capturar y sugerir."
            )

    def _load_cfg(self):
        for p, tag in (
            (CFG_PATH, ""),
            (LEGACY_CFG, " (copiada del directorio actual)"),
        ):
            if not p.exists():
                continue
            try:
                loaded = json.loads(p.read_text())
            except Exception as e:  # noqa
                return f"⚠ No pude leer {p}: {e}"
            self.cfg.update(loaded)
            if "lift" in loaded and "drag_dy" not in loaded:  # migración v1
                self.cfg["drag_dy"] = -float(loaded["lift"])
            return f"Configuración cargada de {p}{tag}"
        return f"Sin configuración previa. Se guardará sola en {CFG_PATH}"

    # ---------------------------------------------------------- UI
    def _build(self):
        main = ttk.Frame(self.root, padding=6)
        main.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            main, width=360, height=640, bg="#202020", highlightthickness=0
        )
        self.canvas.pack(side="left")
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        right = ttk.Frame(main, padding=(10, 0))
        right.pack(side="left", fill="both", expand=True)
        nb = ttk.Notebook(right)
        nb.pack(fill="x")
        tab_a, tab_s = ttk.Frame(nb, padding=6), ttk.Frame(nb, padding=6)
        nb.add(tab_a, text="Acciones")
        nb.add(tab_s, text="Ajustes")

        # --- pestaña acciones
        f = ttk.LabelFrame(tab_a, text="Dispositivo", padding=6)
        f.pack(fill="x")
        self.v_serial = tk.StringVar(value=self.cfg.get("serial", ""))
        ttk.Label(f, text="Serial (opcional):").grid(row=0, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.v_serial, width=16).grid(row=0, column=1, padx=4)
        self.b_adb = ttk.Button(
            f, text="Probar ADB", command=lambda: self.run_job(self.job_devices)
        )
        self.b_adb.grid(row=0, column=2)

        self.calib_frame = ttk.LabelFrame(tab_a, text="Calibración en curso", padding=6)
        self.calib_lbl = ttk.Label(self.calib_frame, wraplength=280, justify="left")
        self.calib_lbl.pack(fill="x")
        bf = ttk.Frame(self.calib_frame)
        bf.pack(fill="x", pady=4)
        ttk.Button(bf, text="✔ Aceptar selección", command=self.calib_accept).pack(
            side="left"
        )
        ttk.Button(bf, text="✖ Cancelar", command=self.calib_cancel).pack(
            side="left", padx=6
        )

        self.actions_frame = ttk.LabelFrame(tab_a, text="Acciones", padding=6)
        self.actions_frame.pack(fill="x", pady=6)
        f = self.actions_frame
        self.b_calib = ttk.Button(
            f, text="🧭 Calibrar tablero y piezas", command=self.start_calib
        )
        self.b_dcal = ttk.Button(
            f,
            text="🎯 Calibrar arrastre (offset del dedo)",
            command=self.start_drag_calib,
        )
        self.b_snap = ttk.Button(
            f,
            text="📸 Capturar y sugerir",
            command=lambda: self.sync_and(self.run_job, self.job_capture),
        )
        self.b_test = ttk.Button(
            f,
            text="🧪 Probar solo la 1ª pieza",
            command=lambda: self.sync_and(self.run_job, self.job_test_move),
        )
        self.b_assist = ttk.Button(
            f, text="👁 Asistente en vivo", command=lambda: self.start_loop(False)
        )
        self.b_auto = ttk.Button(
            f, text="▶ Jugar automáticamente", command=lambda: self.start_loop(True)
        )
        self.b_stop = ttk.Button(
            f, text="■ Detener", command=self.stop_loop, state="disabled"
        )
        self.b_next = ttk.Button(
            f,
            text="⏭ Continuar (paso a paso)",
            command=self.step_ev.set,
            state="disabled",
        )
        self.b_save = ttk.Button(f, text="💾 Guardar imagen", command=self.save_image)
        self.b_edit = ttk.Button(
            f, text="✏ Editar formas de las piezas", command=self.open_editor
        )
        btns = (
            self.b_calib,
            self.b_dcal,
            self.b_snap,
            self.b_test,
            self.b_assist,
            self.b_auto,
            self.b_stop,
            self.b_next,
            self.b_edit,
            self.b_save,
        )
        for i, b in enumerate(btns):
            b.grid(row=i, column=0, sticky="ew", pady=2)
        f.columnconfigure(0, weight=1)
        self.action_btns = [
            self.b_adb,
            self.b_calib,
            self.b_dcal,
            self.b_snap,
            self.b_test,
            self.b_assist,
            self.b_auto,
            self.b_save,
        ]

        # --- pestaña ajustes
        c = self.cfg
        self.v_n = tk.IntVar(value=c["n"])
        self.v_thr = tk.DoubleVar(value=c["thr"])
        self.v_thrp = tk.DoubleVar(value=c["thr_piece"])
        self.v_cell = tk.DoubleVar(value=c["piece_cell"] or 0)
        self.v_dx = tk.DoubleVar(value=c["drag_dx"])
        self.v_dy = tk.DoubleVar(value=c["drag_dy"])
        self.v_ms = tk.IntVar(value=c["ms"])
        self.v_pause = tk.DoubleVar(value=c["pause"])
        self.v_wait = tk.DoubleVar(value=c["wait"])
        o = self.opts
        self.v_interval = tk.DoubleVar(value=o["interval"])
        self.v_detect = tk.BooleanVar(value=o["detect"])
        self.v_step = tk.BooleanVar(value=o["step"])
        self.v_recalc = tk.BooleanVar(value=o["recalc"])
        self.v_reuse = tk.BooleanVar(value=o["reuse"])
        self.v_drag = tk.StringVar(value=o["drag"])
        self.v_grab = tk.StringVar(value=o["grab"])
        self.v_ax = tk.StringVar(value=c["drag_ax"])
        self.v_ay = tk.StringVar(value=c["drag_ay"])
        rows = [
            ("Tamaño tablero (n)", self.v_n, 4, 12, 1),
            ("Umbral color (thr)", self.v_thr, 5, 200, 1),
            ("Umbral piezas bandeja", self.v_thrp, 5, 120, 1),
            ("Cuadrito en bandeja (px)", self.v_cell, 0, 400, 1),
            ("Offset X ancla–dedo (px)", self.v_dx, -800, 800, 5),
            ("Offset Y ancla–dedo (px)", self.v_dy, -1200, 800, 5),
            ("Duración swipe (ms)", self.v_ms, 100, 2000, 50),
            ("Pausa tras cada pieza (s)", self.v_pause, 0.1, 6, 0.1),
            ("Espera nuevas piezas (s)", self.v_wait, 0.2, 8, 0.2),
            ("Refresco asistente (s)", self.v_interval, 0.5, 10, 0.5),
        ]
        for i, (t, v, a, b, inc) in enumerate(rows):
            ttk.Label(tab_s, text=t).grid(row=i, column=0, sticky="w")
            ttk.Spinbox(
                tab_s, from_=a, to=b, increment=inc, textvariable=v, width=8
            ).grid(row=i, column=1, sticky="e", pady=1)
        r0 = len(rows)
        combos = [
            ("Método de arrastre", self.v_drag, ("motionevent", "swipe"), 12),
            ("Punto de agarre en la bandeja", self.v_grab, GRAB_MODES, 26),
            (
                "Ancla X de la pieza flotante",
                self.v_ax,
                ("left", "center", "right"),
                12,
            ),
            (
                "Ancla Y de la pieza flotante",
                self.v_ay,
                ("top", "center", "bottom"),
                12,
            ),
        ]
        for j, (t, v, vals, wd) in enumerate(combos):
            ttk.Label(tab_s, text=t).grid(row=r0 + j, column=0, sticky="w")
            ttk.Combobox(
                tab_s, textvariable=v, values=vals, width=wd, state="readonly"
            ).grid(row=r0 + j, column=1, sticky="e", pady=1)
        r0 += len(combos)
        checks = [
            (
                "Mostrar detección (rejilla, formas y trayectoria del dedo)",
                self.v_detect,
            ),
            ("Recalcular tras cada pieza (recomendado)", self.v_recalc),
            ("Permitir reutilizar casillas de líneas completadas", self.v_reuse),
            ("Paso a paso (confirmar cada pieza)", self.v_step),
        ]
        for j, (t, v) in enumerate(checks):
            ttk.Checkbutton(tab_s, text=t, variable=v).grid(
                row=r0 + j, column=0, columnspan=2, sticky="w"
            )
        ttk.Label(
            tab_s,
            text="Todo se guarda solo (calibración y ajustes).",
            foreground="#2e7d32",
        ).grid(row=r0 + len(checks), column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(
            tab_s,
            text="Guardar ahora",
            command=lambda: (
                self.sync_cfg(),
                self.save_cfg(False),
                self.log(f"Guardado en {CFG_PATH}"),
            ),
        ).grid(
            row=r0 + len(checks) + 1, column=0, columnspan=2, sticky="ew", pady=(2, 0)
        )
        for v in (
            self.v_n,
            self.v_thr,
            self.v_thrp,
            self.v_cell,
            self.v_dx,
            self.v_dy,
            self.v_ms,
            self.v_pause,
            self.v_wait,
            self.v_interval,
            self.v_detect,
            self.v_step,
            self.v_recalc,
            self.v_reuse,
            self.v_drag,
            self.v_grab,
            self.v_ax,
            self.v_ay,
            self.v_serial,
        ):
            v.trace_add("write", self._on_var)
        tab_s.columnconfigure(0, weight=1)

        f = ttk.LabelFrame(
            right,
            text="Piezas detectadas (verde = contorno, cian = cuadritos, rojo = dudosa)",
            padding=4,
        )
        f.pack(fill="x", pady=(6, 0))
        self.lbl_pieces = ttk.Label(f)
        self.lbl_pieces.pack()
        self.pphoto = None

        f = ttk.LabelFrame(right, text="Registro", padding=4)
        f.pack(fill="both", expand=True, pady=6)
        self.txt = tk.Text(f, height=10, width=48, wrap="word", state="disabled")
        sb = ttk.Scrollbar(f, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        self.txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    # ---------------------------------------------------------- utilidades
    def ui(self, fn, *a):
        self.uiq.put((fn, a))

    def _poll(self):
        try:
            while True:
                fn, a = self.uiq.get_nowait()
                fn(*a)
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def log(self, msg):
        self.txt.configure(state="normal")
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def configured(self):
        c = self.cfg
        return bool(
            c.get("grid") and c.get("slots") and c.get("piece_cell") and c.get("empty")
        )

    def sync_cfg(self, quiet=False):
        try:
            c = self.cfg
            c["n"], c["thr"] = int(self.v_n.get()), float(self.v_thr.get())
            c["thr_piece"] = float(self.v_thrp.get())
            if float(self.v_cell.get()) > 0:
                c["piece_cell"] = float(self.v_cell.get())
            c["drag_dx"], c["drag_dy"] = float(self.v_dx.get()), float(self.v_dy.get())
            c["drag_ax"], c["drag_ay"], c["grab"] = (
                self.v_ax.get(),
                self.v_ay.get(),
                self.v_grab.get(),
            )
            c["ms"], c["pause"], c["wait"] = (
                int(self.v_ms.get()),
                float(self.v_pause.get()),
                float(self.v_wait.get()),
            )
            self.opts.update(
                detect=self.v_detect.get(),
                step=self.v_step.get(),
                interval=float(self.v_interval.get()),
                recalc=self.v_recalc.get(),
                reuse=self.v_reuse.get(),
                drag=self.v_drag.get(),
                grab=self.v_grab.get(),
            )
            self.adb.serial = self.v_serial.get().strip()
            return True
        except tk.TclError:
            if not quiet:
                messagebox.showwarning(
                    "Ajustes", "Hay un valor numérico inválido en los ajustes."
                )
            return False

    def sync_and(self, fn, *a, **k):
        self.sync_cfg()
        fn(*a, **k)

    def save_cfg(self, quiet=True):
        try:
            data = dict(self.cfg)
            data["opts"] = dict(self.opts)
            data["serial"] = self.adb.serial
            CFG_PATH.write_text(json.dumps(data, indent=2, default=float))
            return True
        except OSError as e:
            if not quiet:
                self.log(f"❌ No pude guardar {CFG_PATH}: {e}")
            return False

    def _on_var(self, *_):
        """Cualquier cambio en los ajustes se guarda solo (con 0,7 s de retardo)."""
        if self._save_job is not None:
            self.root.after_cancel(self._save_job)
        self._save_job = self.root.after(700, self.autosave)

    def autosave(self):
        self._save_job = None
        if self.sync_cfg(quiet=True):
            self.save_cfg()

    def set_busy(self, busy, loop=False):
        self.busy = busy
        for b in self.action_btns:
            b.configure(state="disabled" if busy else "normal")
        self.b_stop.configure(state="normal" if (busy and loop) else "disabled")
        self.b_next.configure(state="normal" if (busy and loop) else "disabled")

    def run_job(self, fn, *a, loop=False):
        if self.busy or self.calib:
            return
        self.set_busy(True, loop)

        def worker():
            try:
                fn(*a)
            except FileNotFoundError:
                self.ui(
                    self.log, "❌ No encuentro `adb`. Instálalo y ponlo en el PATH."
                )
            except subprocess.CalledProcessError as e:
                self.ui(
                    self.log,
                    "❌ ADB: " + (e.stderr or b"").decode(errors="ignore").strip(),
                )
            except subprocess.TimeoutExpired:
                self.ui(self.log, "❌ ADB tardó demasiado en responder.")
            except Exception as e:  # noqa
                self.ui(self.log, f"❌ {type(e).__name__}: {e}")
            finally:
                self.ui(self.set_busy, False)

        threading.Thread(target=worker, daemon=True).start()

    # ---------------------------------------------------------- imagen
    def show(self, img):
        self.annot = img
        h, w = img.shape[:2]
        maxh = int(self.root.winfo_screenheight() * 0.82)
        self.scale = min(maxh / h, 560 / w, 1.0)
        disp = cv2.resize(
            img,
            (int(w * self.scale), int(h * self.scale)),
            interpolation=cv2.INTER_AREA,
        )
        self.photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        )
        self.canvas.configure(width=disp.shape[1], height=disp.shape[0])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def show_pieces(self, prev):
        h, w = prev.shape[:2]
        if w > 520:
            prev = cv2.resize(
                prev, (520, int(h * 520 / w)), interpolation=cv2.INTER_AREA
            )
        self.pphoto = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB))
        )
        self.lbl_pieces.configure(image=self.pphoto)

    def show_res(self, res):
        self.last_res = res
        self.show(res["annot"])
        self.show_pieces(res["preview"])
        if res.get("cleared"):
            self.log(
                f"Forma manual descartada en pieza(s) {[i + 1 for i in res['cleared']]}: la pieza cambió."
            )
        if self.editor is not None and self.editor.winfo_exists():
            self._editor_redraw()

    def save_image(self):
        if self.annot is not None:
            cv2.imwrite("overlay.png", self.annot)
            self.log("Imagen guardada en overlay.png")

    # ---------------------------------------------------------- análisis
    def analyze(self, img, do_plan=True):
        cfg = self.cfg
        cleared = []
        board, pieces = (
            read_board(img, cfg),
            read_pieces(img, cfg, self.overrides, cleared),
        )
        moves = []
        if do_plan and any(pieces):
            moves = plan_best(board, pieces, cfg["n"], self.opts["reuse"])
        annot = render(img, cfg, board, pieces, moves, self.opts["detect"])
        return dict(
            board=board,
            pieces=pieces,
            moves=moves,
            annot=annot,
            preview=piece_preview(img, cfg, pieces),
            cleared=cleared,
            doubtful=[i + 1 for i, p in enumerate(pieces) if p and not p["ok"]],
        )

    def describe(self, res):
        if not any(res["pieces"]):
            return "No detecto piezas en las ranuras."
        if not res["moves"]:
            return "No hay jugada posible (¿fin de partida?)."
        txt = "  ".join(
            f"#{k + 1}: pieza {i + 1} → fila {r + 1}, col {c + 1}"
            for k, (i, r, c) in enumerate(res["moves"])
        )
        total = sum(1 for p in res["pieces"] if p)
        if len(res["moves"]) < total:
            txt += f"\n⚠ Solo caben {len(res['moves'])} de {total} piezas (fin de partida cercano)."
        if res["doubtful"]:
            txt += f"\n⚠ Forma dudosa en pieza(s) {res['doubtful']}: revisa «Piezas detectadas»."
        return txt

    # ---------------------------------------------------------- arrastre
    def do_drag(self, sx, sy, tx, ty):
        if self.opts["drag"] == "motionevent" and not self.no_motion:
            try:
                self.adb.drag_motion(sx, sy, tx, ty)
                return
            except (RuntimeError, subprocess.CalledProcessError):
                self.no_motion = True
                self.ui(
                    self.log,
                    "⚠ `input motionevent` no disponible (requiere Android 10+). Uso swipe.",
                )
        self.adb.swipe(sx, sy, tx, ty, self.cfg["ms"])

    def execute_move(self, i, p, r, c):
        tx, ty, _, _ = target_point(self.cfg, p, r, c)
        sx, sy = grab_point(self.cfg, p, self.cfg["slots"][i])
        self.ui(self.log, f"   dedo: ({int(sx)},{int(sy)}) → ({int(tx)},{int(ty)})")
        self.do_drag(sx, sy, tx, ty)

    # ---------------------------------------------------------- trabajos
    def job_devices(self):
        self.ui(self.log, "adb devices:\n" + self.adb.devices().strip())

    def job_capture(self):
        img = self.adb.screenshot()
        self.raw = img
        if not self.configured():
            self.ui(self.show, img)
            self.ui(self.log, "Captura tomada. Calibra para obtener sugerencias.")
            return
        res = self.analyze(img)
        self.ui(self.show_res, res)
        self.ui(self.log, self.describe(res))

    def job_test_move(self):
        if not self.configured():
            self.ui(self.log, "Primero calibra.")
            return
        img = self.adb.screenshot()
        res = self.analyze(img)
        self.ui(self.show_res, res)
        if not res["moves"]:
            self.ui(self.log, self.describe(res))
            return
        i, r, c = res["moves"][0]
        self.ui(self.log, f"Probando: pieza {i + 1} → fila {r + 1}, col {c + 1}")
        self.execute_move(i, res["pieces"][i], r, c)
        self.sleep(self.cfg["pause"] + 0.6)
        img2 = self.adb.screenshot()
        self.raw = img2
        res2 = self.analyze(img2)
        self.ui(self.show_res, res2)
        self.ui(
            self.log,
            "Resultado en pantalla. Si no cayó donde indicaba, usa «Calibrar arrastre».",
        )

    def sleep(self, t):
        return self.stop_ev.wait(t) if t > 0 else False

    def start_loop(self, play):
        if self.busy or self.calib:
            return
        if not self.configured():
            messagebox.showinfo(
                "Calibración", "Primero calibra el tablero y las piezas."
            )
            return
        if play and not messagebox.askokcancel(
            "Jugar automáticamente",
            "El script enviará toques al móvil.\nAsegúrate de estar en la pantalla de juego.",
        ):
            return
        self.sync_cfg()
        self.stop_ev.clear()
        self.step_ev.clear()
        self.run_job(self._loop, play, loop=True)

    def stop_loop(self):
        self.stop_ev.set()
        self.step_ev.set()
        self.log("Deteniendo…")

    def _loop(self, play):
        cfg, o = self.cfg, self.opts
        prev_sig, last_log_sig, pending = None, None, None
        no_pieces, bad, doubt = 0, 0, 0
        self.ui(self.log, "▶ Auto iniciado." if play else "👁 Asistente iniciado.")
        while not self.stop_ev.is_set():
            img = self.adb.screenshot()
            self.raw = img
            res = self.analyze(img, do_plan=False)
            pieces = res["pieces"]
            if not any(pieces):
                self.ui(self.show_res, res)
                no_pieces += 1
                prev_sig = None
                if no_pieces >= 8:
                    self.ui(self.log, "No veo piezas varias veces seguidas. Paro.")
                    break
                self.sleep(1.0)
                continue
            no_pieces = 0
            if play and res["doubtful"]:  # no se actúa con una forma dudosa
                doubt += 1
                prev_sig = None
                self.ui(self.show_res, res)
                if doubt == 1:
                    self.ui(
                        self.log,
                        f"⚠ Forma dudosa en pieza(s) {res['doubtful']}; espero a que se estabilice…",
                    )
                if doubt >= 12:
                    self.ui(
                        self.log,
                        "La detección sigue dudosa. Ajusta «Umbral piezas bandeja» / «Cuadrito en bandeja».",
                    )
                    break
                self.sleep(0.6)
                continue
            doubt = 0
            sig = (
                res["board"],
                tuple(tuple(p["masks"]) if p else None for p in pieces),
            )
            if (
                play and sig != prev_sig
            ):  # esperar a que la pantalla esté estable (animaciones)
                prev_sig = sig
                self.ui(self.show_res, res)
                self.sleep(0.4)
                continue
            if pending:  # ¿la pieza anterior realmente se movió?
                pi, pboard, pmasks = pending
                pending = None
                p = pieces[pi]
                if res["board"] == pboard and p and tuple(p["masks"]) == pmasks:
                    bad += 1
                    self.ui(self.log, f"⚠ La pieza {pi + 1} no se movió ({bad}/3).")
                    if bad >= 3:
                        self.ui(
                            self.log,
                            "No logra arrastrar. Revisa método de arrastre / calibrar arrastre.",
                        )
                        break
                else:
                    bad = 0
            res = self.analyze(img, do_plan=True)
            moves = res["moves"]
            self.ui(self.show_res, res)
            if not moves:
                self.ui(self.log, "Sin jugadas posibles: fin de partida.")
                break
            if not play:
                if sig != last_log_sig:
                    self.ui(self.log, self.describe(res))
                    last_log_sig = sig
                self.sleep(o["interval"])
                continue
            self.ui(self.log, self.describe(res))
            todo = moves[:1] if o["recalc"] else moves
            for i, r, c in todo:
                if self.stop_ev.is_set():
                    break
                self.execute_move(i, pieces[i], r, c)
                if o["recalc"]:
                    pending = (i, res["board"], tuple(pieces[i]["masks"]))
                self.sleep(cfg["pause"])
                if o["step"] and not self.stop_ev.is_set():
                    self.step_ev.clear()
                    self.ui(self.log, "Pulsa «Continuar».")
                    while not self.step_ev.wait(0.2) and not self.stop_ev.is_set():
                        pass
            prev_sig = None
            if not o["recalc"]:
                self.sleep(cfg["wait"])
        self.ui(self.log, "■ Detenido.")

    # ---------------------------------------------------------- calibración (genérica)
    def _begin_calib(self, kind, steps, img, shown=None):
        self.raw = img
        self.show(shown if shown is not None else img)
        self.calib = dict(kind=kind, steps=steps, i=0, res={})
        self.calib_frame.pack(fill="x", before=self.actions_frame)
        self._calib_prompt()

    def _calib_prompt(self):
        cb = self.calib
        self.calib_lbl.configure(
            text=f"Paso {cb['i'] + 1}/{len(cb['steps'])}: {cb['steps'][cb['i']][1]}"
        )

    def on_press(self, e):
        if not self.calib:
            return
        self.canvas.delete("sel")
        self.calib["p0"] = (e.x, e.y)
        self.calib.pop("p1", None)
        self.calib["id"] = self.canvas.create_rectangle(
            e.x, e.y, e.x, e.y, outline="#00ff66", width=2, tags="sel"
        )

    def on_drag(self, e):
        if self.calib and "id" in self.calib and "p0" in self.calib:
            x0, y0 = self.calib["p0"]
            self.canvas.coords(self.calib["id"], x0, y0, e.x, e.y)

    def on_release(self, e):
        if self.calib and "p0" in self.calib:
            self.calib["p1"] = (e.x, e.y)

    def calib_accept(self):
        cb = self.calib
        if not cb or "p0" not in cb or "p1" not in cb:
            messagebox.showinfo(
                "Calibración", "Primero arrastra un rectángulo sobre la imagen."
            )
            return
        (x0, y0), (x1, y1) = cb["p0"], cb["p1"]
        H, W = self.raw.shape[:2]
        s = self.scale
        X0, X1 = sorted((max(0, min(W, x0 / s)), max(0, min(W, x1 / s))))
        Y0, Y1 = sorted((max(0, min(H, y0 / s)), max(0, min(H, y1 / s))))
        w, h = int(X1 - X0), int(Y1 - Y0)
        if w < 5 or h < 5:
            messagebox.showwarning("Calibración", "La selección es demasiado pequeña.")
            return
        cb["res"][cb["steps"][cb["i"]][0]] = [int(X0), int(Y0), w, h]
        self.canvas.delete("sel")
        self.canvas.create_rectangle(
            x0, y0, x1, y1, outline="#ffaa00", width=2, tags="done"
        )
        cb["i"] += 1
        for k in ("p0", "p1", "id"):
            cb.pop(k, None)
        if cb["i"] < len(cb["steps"]):
            self._calib_prompt()
        else:
            kind, res = cb["kind"], cb["res"]
            self.calib = None
            self.calib_frame.pack_forget()
            (self._finish_board_calib if kind == "board" else self._finish_drag_calib)(
                res
            )

    def calib_cancel(self):
        kind = self.calib["kind"] if self.calib else None
        self.calib = None
        self.calib_frame.pack_forget()
        if self.raw is not None:
            self.show(self.raw)
        self.log("Calibración cancelada.")
        if kind == "drag":
            self.run_job(self._drag_cancel)

    # ---------------------------------------------------------- calibración tablero
    def start_calib(self):
        if self.busy or self.calib:
            return
        if not messagebox.askokcancel(
            "Calibración",
            "Hazlo con el tablero casi vacío (inicio de partida) y las 3 piezas visibles.\n"
            "Se tomará una captura nueva.",
        ):
            return
        self.sync_cfg()
        self.run_job(
            lambda: self.ui(
                self._begin_calib, "board", self.BOARD_STEPS, self.adb.screenshot()
            )
        )

    def _finish_board_calib(self, res):
        cfg = self.cfg
        cfg["grid"] = res["grid"]
        cfg["slots"] = [res["slot0"], res["slot1"], res["slot2"]]
        cfg["piece_cell"] = (res["cell"][2] + res["cell"][3]) / 2
        par = [[], []]
        for r in range(cfg["n"]):
            for c in range(cfg["n"]):
                par[(r + c) % 2].append(cell_patch(self.raw, cfg, r, c))
        cfg["empty"] = [np.median(p, axis=0).tolist() for p in par]
        self.v_cell.set(round(cfg["piece_cell"], 1))
        self.save_cfg()
        self.log(
            "✔ Calibración guardada. Ahora «Calibrar arrastre» y revisa con «Mostrar detección»."
        )
        self.sync_cfg()
        self.run_job(self._analyze_existing)

    def _analyze_existing(self):
        res = self.analyze(self.raw)
        self.ui(self.show_res, res)
        self.ui(self.log, self.describe(res))

    # ---------------------------------------------------------- calibración de arrastre
    def start_drag_calib(self):
        if self.busy or self.calib:
            return
        if not self.configured():
            messagebox.showinfo(
                "Calibración", "Primero calibra el tablero y las piezas."
            )
            return
        if not messagebox.askokcancel(
            "Calibrar arrastre",
            "La app agarrará cada pieza de la bandeja (una por cada tamaño distinto, máx. 3), la llevará "
            "al centro del tablero y la mantendrá ahí mientras tomas la captura.\n"
            "En cada una marcas la pieza flotante y la app la devuelve a su sitio.\n\n"
            "Con piezas de distinta altura/anchura la app deduce solo a qué punto de la pieza "
            "(arriba, centro, abajo…) se ancla el dedo.",
        ):
            return
        self.sync_cfg()
        self.run_job(self._drag_calib_begin)

    def _drag_calib_begin(self):
        img = self.adb.screenshot()
        pieces = read_pieces(img, self.cfg, self.overrides)
        seen, todo = set(), []
        for k, p in enumerate(pieces):
            if p and (p["rows"], p["cols"]) not in seen:
                seen.add((p["rows"], p["cols"]))
                todo.append(k)
        if not todo:
            self.ui(self.log, "No detecto ninguna pieza en la bandeja.")
            return
        self.dc = dict(todo=todo, total=len(todo), samples=[], pieces=pieces)
        self._drag_calib_hold()

    def _drag_calib_hold(self):
        cfg, dc = self.cfg, self.dc
        k = dc["todo"].pop(0)
        p = dc["pieces"][k]
        sx, sy = grab_point(cfg, p, cfg["slots"][k])
        gx, gy, gw, gh = cfg["grid"]
        tx, ty = gx + gw / 2, gy + gh / 2
        try:
            self.adb.motion(self.adb.path(sx, sy, tx, ty) + [("sleep", 0.6)])
        except RuntimeError:
            self.ui(
                self.log,
                "Tu Android no soporta `input motionevent` (Android 10+). "
                "Ajusta los offsets a mano con «Probar solo la 1ª pieza».",
            )
            return
        try:
            img2 = self.adb.screenshot()
        except Exception:
            self.adb.motion([("MOVE", sx, sy), ("UP", sx, sy)])
            raise
        marked = img2.copy()
        cv2.drawMarker(marked, (int(tx), int(ty)), (0, 0, 255), cv2.MARKER_CROSS, 60, 5)
        self.drag_state = dict(start=(sx, sy), finger=(tx, ty), piece=p)
        steps = [
            (
                "float",
                f"Muestra {len(dc['samples']) + 1}/{dc['total']} (pieza {k + 1}, {p['rows']}x{p['cols']}): "
                "la cruz roja es tu dedo y sostiene la pieza. Rodea SOLO la pieza flotante "
                "(sus cuadritos) y pulsa «Aceptar».",
            )
        ]
        self.ui(self._begin_calib, "drag", steps, img2, marked)

    def _release_job(self):
        ds = self.drag_state
        self.drag_state = None
        if ds:
            sx, sy = ds["start"]
            self.adb.motion([("MOVE", sx, sy), ("sleep", 0.25), ("UP", sx, sy)])

    def _finish_drag_calib(self, res):
        x, y, w, h = res["float"]
        ds = self.drag_state
        fx, fy = ds["finger"]
        p = ds["piece"]
        self.dc["samples"].append(
            dict(x=x, y=y, w=w, h=h, fx=fx, fy=fy, rows=p["rows"], cols=p["cols"])
        )
        self.show(self.raw)
        self.run_job(self._drag_next)

    def _drag_next(self):
        self._release_job()
        if self.dc["todo"]:
            self._drag_calib_hold()
        else:
            self.ui(self._drag_solve)

    def _drag_cancel(self):
        self._release_job()
        if self.dc and self.dc["samples"]:
            self.ui(self._drag_solve)

    def _drag_solve(self):
        cfg, S = self.cfg, self.dc["samples"]
        if not S:
            return
        bw = cfg["grid"][2] / cfg["n"]
        xs = {
            "left": [q["x"] - q["fx"] for q in S],
            "center": [q["x"] + q["w"] / 2 - q["fx"] for q in S],
            "right": [q["x"] + q["w"] - q["fx"] for q in S],
        }
        ys = {
            "top": [q["y"] - q["fy"] for q in S],
            "center": [q["y"] + q["h"] / 2 - q["fy"] for q in S],
            "bottom": [q["y"] + q["h"] - q["fy"] for q in S],
        }
        ax, dx, spx = choose_anchor(xs, bw)
        ay, dy, spy = choose_anchor(ys, bw)
        cfg["drag_ax"], cfg["drag_ay"] = ax, ay
        cfg["drag_dx"], cfg["drag_dy"] = round(dx, 1), round(dy, 1)
        self.v_ax.set(ax)
        self.v_ay.set(ay)
        self.v_dx.set(cfg["drag_dx"])
        self.v_dy.set(cfg["drag_dy"])
        scale = sum(q["w"] / (q["cols"] * bw) for q in S) / len(S)
        self.log(
            f"✔ Arrastre calibrado con {len(S)} muestra(s): ancla X={ax} (offset {cfg['drag_dx']} px), "
            f"ancla Y={ay} (offset {cfg['drag_dy']} px). Pieza flotante ≈ {scale:.2f}× una casilla."
        )
        if len({(q["rows"]) for q in S}) < 2:
            self.log(
                "ℹ Todas las muestras tenían la misma altura: no se puede distinguir el ancla vertical "
                "(se usa «center»). Si las piezas altas/bajas se desvían distinto, repite con piezas "
                "de distinta altura."
            )
        if abs(scale - 1) > 0.12:
            self.log(
                "ℹ La pieza flotante no tiene el tamaño de una casilla del tablero; si hay desvío "
                "residual, prueba el ancla «center» o ajusta el offset a mano."
            )
        self.save_cfg()

    # ---------------------------------------------------------- editor de formas
    ED_CELL = 38

    def reanalyze(self):
        if (
            self.raw is not None
            and self.configured()
            and not self.busy
            and not self.calib
        ):
            self.sync_cfg()
            self.run_job(self._analyze_existing)

    def open_editor(self):
        if self.editor is not None and self.editor.winfo_exists():
            self.editor.lift()
            return
        top = tk.Toplevel(self.root)
        top.title("Editar formas de las piezas")
        self.editor = top
        self.v_gn = tk.IntVar(value=int(self.cfg.get("editor_grid", 4)))
        bar = ttk.Frame(top, padding=6)
        bar.pack(fill="x")
        ttk.Label(bar, text="Cuadrícula:").pack(side="left")
        sp = ttk.Spinbox(
            bar,
            from_=3,
            to=6,
            width=4,
            textvariable=self.v_gn,
            command=self._editor_build,
        )
        sp.pack(side="left", padx=4)
        sp.bind("<Return>", lambda e: self._editor_build())
        ttk.Label(
            bar,
            text="Clic en un cuadrito = corregir la forma (pasa a modo manual).\n"
            "La forma manual se descarta sola cuando cambia la pieza de esa ranura.",
            justify="left",
        ).pack(side="left", padx=10)
        self.ed_body = ttk.Frame(top, padding=6)
        self.ed_body.pack()
        self.ed_widgets = []
        self._editor_build()

    def _editor_build(self):
        for w in self.ed_body.winfo_children():
            w.destroy()
        try:
            n = max(3, min(6, int(self.v_gn.get())))
        except tk.TclError:
            n = 4
        self.ed_widgets = []
        self.cfg["editor_grid"] = n
        self._on_var()
        for i in range(3):
            f = ttk.LabelFrame(self.ed_body, text=f"Pieza {i + 1}", padding=6)
            f.grid(row=0, column=i, padx=6)
            cv = tk.Canvas(
                f,
                width=n * self.ED_CELL,
                height=n * self.ED_CELL,
                bg="#1e1e1e",
                highlightthickness=1,
            )
            cv.pack()
            cv.bind("<Button-1>", lambda e, i=i: self._editor_click(i, e))
            st = ttk.Label(f, text="", width=26)
            st.pack(pady=3)
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                f,
                text="Forma manual",
                variable=var,
                command=lambda i=i: self._editor_toggle(i),
            ).pack(anchor="w")
            bf = ttk.Frame(f)
            bf.pack(fill="x", pady=2)
            ttk.Button(
                bf, text="Cargar detectada", command=lambda i=i: self._editor_load(i)
            ).pack(side="left")
            ttk.Button(
                bf, text="Limpiar", command=lambda i=i: self._editor_clear(i)
            ).pack(side="left", padx=4)
            self.ed_widgets.append(dict(cv=cv, st=st, var=var, n=n))
        self._editor_redraw()

    def _editor_piece(self, i):
        return self.last_res["pieces"][i] if self.last_res else None

    def _editor_ensure(self, i):
        if i in self.overrides:
            return True
        p = self._editor_piece(i)
        if not p:
            messagebox.showinfo(
                "Editor",
                f"No hay una pieza detectada en la ranura {i + 1}.\n"
                "Toma una captura con la pieza visible.",
            )
            return False
        self.overrides[i] = dict(cells=set(p["auto"]["cells"]), sig=p["auto_sig"])
        return True

    def _editor_click(self, i, e):
        w = self.ed_widgets[i]
        r, c = e.y // self.ED_CELL, e.x // self.ED_CELL
        if not (0 <= r < w["n"] and 0 <= c < w["n"]) or not self._editor_ensure(i):
            return
        self.overrides[i]["cells"] ^= {(r, c)}
        self._editor_redraw()
        self.reanalyze()

    def _editor_toggle(self, i):
        if self.ed_widgets[i]["var"].get():
            if not self._editor_ensure(i):
                self.ed_widgets[i]["var"].set(False)
        else:
            self.overrides.pop(i, None)
        self._editor_redraw()
        self.reanalyze()

    def _editor_load(self, i):
        p = self._editor_piece(i)
        if p and self._editor_ensure(i):
            self.overrides[i]["cells"] = set(p["auto"]["cells"])
            self._editor_redraw()
            self.reanalyze()

    def _editor_clear(self, i):
        if self._editor_ensure(i):
            self.overrides[i]["cells"] = set()
            self._editor_redraw()
            self.reanalyze()

    def _editor_redraw(self):
        for i, w in enumerate(self.ed_widgets):
            cv, n, k = w["cv"], w["n"], self.ED_CELL
            ov, p = self.overrides.get(i), self._editor_piece(i)
            if ov:
                cells, color = set(ov["cells"]), "#2ecc71"
                if not cells:
                    txt = "MANUAL: dibuja la forma"
                else:
                    rr = max(r for r, _ in cells) - min(r for r, _ in cells) + 1
                    cc = max(c for _, c in cells) - min(c for _, c in cells) + 1
                    ok = p and p["ok"]
                    txt = f"MANUAL {rr}x{cc}" + ("" if ok else "  (¡no conexa!)")
            elif p:
                cells, color = set(p["auto"]["cells"]), "#5dade2"
                a = p["auto"]
                txt = f"Auto {a['rows']}x{a['cols']}" + (
                    "" if p["ok"] else "  (dudosa)"
                )
                if a["rows"] > n or a["cols"] > n:
                    txt += " — mayor que la cuadrícula"
            else:
                cells, color, txt = set(), "#5dade2", "Sin pieza detectada"
            w["var"].set(bool(ov))
            w["st"].configure(text=txt)
            cv.delete("all")
            for r in range(n):
                for c in range(n):
                    on = (r, c) in cells
                    cv.create_rectangle(
                        c * k + 2,
                        r * k + 2,
                        (c + 1) * k - 2,
                        (r + 1) * k - 2,
                        fill=color if on else "#2b2b2b",
                        outline="#ffffff" if on else "#555555",
                    )

    # ---------------------------------------------------------- cierre
    def on_close(self):
        self.stop_ev.set()
        self.step_ev.set()
        self.sync_cfg(quiet=True)
        self.save_cfg()
        if self.drag_state:
            try:
                self._release_job()
            except Exception:
                pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
