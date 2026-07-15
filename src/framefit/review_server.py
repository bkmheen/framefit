"""Interactive review front (Module B): a local browser page to confirm or edit
the auto-detected slide corners, one image at a time.

Why a local ``http.server`` + the default browser rather than a native GUI: the
shipped dependency is ``opencv-python-headless`` (no highgui — ``cv2.imshow``
raises), and macOS Tk/Cocoa GUIs are exactly the flakiness this must avoid. The
browser is the most stable image surface on a Mac, needs zero extra dependencies,
and makes drag-handles + accept/edit + POST-back trivial. The hard parts
(HEIC→JPEG transcode, display↔full-res coordinate mapping) were already solved in
``picker.py`` and are reused here.

Flow: ``A`` = :func:`pipeline.process_image` proposes corners + confidence; this
server shows them as draggable handles; the human accepts / edits / skips; ``C`` =
:func:`pipeline.process_manual` warps the confirmed corners; :mod:`feedback` logs
the decision and copies the original+crop into the shared dataset.
"""
from __future__ import annotations

import base64
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import cv2
import numpy as np

from . import feedback, io
from .backends import get_backend
from .geometry import aspect_score_wh, order_corners
from .pipeline import process_image, process_manual


class ReviewState:
    """Queue + per-image preparation. Thread-safe (single-client, but the
    threading server may overlap requests)."""

    def __init__(self, files, outdir, backend="auto", display_max=1400,
                 review_threshold=0.90, out_format="jpg", quality=95,
                 skip_decided=True, beside=False, force=False,
                 only_flagged=False):
        self.outdir = Path(outdir)
        self.display_max = display_max
        self.review_threshold = review_threshold
        self.out_format = out_format
        self.quality = quality
        self.beside = beside
        self.force = force
        self.only_flagged = only_flagged
        self.auto_accepted = 0
        self._lock = threading.RLock()
        self._backend = get_backend(backend)   # instantiate once (loads model)
        self._prepared: Optional[dict] = None   # cache for current index
        self.done_count = 0
        self.skipped_count = 0

        decided = feedback.decided_hashes() if skip_decided else set()
        self._pre_decided = 0
        queue = []
        for f in files:
            queue.append(Path(f))
        self.files = queue
        self.total = len(queue)
        self.decided = decided
        self.idx = 0
        self._advance_to_undecided()
        # forward frontier vs. currently-displayed idx: they match during normal
        # forward review. After a "이전으로" (back) jump, idx < frontier and we are
        # re-editing an already-confirmed image; saving/skipping a revisit returns
        # to the frontier instead of advancing past it.
        self.frontier = self.idx
        self.session_decided = []   # queue indices decided this session, in order
        self.session_actions = {}   # idx -> "save" | "skip"
        self.saved_full = {}        # idx -> final full-res quad (None if skipped)

    # -- queue movement --------------------------------------------------- #
    def _advance_to_undecided(self):
        """Skip images already decided in a prior run (resume-safety)."""
        while self.idx < len(self.files):
            try:
                sha = feedback.sha1_of_file(self.files[self.idx])
            except OSError:
                self.idx += 1
                continue
            if sha in self.decided:
                self._pre_decided += 1
                self.idx += 1
                continue
            break
        self._prepared = None

    def _prepare_current(self) -> Optional[dict]:
        """Load + detect the current image; cache the display jpeg, auto quad
        (in display coords) and confidence. Returns None when the queue is done."""
        with self._lock:
            if self._prepared is not None:
                return self._prepared
            while self.idx < len(self.files):
                path = self.files[self.idx]
                try:
                    bgr = io.load_bgr(path)
                except Exception as e:  # HEIC-without-extra, unreadable, etc.
                    # Don't kill the queue — record nothing, just move on.
                    self._error = f"{path.name}: {e}"
                    self.idx += 1
                    continue
                fh, fw = bgr.shape[:2]
                scale = (self.display_max / max(fh, fw)
                         if max(fh, fw) > self.display_max else 1.0)
                disp = (cv2.resize(bgr, (int(fw * scale), int(fh * scale)),
                                   interpolation=cv2.INTER_AREA)
                        if scale < 1.0 else bgr)
                ok, buf = cv2.imencode(".jpg", disp,
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64 = base64.b64encode(buf.tobytes()).decode()

                res = process_image(bgr, backend=self._backend,
                                    expand=0.0, inset=0.0, refine=False)
                if res.ok and res.quad is not None:
                    q = order_corners(np.asarray(res.quad, dtype=np.float32))
                    auto_disp = [[round(float(x * scale), 2),
                                  round(float(y * scale), 2)] for x, y in q]
                    auto_full = [[float(x), float(y)] for x, y in q]
                    backend_name = res.backend
                    low_conf = res.low_confidence
                    aspect = res.aspect_score
                else:
                    auto_disp = None
                    auto_full = None
                    backend_name = res.backend
                    low_conf = True
                    aspect = 0.0

                verdict = feedback.confidence_verdict(
                    low_conf, aspect, auto_full is not None,
                    self.review_threshold)
                prep = {
                    "path": path,
                    "bgr": bgr,
                    "sha1": feedback.sha1_of_file(path),
                    "width": fw, "height": fh,
                    "disp_w": disp.shape[1], "disp_h": disp.shape[0],
                    "scale": scale, "inv": (1.0 / scale) if scale else 1.0,
                    "image_b64": b64,
                    "auto_disp": auto_disp,
                    "auto_full": auto_full,
                    "backend": backend_name,
                    "low_confidence": low_conf,
                    "aspect_score": aspect,
                    "detect_score": res.detect_score if res.ok else None,
                    "verdict": verdict,
                }
                # --only-flagged: silently auto-accept confident detections so the
                # interface only surfaces the problematic ones. Forward-only: never
                # auto-accept an image being revisited via "이전으로".
                if (self.only_flagged and verdict["level"] == "good"
                        and auto_full is not None
                        and self.idx == self.frontier
                        and self.idx not in self.session_actions):
                    idx0 = self.idx
                    full = order_corners(np.asarray(auto_full, dtype=np.float32))
                    # presented=False / was_auto_accepted=True: the human was never
                    # asked. If they later step back into this image and change it,
                    # that revision is the "under_flag" signal.
                    res = self._write_crop_and_log(
                        prep, full, "accept", allow_overwrite=self.force,
                        meta=self._decision_meta(prep, presented=False,
                                                 was_auto_accepted=True,
                                                 revised=False, idx0=idx0))
                    if res.get("ok"):
                        # register in the session history so '◀ 이전' can reach and
                        # correct a wrongly auto-accepted image
                        self.saved_full[idx0] = [[float(x), float(y)] for x, y in full]
                        self._record_action(idx0, "auto")
                    self.idx = idx0 + 1          # advance regardless — never loop
                    self._advance_to_undecided()
                    self.frontier = self.idx
                    continue
                self._prepared = prep
                return self._prepared
            self._prepared = None
            return None

    def _can_back(self) -> bool:
        """True when some image decided this session sits before the current one —
        i.e. there is a previous confirmed image to return to."""
        return any(i < self.idx for i in self.session_decided)

    def state_json(self) -> dict:
        prep = self._prepare_current()
        if prep is None:
            return {
                "done": True,
                "total": self.total,
                "decided": self.done_count,
                "skipped": self.skipped_count,
                "pre_decided": self._pre_decided,
                "auto_accepted": self.auto_accepted,
                "can_back": self._can_back(),
            }
        # When revisiting an already-confirmed image, reload the corners the human
        # last set (full-res → display coords) so they can be tweaked, not redone.
        saved = self.saved_full.get(self.idx)
        prev_disp = None
        if saved:
            sc = prep["scale"]
            prev_disp = [[round(float(x * sc), 2), round(float(y * sc), 2)]
                         for x, y in saved]
        return {
            "done": False,
            "index": self.idx,
            "total": self.total,
            "pre_decided": self._pre_decided,
            "name": prep["path"].name,
            "image_b64": prep["image_b64"],
            "disp_w": prep["disp_w"], "disp_h": prep["disp_h"],
            "auto_disp": prep["auto_disp"],
            "prev_disp": prev_disp,
            "revisit": self.idx in self.session_actions,
            "can_back": self._can_back(),
            "backend": prep["backend"],
            "low_confidence": prep["low_confidence"],
            "aspect_score": round(float(prep["aspect_score"]), 3),
            "verdict": prep["verdict"],
        }

    def decide(self, action: str, corners_disp) -> dict:
        """Apply a decision for the current image, log it, advance. ``corners_disp``
        are 4 [x,y] in DISPLAY coords (ignored for skip/back). ``action`` "back"
        returns to the previous confirmed image — with its saved corners reloaded —
        for re-editing."""
        with self._lock:
            if action == "back":
                return self._go_back()
            prep = self._prepare_current()
            if prep is None:
                return {"ok": False, "error": "queue empty"}

            idx0 = self.idx
            revisit = idx0 < self.frontier
            inv = prep["inv"]
            auto_full = prep["auto_full"]
            # classification context — captured BEFORE _record_action overwrites the
            # prior action, so a revision of an auto-accept is recorded as such.
            meta = self._decision_meta(prep, presented=True, was_auto_accepted=False,
                                       revised=revisit, idx0=idx0)

            if action == "skip":
                orig_rel, _ = feedback.store_dataset(prep["sha1"], prep["bgr"], None)
                rec = feedback.build_record(
                    source_path=prep["path"], source_sha1=prep["sha1"],
                    image_width=prep["width"], image_height=prep["height"],
                    backend=prep["backend"],
                    auto_low_confidence=prep["low_confidence"],
                    auto_aspect_score=prep["aspect_score"],
                    auto_detect_score=prep.get("detect_score"),
                    auto_quad=auto_full, display_scale=prep["scale"],
                    final_quad=None, action="skip",
                    output_path=None, dataset_original=orig_rel, dataset_crop=None,
                    **meta)
                feedback.append_record(rec)
                self.decided.add(prep["sha1"])
                self.saved_full[idx0] = None
                self._record_action(idx0, "skip")
                self._navigate_after(idx0, revisit)
                return {"ok": True, "action": "skip"}

            # save: map display corners -> full-res
            if not corners_disp or len(corners_disp) != 4:
                return {"ok": False, "error": "need 4 corners"}
            full = order_corners(np.asarray(
                [[float(x) * inv, float(y) * inv] for x, y in corners_disp],
                dtype=np.float32))

            # accept vs modify vs manual: compare against auto (server decides)
            if auto_full is None:
                action_kind = "manual_from_scratch"
            else:
                auto_ord = order_corners(np.asarray(auto_full, dtype=np.float32))
                max_delta = float(np.linalg.norm(full - auto_ord, axis=1).max())
                action_kind = "accept" if max_delta <= 1.0 else "modify"

            # A revisit save always overwrites the crop it wrote earlier this
            # session, regardless of --force.
            res = self._write_crop_and_log(prep, full, action_kind,
                                           allow_overwrite=(revisit or self.force),
                                           meta=meta)
            if not res.get("ok"):
                return res
            self.saved_full[idx0] = [[float(x), float(y)] for x, y in full]
            self._record_action(idx0, "save")
            self._navigate_after(idx0, revisit)
            return res

    def _decision_meta(self, prep, *, presented, was_auto_accepted, revised, idx0):
        """The review-signal context for one decision, read from the current prep +
        the prior session action for ``idx0`` (call BEFORE overwriting it)."""
        prior = self.session_actions.get(idx0)
        v = prep["verdict"]
        return dict(
            verdict_level=v["level"], verdict_reason=v.get("reason", ""),
            presented=presented, was_auto_accepted=was_auto_accepted,
            revised=revised, prior_action=prior,
            prior_was_auto_accepted=(prior == "auto"),
        )

    # -- decision bookkeeping / navigation -------------------------------- #
    def _record_action(self, idx, kind):
        """Remember this session's decision for ``idx`` and recompute the visible
        counters (idempotent, so re-deciding a revisited image never double-counts)."""
        self.session_actions[idx] = kind
        if idx not in self.session_decided:
            self.session_decided.append(idx)
        vals = self.session_actions.values()
        self.done_count = sum(1 for v in vals if v == "save")
        self.skipped_count = sum(1 for v in vals if v == "skip")
        # recomputed (not incremented) so correcting an auto-accept via '◀ 이전'
        # moves it out of the auto tally into the save tally
        self.auto_accepted = sum(1 for v in vals if v == "auto")

    def _navigate_after(self, idx0, revisit):
        """After a save/skip: a revisit hops back to the live frontier; a normal
        forward decision advances the frontier to the next undecided image."""
        if revisit:
            self.idx = self.frontier
        else:
            self.idx = idx0 + 1
            self._advance_to_undecided()
            self.frontier = self.idx
        self._prepared = None

    def _go_back(self) -> dict:
        """Jump to the nearest image decided this session that sits before the
        current one, leaving the frontier untouched."""
        prev = max((i for i in self.session_decided if i < self.idx), default=None)
        if prev is None:
            return {"ok": False, "error": "이전 항목이 없습니다"}
        self.idx = prev
        self._prepared = None
        return {"ok": True, "action": "back", "index": prev}

    def _write_crop_and_log(self, prep, full, action_kind, allow_overwrite,
                            meta=None) -> dict:
        """Warp → save (beside/force-aware) → store dataset → log. Does NOT touch
        counters or the queue position — callers own navigation. Shared by the
        interactive save path and the ``--only-flagged`` auto-accept. ``meta`` carries
        the review-signal context (verdict/presented/auto/revised)."""
        src = prep["path"]
        dst = (src.parent / f"{src.stem}.jpg" if self.beside
               else self.outdir / f"{src.stem}_framefit.{self.out_format}")
        if dst.exists() and not allow_overwrite:
            return {"ok": False, "error": f"{dst.name} 이미 존재 — --force 필요"}
        r = process_manual(prep["bgr"], full, refine=False)
        dst.parent.mkdir(parents=True, exist_ok=True)
        io.save_bgr(r.image, dst, quality=self.quality)

        orig_rel, crop_rel = feedback.store_dataset(prep["sha1"], prep["bgr"], r.image)
        rec = feedback.build_record(
            source_path=prep["path"], source_sha1=prep["sha1"],
            image_width=prep["width"], image_height=prep["height"],
            backend=prep["backend"], auto_low_confidence=prep["low_confidence"],
            auto_aspect_score=prep["aspect_score"],
            auto_detect_score=prep.get("detect_score"), auto_quad=prep["auto_full"],
            display_scale=prep["scale"],
            final_quad=[[float(x), float(y)] for x, y in full],
            action=action_kind, output_path=str(dst),
            output_aspect_ratio=r.aspect_ratio, output_aspect_score=r.aspect_score,
            dataset_original=orig_rel, dataset_crop=crop_rel,
            **(meta or {}))
        feedback.append_record(rec)
        self.decided.add(prep["sha1"])
        return {"ok": True, "action": action_kind, "output": str(dst),
                "aspect_score": round(float(r.aspect_score), 3)}


class _Handler(BaseHTTPRequestHandler):
    state: ReviewState = None       # set on the server instance's handler class
    server_version = "framefit-review"

    def log_message(self, *a):      # quiet
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/state":
            body = json.dumps(self.state.state_json(), ensure_ascii=False)
            self._send(200, body.encode("utf-8"))
        elif path == "/quit":
            self._send(200, b'{"ok":true}')
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/decide":
            self._send(404, b'{"error":"not found"}')
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, b'{"error":"bad json"}')
            return
        result = self.state.decide(data.get("action", "save"),
                                   data.get("corners"))
        self._send(200, json.dumps(result, ensure_ascii=False).encode("utf-8"))


def run_review(files, outdir, backend="auto", display_max=1400,
               review_threshold=0.90, out_format="jpg", quality=95,
               open_browser=True, skip_decided=True, beside=False, force=False,
               only_flagged=False) -> int:
    """Boot the review server, open the browser, block until the queue is done
    or the user quits. Returns the number of images cropped. ``skip_decided``
    auto-skips images already decided in the review log (resume-safety); pass
    False to re-present a previously-decided image for correction."""
    if not files:
        print("framefit review: no input images")
        return 0
    state = ReviewState(files, outdir, backend=backend, display_max=display_max,
                        review_threshold=review_threshold,
                        out_format=out_format, quality=quality,
                        skip_decided=skip_decided,
                        beside=beside, force=force, only_flagged=only_flagged)

    handler = type("_H", (_Handler,), {"state": state})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    host, port = httpd.server_address
    url = f"http://127.0.0.1:{port}/"
    print(f"framefit review: {state.total} image(s) — "
          f"{state._pre_decided} already decided (skipped)")
    print(f"  open: {url}")
    print(f"  dataset: {feedback.review_root()}")
    print("  (close the browser tab and press Ctrl-C, or click 종료, when done)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    print(f"\nframefit review: {state.done_count} cropped, "
          f"{state.skipped_count} skipped.")
    return state.done_count


# --------------------------------------------------------------------------- #
# The review page (self-contained: draggable handles, confidence badge, buttons)
# --------------------------------------------------------------------------- #
_PAGE = r"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>framefit review</title>
<style>
 :root{color-scheme:dark}
 html,body{height:100%}
 body{font-family:-apple-system,sans-serif;background:#111;color:#eee;margin:0;
   padding:8px;box-sizing:border-box;display:flex;flex-direction:column;height:100vh}
 header{flex:0 0 auto;display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
 h1{font-size:14px;margin:0;font-weight:600}
 .badge{padding:3px 10px;border-radius:12px;font-size:13px;font-weight:600}
 .good{background:#0a5;color:#fff}.suspect{background:#b70;color:#fff}
 .fail{background:#b33;color:#fff}
 .meta{color:#9ab;font-size:12px}
 #stagewrap{flex:1 1 auto;display:flex;align-items:center;justify-content:center;
   min-height:0;min-width:0;overflow:hidden}
 #stage{position:relative;touch-action:none;user-select:none}
 #img{display:block;width:100%;height:100%;border:1px solid #444;box-sizing:border-box}
 svg{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;overflow:visible}
 .handle{position:absolute;width:24px;height:24px;margin:-12px 0 0 -12px;border-radius:50%;
   background:rgba(0,255,150,.25);border:2px solid #0f8;cursor:grab;
   display:flex;align-items:center;justify-content:center;font-size:11px;
   font-weight:700;color:#0f8;pointer-events:auto}
 .handle:active{cursor:grabbing}
 #bar{flex:0 0 auto;margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
 button{background:#2a2a2a;color:#eee;border:1px solid #555;border-radius:6px;
   padding:8px 16px;cursor:pointer;font-size:14px}
 button.primary{background:#0a5;border-color:#0a5;font-weight:700}
 button:disabled{opacity:.4;cursor:not-allowed}
 #hint{color:#9cf;font-size:12px}#done{font-size:16px;padding:30px}
</style>
<header>
 <h1 id="name">…</h1>
 <span id="badge" class="badge">…</span>
 <span class="meta" id="meta"></span>
 <span class="meta" id="prog"></span>
</header>
<div id="stagewrap"><div id="stage"><img id="img"><svg id="svg" preserveAspectRatio="none"></svg></div></div>
<div id="done" style="display:none"></div>
<div id="bar">
 <button id="back">◀ 이전</button>
 <button id="save" class="primary">저장 (OK)</button>
 <button id="reset">원위치</button>
 <button id="skip">건너뛰기</button>
 <button id="quit">종료</button>
 <span id="hint"></span>
</div>
<script>
const wrap=document.getElementById('stagewrap'),stage=document.getElementById('stage'),
 img=document.getElementById('img'),svg=document.getElementById('svg');
let pts=[],inv=1,dispW=0,dispH=0,auto=null,dragging=-1,fit=1;

function computeFit(){
 if(!dispW||!dispH){fit=1;return;}
 const f=Math.min(wrap.clientWidth/dispW, wrap.clientHeight/dispH);
 fit=(isFinite(f)&&f>0)?f:1;
}
async function load(){
 const s=await (await fetch('/state')).json();
 if(s.done){finish(s);return;}
 // returning from the done screen (via 이전): un-hide the editor
 wrap.style.display='';document.getElementById('bar').style.display='';
 document.querySelector('header').style.display='';
 document.getElementById('done').style.display='none';
 document.getElementById('name').textContent=s.name;
 const b=document.getElementById('badge');
 b.textContent=s.verdict.label; b.className='badge '+s.verdict.level;
 document.getElementById('meta').textContent=
   `백엔드 ${s.backend} · 종횡비점수 ${s.aspect_score}`+
   (s.low_confidence?' · 저신뢰':'');
 document.getElementById('prog').textContent=
   `[${s.index+1}/${s.total}] 완료 대기 ${s.total-s.index}`;
 dispW=s.disp_w;dispH=s.disp_h;
 img.src='data:image/jpeg;base64,'+s.image_b64;
 svg.setAttribute('viewBox','0 0 '+dispW+' '+dispH);
 auto=s.auto_disp;
 // a revisited image reloads the corners last confirmed; otherwise start from auto
 const init = s.prev_disp || auto;
 pts = init ? init.map(p=>[p[0],p[1]]) : [];
 document.getElementById('back').disabled = !s.can_back;
 document.getElementById('hint').textContent =
   (s.revisit ? '↩ 이전에 확정한 항목 — 저장했던 모서리를 불러왔습니다. 고친 뒤 저장하면 원래 위치로 돌아갑니다. ' : '')
   + (auto
      ? '핸들을 드래그해 모서리를 맞추세요. 맞으면 저장(OK).'
      : '자동검출 실패 — 이미지 위 네 모서리를 순서대로(좌상→우상→우하→좌하) 클릭.');
 computeFit();render();
}
function finish(s){
 wrap.style.display='none';
 document.getElementById('bar').style.display='none';
 document.querySelector('header').style.display='none';
 const d=document.getElementById('done');d.style.display='block';
 d.innerHTML=`✅ 완료 — 크롭 ${s.decided}장, 건너뜀 ${s.skipped}장`+
   (s.auto_accepted?`, 자동수락 ${s.auto_accepted}장`:'')+
   (s.pre_decided?`, 이전 결정 ${s.pre_decided}장 건너뜀`:'')+
   `<br><br>`+
   (s.can_back?`<button id="backdone">◀ 이전 항목으로 돌아가 수정</button><br><br>`:'')+
   `이 탭을 닫으셔도 됩니다.`;
 if(s.can_back){document.getElementById('backdone').onclick=()=>post({action:'back'});}
}
function render(){
 stage.style.width=(dispW*fit)+'px';stage.style.height=(dispH*fit)+'px';
 [...svg.querySelectorAll('*')].forEach(e=>e.remove());
 [...stage.querySelectorAll('.handle')].forEach(e=>e.remove());
 if(pts.length>=2){
  const poly=document.createElementNS('http://www.w3.org/2000/svg','polygon');
  poly.setAttribute('points',pts.map(p=>p.join(',')).join(' '));
  poly.setAttribute('fill','rgba(0,255,150,.10)');
  poly.setAttribute('stroke','#0f8');poly.setAttribute('stroke-width','2');
  poly.setAttribute('vector-effect','non-scaling-stroke');
  svg.appendChild(poly);
 }
 pts.forEach((p,i)=>{
  const h=document.createElement('div');h.className='handle';h.textContent=i+1;
  h.style.left=(p[0]*fit)+'px';h.style.top=(p[1]*fit)+'px';
  h.addEventListener('pointerdown',ev=>{ev.preventDefault();dragging=i;
    h.setPointerCapture(ev.pointerId);});
  stage.appendChild(h);
 });
 document.getElementById('save').disabled = pts.length!==4;
}
function xy(ev){const r=stage.getBoundingClientRect();
 return [Math.max(0,Math.min(dispW,(ev.clientX-r.left)/fit)),
         Math.max(0,Math.min(dispH,(ev.clientY-r.top)/fit))];}
stage.addEventListener('pointermove',ev=>{
 if(dragging<0)return;pts[dragging]=xy(ev);render();});
window.addEventListener('pointerup',()=>{dragging=-1;});
window.addEventListener('resize',()=>{if(dispW){computeFit();render();}});
img.addEventListener('click',ev=>{
 if(auto||pts.length>=4)return;pts.push(xy(ev));render();});
document.getElementById('reset').onclick=()=>{
 pts=auto?auto.map(p=>[p[0],p[1]]):[];render();};
document.getElementById('save').onclick=async()=>{
 if(pts.length!==4)return;
 await post({action:'save',corners:pts});};
document.getElementById('skip').onclick=async()=>{await post({action:'skip'});};
document.getElementById('back').onclick=async()=>{await post({action:'back'});};
document.getElementById('quit').onclick=async()=>{
 await fetch('/quit');document.getElementById('done').style.display='block';
 document.getElementById('done').textContent='종료했습니다. 탭을 닫으세요.';
 document.getElementById('stage').style.display='none';
 document.getElementById('bar').style.display='none';};
async function post(body){
 const bar=document.getElementById('bar');bar.style.pointerEvents='none';
 try{
   const res=await (await fetch('/decide',{method:'POST',
     headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
   if(res.ok===false){document.getElementById('hint').textContent='⚠ '+res.error;return;}
   await load();
 }finally{bar.style.pointerEvents='';}
}
load();
</script>
</html>"""
