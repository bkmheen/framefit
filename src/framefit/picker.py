"""Generate a self-contained HTML corner-picker for a single image.

For shots the detector can't crop (flagged low-confidence), open the generated
HTML in a browser, click the four slide corners (TL, TR, BR, BL), and copy the
ready-to-run ``framefit ... --corners`` command it prints. Works headless — it only
writes a file; the picking happens in the browser.
"""
from __future__ import annotations

import base64
from pathlib import Path

import cv2

from . import io


def write_picker_html(
    image_path: str | Path,
    out_html: str | Path,
    out_dir: str = "framefit_out",
    display_max: int = 1600,
) -> Path:
    image_path = Path(image_path).resolve()
    out_html = Path(out_html)
    bgr = io.load_bgr(image_path)
    fh, fw = bgr.shape[:2]
    scale = display_max / max(fh, fw) if max(fh, fw) > display_max else 1.0
    disp = cv2.resize(bgr, (int(fw * scale), int(fh * scale)),
                      interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
    ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode()
    inv = 1.0 / scale  # display px -> full-res px

    html = _TEMPLATE.replace("__B64__", b64) \
        .replace("__INV__", f"{inv:.6f}") \
        .replace("__IMG__", str(image_path).replace("\\", "\\\\")) \
        .replace("__OUT__", out_dir) \
        .replace("__NAME__", image_path.name)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    return out_html


_TEMPLATE = """<!doctype html><meta charset='utf-8'><title>framefit corner picker — __NAME__</title>
<style>
 body{font-family:sans-serif;background:#111;color:#eee;margin:16px}
 #wrap{position:relative;display:inline-block;cursor:crosshair}
 #img{display:block;max-width:100%;border:1px solid #444}
 .dot{position:absolute;width:16px;height:16px;margin:-8px 0 0 -8px;border-radius:50%;
      background:#0f8;border:2px solid #fff;color:#000;font-size:11px;font-weight:700;
      text-align:center;line-height:14px;pointer-events:none}
 #cmd{width:100%;height:70px;background:#000;color:#4f8;font-family:monospace;font-size:13px}
 button{background:#333;color:#eee;border:1px solid #555;padding:6px 12px;cursor:pointer}
 .hint{color:#9cf}
</style>
<h2>framefit corner picker — __NAME__</h2>
<p class='hint'>Click the four slide corners in order: <b>1 top-left → 2 top-right →
 3 bottom-right → 4 bottom-left</b>. Then copy the command below and run it.</p>
<div id='wrap'><img id='img' src='data:image/jpeg;base64,__B64__'></div>
<p><button onclick='reset()'>Reset</button></p>
<textarea id='cmd' readonly placeholder='Pick 4 corners…'></textarea>
<script>
const INV=__INV__, IMG="__IMG__", OUT="__OUT__";
const wrap=document.getElementById('wrap'), img=document.getElementById('img');
let pts=[];
img.addEventListener('click',e=>{
  if(pts.length>=4) return;
  const r=img.getBoundingClientRect();
  const dx=(e.clientX-r.left)*(img.naturalWidth/r.width);
  const dy=(e.clientY-r.top)*(img.naturalHeight/r.height);
  pts.push([dx,dy]);
  const d=document.createElement('div'); d.className='dot'; d.textContent=pts.length;
  d.style.left=(e.clientX-r.left)+'px'; d.style.top=(e.clientY-r.top)+'px';
  wrap.appendChild(d); render();
});
function render(){
  if(pts.length<4){document.getElementById('cmd').value='Picked '+pts.length+'/4 corners…';return;}
  const c=pts.map(p=>Math.round(p[0]*INV)+','+Math.round(p[1]*INV)).join(' ');
  document.getElementById('cmd').value='framefit "'+IMG+'" --corners "'+c+'" -o "'+OUT+'"';
}
function reset(){pts=[];document.querySelectorAll('.dot').forEach(d=>d.remove());render();}
render();
</script>
"""
