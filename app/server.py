import os
import sys
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool


# Project root and sys.path setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from vgqa.inference.grounding import predict as stvg_predict
except Exception as e:
    raise RuntimeError("Failed to import vgqa.inference.grounding. Check PYTHONPATH and dependencies.") from e

try:
    from vgqa.inference.qa import predict as qa_predict
except Exception as e:
    print(f"Warning: QA module import failed: {e}")
    qa_predict = None

try:
    from decord import VideoReader, cpu
except Exception as e:
    raise RuntimeError("decord not installed: pip install decord") from e


# Video root path (can be overridden with environment variable)
DEFAULT_VIDEOS = PROJECT_ROOT / "videos"
VIDEOS_ROOT = Path(os.getenv("VGQA_VIDEOS_DIR", str(DEFAULT_VIDEOS))).resolve()

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="VGQA Web Interface")

# Serve static frontend and video files
app.mount("/app", StaticFiles(directory=str(STATIC_DIR), html=True), name="app")
if VIDEOS_ROOT.exists():
    app.mount("/videos", StaticFiles(directory=str(VIDEOS_ROOT), html=False), name="videos")

# Single concurrent inference lock (GPU protection)
_infer_lock = threading.Lock()


def _safe_join_video(name: str) -> Path:
    p = (VIDEOS_ROOT / name).resolve()
    if not str(p).startswith(str(VIDEOS_ROOT)):
        raise HTTPException(400, "Invalid path")
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "Video not found")
    return p


def _list_videos_in(dir_path: Optional[str]) -> List[str]:
    base = VIDEOS_ROOT if not dir_path else (VIDEOS_ROOT / dir_path)
    base = base.resolve()
    if not str(base).startswith(str(VIDEOS_ROOT)):
        raise HTTPException(400, "Invalid directory")
    if not base.exists():
        return []
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return sorted([f.name for f in base.iterdir() if f.is_file() and f.suffix.lower() in exts])


def _video_meta(path: Path):
    vr = VideoReader(str(path), ctx=cpu(0))
    fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
    total_frames = int(len(vr))
    frame0 = vr[0].asnumpy()
    h, w = frame0.shape[0], frame0.shape[1]
    return {"fps": fps, "total_frames": total_frames, "width": w, "height": h}


@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return {"message": "Static UI not found. Visit /app if configured."}
    return FileResponse(str(index))


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.get("/api/videos")
async def list_videos(dir: Optional[str] = Query(default=None, description="상대 경로 (VIDEOS_ROOT 기준)")):
    files = _list_videos_in(dir)
    return {"directory": str(VIDEOS_ROOT), "files": files}


@app.get("/api/meta")
async def video_meta(video: str = Query(..., description="파일명(또는 VIDEOS_ROOT 기준 상대경로)")):
    path = _safe_join_video(video)
    return _video_meta(path)


class PredictBody(BaseModel):
    video: str
    query: str
    device: Optional[str] = "auto"  # "auto" | "cpu" | "cuda"


class GenerateQueriesBody(BaseModel):
    video: str
    num_queries: Optional[int] = 10
    num_frames: Optional[int] = 64
    max_tokens: Optional[int] = 300


class QABody(BaseModel):
    video: str
    question: str
    num_frames: Optional[int] = 32
    max_tokens: Optional[int] = 256
    bound_start: Optional[float] = None
    bound_end: Optional[float] = None


@app.post("/api/predict")
async def predict(body: PredictBody):
    path = _safe_join_video(body.video)
    meta = _video_meta(path)

    # 단일 동시 추론 강제
    if not _infer_lock.acquire(blocking=False):
        raise HTTPException(409, "Another inference is in progress. Please wait.")

    try:
        device_arg = None if (body.device or "auto").lower() == "auto" else body.device
        # run_stvg.predict 는 블로킹 → 스레드풀에서 실행
        res = await run_in_threadpool(
            stvg_predict,
            str(path),
            body.query,
            # cfg_path와 ckpt_path는 run_stvg.py 기본값 사용
            device_str=device_arg,
        )
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {type(e).__name__}: {e}") from e
    finally:
        _infer_lock.release()

    # 메타정보 포함해 반환 (프론트가 즉시 사용)
    return {
        "video": {"name": path.name, "url": f"/videos/{path.name}"},
        "meta": meta,
        "result": res,
    }


@app.post("/api/generate-queries")
async def generate_queries(body: GenerateQueriesBody):
    """Generate grounding queries from video using QA model."""
    if qa_predict is None:
        raise HTTPException(503, "QA module not available")

    path = _safe_join_video(body.video)

    # 단일 동시 추론 강제
    if not _infer_lock.acquire(blocking=False):
        raise HTTPException(409, "Another inference is in progress. Please wait.")

    try:
        question = (
            f"Generate {body.num_queries} text queries for video grounding. "
            "Each query should be a short phrase describing a visible action "
            "(e.g., 'a person walking to the left', 'a red ball rolling', 'someone waving hand'). "
            "List them numbered."
        )

        # QA 추론 실행
        res = await run_in_threadpool(
            qa_predict,
            str(path),
            question,
            bound=None,
            num_frames=body.num_frames,
            max_new_tokens=body.max_tokens,
        )

        # 응답에서 쿼리 추출
        answer = res.get("answer", "")
        queries = _parse_queries_from_answer(answer)

        return {
            "queries": queries,
            "raw_answer": answer,
        }
    except Exception as e:
        raise HTTPException(500, f"Query generation failed: {type(e).__name__}: {e}") from e
    finally:
        _infer_lock.release()


@app.post("/api/qa")
async def qa(body: QABody):
    """Answer questions about the video using QA model."""
    if qa_predict is None:
        raise HTTPException(503, "QA module not available")

    path = _safe_join_video(body.video)

    # 단일 동시 추론 강제
    if not _infer_lock.acquire(blocking=False):
        raise HTTPException(409, "Another inference is in progress. Please wait.")

    try:
        bound = None
        if body.bound_start is not None and body.bound_end is not None:
            bound = (body.bound_start, body.bound_end)

        # QA 추론 실행
        res = await run_in_threadpool(
            qa_predict,
            str(path),
            body.question,
            bound=bound,
            num_frames=body.num_frames,
            max_new_tokens=body.max_tokens,
        )

        return res
    except Exception as e:
        raise HTTPException(500, f"QA failed: {type(e).__name__}: {e}") from e
    finally:
        _infer_lock.release()


def _parse_queries_from_answer(answer: str) -> List[str]:
    """Parse numbered queries from QA model answer."""
    import re

    queries = []

    # Try different patterns
    patterns = [
        r'^\d+[\.)]\s*(.+)$',  # "1. query" or "1) query"
        r'^[-•]\s*(.+)$',       # "- query" or "• query"
    ]

    for line in answer.split('\n'):
        line = line.strip()
        if not line:
            continue

        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                query = match.group(1).strip()
                # Clean up
                query = query.strip('"\'.,:')
                if query and len(query) > 5:  # Minimum length
                    queries.append(query)
                break

    # If no structured format found, try to split by common delimiters
    if not queries and answer:
        # Try splitting by sentences
        for sentence in re.split(r'[.!?]\s+', answer):
            sentence = sentence.strip()
            if len(sentence) > 10 and len(sentence) < 100:
                queries.append(sentence)

    return queries[:20]  # Max 20 queries


if __name__ == "__main__":
    # Local development server
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.server:app", host="0.0.0.0", port=port, reload=False)


