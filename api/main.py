import pathlib

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes.submissions import router as submissions_router
from api.routes.tasks import router as tasks_router

app = FastAPI(
    title="ForgeEval",
    version="0.1.0",
    description="ML engineering benchmark and adversarial evaluation platform",
)

app.include_router(tasks_router)
app.include_router(submissions_router)

# Mount dashboard static assets
dashboard_dir = pathlib.Path(__file__).resolve().parents[1] / "dashboard"
if dashboard_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "ForgeEval", "version": "0.1.0", "dashboard": "/dashboard"}
