"""RentLens: Dash app factory, GeoJSON endpoint and command-line entry point.

One platform with the entry points of the proposal: the map (start page), the Fair-Rent Check
for tenants and the price band with what-if for landlords. The map bundle is read from
``$RENTML_MAP_BUNDLE`` or ``dspro2/data/app/map`` (``uv run python -m rentml.mapbundle``), the
model bundle from ``$RENTML_MODEL_BUNDLE`` or ``dspro2/models/rent_bundle_v1.joblib`` (written by
the notebook). Without a model the map still works and the tools show a notice.
"""

import argparse
import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import dash
from dash import Dash, dcc, html
from flask import Response, abort

from rentml.config import ProjectPaths
from rentml.estimate import RentEstimator
from rentml.mapbundle import MapBundle, default_bundle_dir
from rentml.mapdata import LEVELS
from rentml.webapp import home, landlord, tenant
from rentml.webapp.services import Services, live_services

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
BUNDLE_ENV = "RENTML_MAP_BUNDLE"
MODEL_ENV = "RENTML_MODEL_BUNDLE"
MODEL_FILE = "rent_bundle_v1.joblib"
APP_TITLE = "RentLens"
APP_TAGLINE = "Mietpreise der Schweiz, transparent erklärt"
GEOJSON_ROUTE = "/map-data/<level>.geojson"
CACHE_SECONDS = 86_400


def resolve_bundle_dir(bundle_dir: Path | None = None) -> Path:
    """Bundle directory: argument, then ``$RENTML_MAP_BUNDLE``, then the project default."""
    if bundle_dir is not None:
        return Path(bundle_dir)
    if env := os.environ.get(BUNDLE_ENV):
        return Path(env)
    return default_bundle_dir(ProjectPaths.discover())


def resolve_model_path(model_path: Path | None = None) -> Path:
    """Model bundle: argument, then ``$RENTML_MODEL_BUNDLE``, then ``dspro2/models``."""
    if model_path is not None:
        return Path(model_path)
    if env := os.environ.get(MODEL_ENV):
        return Path(env)
    return ProjectPaths.discover().models / MODEL_FILE


def load_estimator(model_path: Path, bundle: MapBundle) -> RentEstimator | None:
    """Load the model bundle; ``None`` (with a warning) if the file does not exist."""
    if not model_path.is_file():
        logger.warning("No model bundle at %s: tenant and landlord tools are disabled", model_path)
        return None
    return RentEstimator.load(model_path, bundle)


def create_app(
    bundle: MapBundle | None = None,
    *,
    bundle_dir: Path | None = None,
    services: Services | None = None,
    model_path: Path | None = None,
) -> Dash:
    """Create RentLens with the map as start page.

    Args:
        bundle: Map bundle; ``None`` = load it from ``bundle_dir``.
        bundle_dir: Directory of the bundle (see :func:`resolve_bundle_dir`).
        services: Estimator and geo.admin services; ``None`` = live services with the model
            from ``model_path`` (see :func:`resolve_model_path`).
        model_path: Model bundle file.

    Returns:
        The configured app (``app.server`` is the Flask server for WSGI hosting).

    Raises:
        FileNotFoundError: If no bundle is given and none exists on disk.
    """
    bundle = bundle or MapBundle.load(resolve_bundle_dir(bundle_dir))
    app = Dash(
        __name__,
        use_pages=True,
        pages_folder="",
        assets_folder=str(ASSETS_DIR),
        title=APP_TITLE,
        update_title=None,
        compress=True,
        suppress_callback_exceptions=True,
    )
    if services is None:
        services = live_services(load_estimator(resolve_model_path(model_path), bundle))
    home.register(app, bundle, _serve_geojson(app, bundle))
    tenant.register(app, services)
    landlord.register(app, services)
    app.layout = _shell()
    return app


def _serve_geojson(app: Dash, bundle: MapBundle) -> Callable[[str], str]:
    """Serve each level's GeoJSON once per browser (cached) and return the URL builder."""
    payloads = {
        level: json.dumps(bundle.geojson[level], separators=(",", ":")).encode() for level in LEVELS
    }
    version = str(bundle.meta.get("created_at", "0")).replace(":", "")

    def geojson(level: str) -> Response:
        if level not in payloads:
            abort(404)
        response = Response(payloads[level], mimetype="application/json")
        response.headers["Cache-Control"] = f"public, max-age={CACHE_SECONDS}"
        return response

    app.server.add_url_rule(GEOJSON_ROUTE, "map_geojson", geojson)

    def url(level: str) -> str:
        return app.get_relative_path(f"/map-data/{level}.geojson") + f"?v={version}"

    return url


def _shell() -> html.Div:
    nav = [
        dcc.Link(page["name"], href=page["relative_path"], className="nav-link")
        for page in sorted(dash.page_registry.values(), key=lambda p: p.get("order", 99))
    ]
    header = html.Header(
        [
            html.Div(
                [
                    html.Div(APP_TITLE, className="brand-title"),
                    html.Div(f"{APP_TAGLINE} · DSPRO2 HSLU", className="brand-sub"),
                ],
                className="brand",
            ),
            html.Nav(nav, className="nav"),
        ],
        className="app-header",
    )
    return html.Div([header, html.Main(dash.page_container, className="app-main")])


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: start the development server."""
    parser = argparse.ArgumentParser(description="RentLens (DSPRO2 Dash app)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--bundle", type=Path, default=None, help="map bundle directory")
    parser.add_argument("--model", type=Path, default=None, help="model bundle (.joblib)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = create_app(bundle_dir=args.bundle, model_path=args.model)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
