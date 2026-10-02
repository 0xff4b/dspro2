"""Tests for the tenant and landlord pages with stub services; skipped without the app extra."""

import pytest

pytest.importorskip("dash")

from rentml.estimate import RentEstimator  # noqa: E402 -- after the optional-dependency skip
from rentml.geoadmin import Address, LocationData  # noqa: E402
from rentml.webapp import landlord, tenant  # noqa: E402
from rentml.webapp.app import create_app  # noqa: E402
from rentml.webapp.components import attribute_labels, encode_address  # noqa: E402
from rentml.webapp.services import Services  # noqa: E402

ADDRESS = Address("Teststrasse 1 8000 Aach", 2_600_500.0, 1_200_500.0, 1)


@pytest.fixture
def services(rent_model, map_bundle) -> Services:
    location = LocationData(
        year_built=1990.0,
        apartments=6.0,
        oev=30000.0,
        solar=2.0,
        population=90.0,
        elevation=420.0,
        missing=["land_area"],
    )
    return Services(
        estimator=RentEstimator(rent_model, map_bundle),
        search=lambda q: [ADDRESS] if len(q) >= 3 else [],
        lookup=lambda a: location,
    )


def test_tenant_check_renders_verdict(services) -> None:
    html_text = str(
        tenant.run_check(services, encode_address(ADDRESS), 80, 3.5, 9000, "Wohnung mit Lift")
    )
    assert "Über der erwarteten Spanne" in html_text
    assert "Lift" in html_text and "Gebäudegrundfläche" in html_text  # missing input shown
    assert "Art. 270 OR" in html_text and "Kanton ZH" in html_text


def test_tenant_check_validates_input(services) -> None:
    assert "Bitte Adresse" in str(tenant.run_check(services, None, 80, 3.5, 1500, None))
    too_small = str(tenant.run_check(services, encode_address(ADDRESS), 3, 1, 1500, None))
    assert "Keine Schätzung möglich" in too_small and "Wohnfläche" in too_small


def test_landlord_band_and_what_if(services) -> None:
    html_text = str(landlord.run_band(services, encode_address(ADDRESS), 80, 3.5, None))
    assert "Empfohlenes Angebotsband" in html_text and "What-if" in html_text
    table = services.estimator.what_if(
        landlord.Apartment(ADDRESS, 80.0, 3.5), services.lookup(ADDRESS)
    )
    labels = landlord.what_if_labels(table)
    assert labels[0] == "Ihre Angaben" and "Wohnfläche 90 m² (+10)" in labels
    assert len(landlord.what_if_figure(table).data[0].x) == len(table) - 1


def test_attribute_labels_include_evidence(services) -> None:
    attributes = services.estimator.estimate(
        landlord.Apartment(ADDRESS, 80.0, 3.5, "Balkon mit Seesicht"), services.lookup(ADDRESS)
    ).attributes
    labels = dict(attribute_labels(attributes))
    assert labels["Balkon/Terrasse"] == "Balkon" and "Seesicht" in labels


def test_app_without_model_still_serves_all_pages(map_bundle) -> None:
    stub = Services(estimator=None, search=lambda q: [], lookup=lambda a: LocationData())
    client = create_app(map_bundle, services=stub).server.test_client()
    for path in ("/", "/mietcheck", "/vermieter"):
        assert client.get(path).status_code == 200


def test_app_registers_tool_callbacks(map_bundle, services) -> None:
    app = create_app(map_bundle, services=services)
    outputs = " ".join(app.callback_map)
    assert "tenant-result.children" in outputs and "landlord-result.children" in outputs
    assert "tenant-address.options" in outputs
