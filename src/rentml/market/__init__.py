"""RentLens marketplace: landlords publish listings, tenants apply, the models rate every listing.

The package holds the domain layer without Dash imports, so it is testable on its own:

* :mod:`rentml.market.models`: listings, applications, queries and their validation;
* :mod:`rentml.market.store`: persistence on SQLAlchemy Core (SQLite locally, PostgreSQL in
  production), secret links instead of accounts;
* :mod:`rentml.market.insight`: the fusion with the models (estimate, calibrated interval,
  verdict, text attributes, duplicate check, value score);
* :mod:`rentml.market.quality`: quality-of-life profile of the municipality (QoLI);
* :mod:`rentml.market.photos`: validation and re-encoding of uploaded photos;
* :mod:`rentml.market.seed`: synthetic demo listings.

The pages live in :mod:`rentml.webapp`. Unlike the Fair-Rent Check, the marketplace stores what
users explicitly publish or submit; applicants are never scored or ranked by a model.
"""
