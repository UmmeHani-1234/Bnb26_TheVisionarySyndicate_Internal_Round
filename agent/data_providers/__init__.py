"""Product Data Providers package for Black Box Electronics Consultant."""

from .base import ProductDataProvider
from .cache import ProductDataCache
from .composite_provider import CompositeProductProvider, get_product_provider, reset_product_provider
from .external_provider import ExternalProductProvider
from .local_provider import LocalCatalogueProvider
from .schema import NormalizedProduct, SearchFilter

__all__ = [
    "ProductDataProvider",
    "NormalizedProduct",
    "SearchFilter",
    "ProductDataCache",
    "LocalCatalogueProvider",
    "ExternalProductProvider",
    "CompositeProductProvider",
    "get_product_provider",
    "reset_product_provider",
]
