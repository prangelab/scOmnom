def test_celltypist_fallback_list():
    from scomnom.io_utils import get_available_celltypist_models
    models = get_available_celltypist_models(timeout=1)
    assert isinstance(models, list)
    assert all(m.endswith(".pkl") for m in models)
