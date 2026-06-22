def test_celltypist_fallback_list():
    from scomnom.io_utils import get_available_celltypist_models
    models = get_available_celltypist_models()
    assert isinstance(models, list)
    assert all(isinstance(m, dict) for m in models)
    assert all("name" in m for m in models)
