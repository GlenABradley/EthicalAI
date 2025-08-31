def test_app_importable():
    from coherence.api.main import app
    assert getattr(app, "title", "").lower().find("coherence") != -1
