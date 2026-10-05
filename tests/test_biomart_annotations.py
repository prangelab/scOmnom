import gzip
import hashlib
import json
from http.client import IncompleteRead
from urllib.parse import parse_qs, urlparse
from xml.etree import ElementTree

import anndata as ad
import pandas as pd
import pytest

from scomnom import io_utils


VALID = (
    "Gene name\tGene type\tChromosome/scaffold name\tGene stable ID\n"
    "HOTAIRM1\tlncRNA\t7\tENSG00000233429\n"
    "STAT2\tprotein_coding\t12\tENSG00000170581\n"
    "\tlncRNA\tX\tENSG00000999999\n"
).encode()
ARCHIVE = "https://jun2026.archive.ensembl.org/biomart/martservice"
TRANSPORT = io_utils._biomart_response


@pytest.fixture(autouse=True)
def isolated_biomart(monkeypatch, tmp_path):
    monkeypatch.setattr(io_utils, "GENE_ANNOTATION_CACHE", tmp_path)
    monkeypatch.setattr(io_utils, "_GENE_TYPE_MAP_CACHE", {})
    monkeypatch.setattr(io_utils, "_GENE_TYPE_SOURCE_CACHE", {})
    monkeypatch.setattr(io_utils.time, "sleep", lambda _: None)
    # Unit tests must never contact Ensembl.
    monkeypatch.setattr(io_utils, "_biomart_response", lambda _: (VALID, ARCHIVE))


def test_valid_response_cache_and_provenance(monkeypatch, tmp_path):
    df = io_utils.fetch_ensembl_gene_annotations()
    assert list(df.gene) == ["HOTAIRM1", "STAT2"]
    assert list(df.gene_type) == ["lncRNA", "protein_coding"]
    provenance = df.attrs["annotation_provenance"]
    assert provenance["resolved_url"] == ARCHIVE
    assert provenance["response_sha256"] == hashlib.sha256(VALID).hexdigest()
    assert provenance["dataset"] == "hsapiens_gene_ensembl"
    assert provenance["retrieved_at_utc"]
    monkeypatch.setattr(io_utils, "_biomart_response", lambda _: pytest.fail("cache miss"))
    pd.testing.assert_frame_equal(df, io_utils.fetch_ensembl_gene_annotations())
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("body", [
    b"<html>Service unavailable</html>", b"Query ERROR: unavailable",
    b"", b"gene\tgene_type\nABC\tprotein_coding\n",
    VALID + b"truncated\trow\n",
    VALID.replace(b"ENSG00000170581", b""),
    VALID.replace(b"protein_coding", b""),
])
def test_invalid_response_never_cached(monkeypatch, tmp_path, body):
    calls = []

    def response(query):
        calls.append(query)
        return body, ARCHIVE

    monkeypatch.setattr(io_utils, "_biomart_response", response)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        io_utils.fetch_ensembl_gene_annotations()
    assert len(calls) == 3
    assert not list(tmp_path.iterdir())


def test_retry_recovers_from_html_and_timeout(monkeypatch):
    responses = iter([b"<html>unavailable</html>", TimeoutError("timed out"), VALID])

    def response(_):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value, ARCHIVE

    monkeypatch.setattr(io_utils, "_biomart_response", response)
    assert len(io_utils.fetch_ensembl_gene_annotations()) == 2


def test_truncated_http_body_is_retried(monkeypatch):
    calls = []

    def response(query):
        calls.append(query)
        if len(calls) == 1:
            raise IncompleteRead(b"partial", 100)
        return VALID, ARCHIVE

    monkeypatch.setattr(io_utils, "_biomart_response", response)
    assert len(io_utils.fetch_ensembl_gene_annotations()) == 2
    assert len(calls) == 2


@pytest.mark.parametrize("damage", ["html", "checksum", "schema", "dataset", "query"])
def test_invalid_cache_is_not_trusted(monkeypatch, tmp_path, damage):
    io_utils.fetch_ensembl_gene_annotations()
    cache = next(tmp_path.glob("*.json.gz"))
    with gzip.open(cache, "rt") as handle:
        payload = json.load(handle)
    if damage == "html":
        payload["table"] = "<html>unavailable</html>"
        payload["provenance"]["table_sha256"] = hashlib.sha256(payload["table"].encode()).hexdigest()
    elif damage == "checksum":
        payload["table"] += "tampered"
    elif damage == "schema":
        payload["schema_version"] = 999
    elif damage == "dataset":
        payload["provenance"]["dataset"] = "mmusculus_gene_ensembl"
    else:
        payload["provenance"]["query_sha256"] = "wrong"
    with gzip.open(cache, "wt") as handle:
        json.dump(payload, handle)
    calls = []
    monkeypatch.setattr(io_utils, "_biomart_response", lambda q: (calls.append(q) or VALID, ARCHIVE))
    assert len(io_utils.fetch_ensembl_gene_annotations()) == 2
    assert len(calls) == 1


def test_failed_force_refresh_preserves_valid_cache(monkeypatch, tmp_path):
    expected = io_utils.fetch_ensembl_gene_annotations()
    cache = next(tmp_path.glob("*.json.gz"))
    original = cache.read_bytes()
    monkeypatch.setattr(io_utils, "_biomart_response", lambda _: (b"<html>down</html>", ARCHIVE))
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        io_utils.fetch_ensembl_gene_annotations(force=True)
    assert cache.read_bytes() == original
    pd.testing.assert_frame_equal(expected, io_utils.fetch_ensembl_gene_annotations())


def test_unvalidated_legacy_cache_ignored(tmp_path):
    legacy = tmp_path / "hsapiens_gene_ensembl__gene_annotations.tsv.gz"
    with gzip.open(legacy, "wt") as handle:
        handle.write("<html>unavailable</html>")
    assert len(io_utils.fetch_ensembl_gene_annotations()) == 2
    assert legacy.exists()


def test_duplicate_symbols_prefer_standard_chromosome():
    body = VALID + b"STAT2\tprotein_coding\tpatch\tENSG00000123456\n"
    df = io_utils._parse_biomart_annotations(body)
    assert df.set_index("gene").loc["STAT2", "gene_id"] == "ENSG00000170581"


def test_marker_annotation_fails_closed(monkeypatch):
    monkeypatch.setattr(io_utils, "_biomart_response", lambda _: (b"<html>down</html>", ARCHIVE))
    adata = ad.AnnData(var=pd.DataFrame(index=["HOTAIRM1"]))
    with pytest.raises(RuntimeError, match="No inferred protein-coding"):
        io_utils._add_gene_type_column(adata, pd.DataFrame({"gene": ["HOTAIRM1"]}))


def test_explicit_fallback_is_unknown_preserves_ids_and_is_object_specific(monkeypatch):
    monkeypatch.setattr(io_utils, "_biomart_response", lambda _: (b"<html>down</html>", ARCHIVE))
    for gene in ("HOTAIRM1", "STAT2"):
        adata = ad.AnnData(var=pd.DataFrame({"gene_id": ["input_id"]}, index=[gene]))
        types, source, _, _, ids = io_utils.get_gene_type_map(adata, allow_fallback=True)
        assert types == {gene: "unknown"}
        assert source == "fallback"
        assert ids == {gene: "input_id"}


def test_existing_biotypes_work_offline(monkeypatch):
    monkeypatch.setattr(io_utils, "_biomart_response", lambda _: pytest.fail("network"))
    adata = ad.AnnData(var=pd.DataFrame({
        "gene_type": ["lncRNA"], "gene_id": ["input_id"],
    }, index=["HOTAIRM1"]))
    df = io_utils._add_gene_type_column(adata, pd.DataFrame({"gene": ["HOTAIRM1"]}))
    assert df.loc[0, "gene_type"] == "lncRNA"
    assert df.loc[0, "gene_id"] == "input_id"


def test_reference_lookup_preserves_input_ids_without_polluting_memory_cache():
    adata = ad.AnnData(var=pd.DataFrame({"gene_id": ["input_id"]}, index=["STAT2"]))
    assert io_utils.get_gene_type_map(adata)[4]["STAT2"] == "input_id"
    other = ad.AnnData(var=pd.DataFrame(index=["STAT2"]))
    assert io_utils.get_gene_type_map(other)[4]["STAT2"] == "ENSG00000170581"


@pytest.mark.parametrize("url,content_type,error", [
    (ARCHIVE, "text/plain", None),
    (ARCHIVE, "text/html", "returned HTML"),
    ("https://untrusted.example/biomart", "text/plain", "Unexpected BioMart"),
    ("http://www.ensembl.org/biomart", "text/plain", "Unexpected BioMart"),
])
def test_https_transport_timeout_and_archive_redirect(monkeypatch, url, content_type, error):
    class Response:
        headers = {"Content-Type": content_type}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def geturl(self):
            return url

        def read(self):
            return VALID

    def urlopen(request, timeout):
        assert timeout == 30
        assert request.full_url.startswith(io_utils._BIOMART_URL)
        query = parse_qs(urlparse(request.full_url).query)["query"][0]
        assert ElementTree.fromstring(query).find("Dataset").attrib["name"] == "hsapiens_gene_ensembl"
        return Response()

    monkeypatch.setattr(io_utils.urllib.request, "urlopen", urlopen)
    query = b'<Query><Dataset name="hsapiens_gene_ensembl" /></Query>'
    if error:
        with pytest.raises(ValueError, match=error):
            TRANSPORT(query)
    else:
        body, resolved = TRANSPORT(query)
        assert body == VALID and resolved == ARCHIVE


@pytest.mark.parametrize("species", ["../human", "", '<Dataset name="x">'])
def test_species_identifier_validation(species):
    with pytest.raises(ValueError, match="species identifier"):
        io_utils.fetch_ensembl_gene_annotations(species=species)
