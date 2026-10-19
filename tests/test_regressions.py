"""Offline regression tests for installation, loading, mapping, and error handling."""

import json
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
import spacy
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from ojd_daps_skills import setup_spacy_extensions
from ojd_daps_skills.extract_skills import extract_skills_utils as extraction
from ojd_daps_skills.extract_skills.extract_skills import SkillsExtractor
from ojd_daps_skills.extract_skills.extract_skills_utils import (
    ExtractConfig,
    MultiSkillTransformer,
)
from ojd_daps_skills.map_skills import skill_mapper_utils as mapping
from ojd_daps_skills.map_skills import skill_mapper as mapper_module
from ojd_daps_skills.map_skills.skill_mapper import SkillsMapper
from ojd_daps_skills.map_skills.skill_mapper_utils import (
    MapConfig,
    get_most_common_code,
)
from ojd_daps_skills.utils.bert_vectorizer import BertVectorizer
from ojd_daps_skills.utils.data_getters import (
    save_json_dict,
    load_json_dict,
    load_s3_data,
)
from ojd_daps_skills.utils.text_cleaning import clean_text, short_hash


@pytest.fixture
def nlp():
    setup_spacy_extensions()
    return spacy.blank("en")


def fake_vectorizer():
    model = BertVectorizer()
    model.transform = Mock(
        side_effect=lambda texts: np.tile([1.0, 0.0], (len(texts), 1))
    )
    return model


@pytest.fixture
def config():
    return MapConfig(
        taxonomy_name="test",
        taxonomy_config={
            "skill_type_dict": {"skill_types": ["skill"], "hier_types": []},
            "skill_type_col": "type",
            "skill_name_col": "description",
            "skill_id_col": "id",
            "skill_hier_info_col": None,
            "num_hier_levels": 0,
            "match_thresholds_dict": {"skill_match_thresh": 0.7},
        },
        bert_model=fake_vectorizer(),
        taxonomy_data=pd.DataFrame(
            {"type": ["skill"], "description": ["Python"], "id": ["py"]}
        ),
        taxonomy_embeddings={0: np.array([1.0, 0.0])},
        hier_mapper={},
    )


@pytest.fixture
def extractor(nlp, config):
    nlp.add_pipe("entity_ruler").add_patterns(
        [
            {"label": "SKILL", "pattern": "Python"},
            {"label": "SKILL", "pattern": "skillsBe"},
        ]
    )
    classifier = DummyClassifier(strategy="constant", constant=0).fit([[1, 0, 0]], [0])
    extract = ExtractConfig(
        nlp=nlp,
        ms_model=Pipeline(
            [("features", MultiSkillTransformer()), ("classifier", classifier)]
        ),
    )
    return SkillsExtractor(extract_config=extract, map_config=config)


def test_injected_configs_are_used_without_download(extractor):
    docs = extractor("Python")
    assert docs[0]._.mapped_skills[0]["match_id"] == "py"
    assert [span.text for span in docs[0]._.skill_spans] == ["Python"]


@pytest.mark.parametrize("bad", [None, 42, ("Python",), ["Python", 1]])
def test_invalid_input_raises_exception_not_system_exit(extractor, bad):
    with pytest.raises(TypeError):
        extractor(bad)
    with pytest.raises(TypeError):
        extractor.map_skills(bad)


def test_empty_input_and_no_skills(extractor):
    assert extractor([]) == []
    docs = extractor(["", "Nothing here"])
    assert [doc._.skill_spans for doc in docs] == [[], []]
    assert [doc._.mapped_skills for doc in docs] == [[], []]


def test_mapping_without_hierarchy(extractor):
    assert extractor("Python")[0]._.mapped_skills[0]["match_type"] == "skill"


def test_cleaned_hash_is_used_for_mapping(extractor, nlp):
    doc = nlp.make_doc("skillsBe")
    doc._.skill_spans = [doc[0:1]]
    mapped = extractor.map_skills(doc)[0]._.mapped_skills[0]
    assert mapped is not None
    assert mapped["ojo_skill_id"] == short_hash(clean_text("skillsBe"))


def test_spans_deduplicate_by_text_and_keep_order(nlp, config):
    docs = [nlp("Python SQL"), nlp("Python")]
    docs[0]._.skill_spans = [docs[0][0:1], docs[0][1:2]]
    docs[1]._.skill_spans = [docs[1][0:1]]
    mapper = SkillsMapper(config=config)
    mapper.get_embeddings(docs)
    assert list(mapper.all_skills_unique_dict.values()) == ["Python", "SQL"]


def test_short_hierarchy_paths():
    assert get_most_common_code([["A"], ["A", "B"]], 1) == ("B", 0.5)
    assert get_most_common_code([], 0) == (None, None)


def test_embedding_cache_is_outside_package(nlp, config, tmp_path, monkeypatch):
    monkeypatch.setattr(mapper_module, "PUBLIC_MODEL_FOLDER_PATH", tmp_path)
    config.taxonomy_embeddings = None
    doc = nlp("Python")
    doc._.skill_spans = [doc[0:1]]
    SkillsMapper(config=config).get_embeddings([doc])
    assert json.loads((tmp_path / "taxonomies/test_embeddings.json").read_text()) == {
        "0": [1.0, 0.0]
    }


def test_atomic_json_accepts_paths_and_numpy(tmp_path):
    file = tmp_path / "nested/result.json"
    save_json_dict({"a": np.array([1, 2])}, file)
    assert load_json_dict(file) == {"a": [1, 2]}
    with pytest.raises(TypeError):
        save_json_dict({"bad": object()}, file)
    assert load_json_dict(file) == {"a": [1, 2]}
    assert list(file.parent.iterdir()) == [file]


def test_s3_pickle_is_read_as_bytes():
    import pickle

    s3 = Mock()
    s3.Object.return_value.get.return_value = {
        "Body": Mock(read=Mock(return_value=pickle.dumps({"value": 3})))
    }
    assert load_s3_data(s3, "bucket", "file.pkl") == {"value": 3}


def test_unknown_taxonomy_fails_before_model_download(monkeypatch):
    download = Mock(side_effect=AssertionError("should not download"))
    monkeypatch.setattr(mapping, "hf_hub_download", download)
    monkeypatch.setattr(BertVectorizer, "fit", download)
    with pytest.raises(ValueError, match="taxonomy_name"):
        MapConfig.create("unknown")
    download.assert_not_called()


@pytest.mark.parametrize("taxonomy", ["toy", "esco", "lightcast"])
def test_bundled_taxonomies_and_offline_embedding_fallback(
    taxonomy, tmp_path, monkeypatch
):
    monkeypatch.setattr(mapping, "PUBLIC_MODEL_FOLDER_PATH", tmp_path)
    monkeypatch.setattr(BertVectorizer, "fit", lambda self: self)
    monkeypatch.setattr(
        mapping, "hf_hub_download", Mock(side_effect=OSError("offline"))
    )
    with pytest.warns(RuntimeWarning, match="calculated locally"):
        result = MapConfig.create(taxonomy)
    assert not result.taxonomy_data.empty
    assert result.taxonomy_embeddings is None
    assert result.hier_mapper


def test_cached_taxonomy_embeddings_are_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(mapping, "PUBLIC_MODEL_FOLDER_PATH", tmp_path)
    monkeypatch.setattr(BertVectorizer, "fit", lambda self: self)
    download = Mock(side_effect=AssertionError("should not download"))
    monkeypatch.setattr(mapping, "hf_hub_download", download)
    save_json_dict({0: [1.0, 0.0]}, tmp_path / "taxonomies/toy_embeddings.json")
    result = MapConfig.create("toy")
    assert np.array_equal(result.taxonomy_embeddings[0], [1.0, 0.0])
    download.assert_not_called()


def test_ner_download_loads_model_files_without_pip(tmp_path, monkeypatch, nlp):
    local_model = tmp_path / "ner"
    nlp.to_disk(local_model)
    classifier = DummyClassifier(strategy="constant", constant=0).fit([[1, 0, 0]], [0])
    model_file = tmp_path / "classifier.pkl"
    joblib.dump(classifier, model_file)
    snapshot = Mock(return_value=str(local_model))
    monkeypatch.setattr(extraction, "snapshot_download", snapshot)
    single_file = Mock(return_value=str(model_file))
    monkeypatch.setattr(extraction, "hf_hub_download", single_file)
    result = ExtractConfig.create("example/ner", "example/multiskill")
    assert result.nlp("test").text == "test"
    assert result.ms_model.predict(["Python"]).tolist() == [0]
    assert snapshot.call_args.kwargs["repo_id"] == "example/ner"
    assert "*.whl" not in snapshot.call_args.kwargs["allow_patterns"]
    assert single_file.call_args.kwargs["repo_id"] == "example/multiskill"


def test_failed_ner_download_is_actionable(monkeypatch):
    monkeypatch.setattr(
        extraction, "snapshot_download", Mock(side_effect=OSError("offline"))
    )
    with pytest.raises(RuntimeError, match="first use requires"):
        ExtractConfig.create("example/ner")


def test_invalid_model_name_raises_value_error():
    with pytest.raises(ValueError, match="ner_model_name"):
        ExtractConfig.create("")


def test_empty_vectorizer_input_has_correct_shape():
    vectorizer = BertVectorizer()
    vectorizer.bert_model = Mock()
    vectorizer.bert_model.get_sentence_embedding_dimension.return_value = 384
    assert vectorizer.transform([]).shape == (0, 384)
    vectorizer.bert_model.encode.assert_not_called()


def test_multiprocessing_pool_stops_when_encoding_fails():
    vectorizer = BertVectorizer(multi_process=True)
    vectorizer.bert_model = Mock()
    vectorizer.bert_model.encode_multi_process.side_effect = RuntimeError("failed")
    with pytest.raises(RuntimeError, match="failed"):
        vectorizer.transform(["Python"])
    vectorizer.bert_model.stop_multi_process_pool.assert_called_once_with(
        vectorizer.bert_model.start_multi_process_pool.return_value
    )
