"""End-to-end checks against actual pretrained models and bundled taxonomies."""

import pytest

from ojd_daps_skills.extract_skills.extract_skills import SkillsExtractor
from ojd_daps_skills.extract_skills.extract_skills_utils import ExtractConfig
from ojd_daps_skills.map_skills.skill_mapper_utils import MapConfig

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def extract_config():
    return ExtractConfig.create()


@pytest.mark.parametrize("taxonomy", ["toy", "esco", "lightcast"])
def test_live_taxonomy(taxonomy, extract_config):
    extractor = SkillsExtractor(
        taxonomy_name=taxonomy,
        extract_config=extract_config,
        map_config=MapConfig.create(taxonomy),
    )
    docs = extractor(
        [
            "The job involves Excel skills. You will also need good presentation skills.",
            "We need experience using Python and SQL to analyse data.",
            "",
            "Hello.",
        ]
    )
    assert len(docs) == 4
    assert docs[0]._.skill_spans
    assert any(match for match in docs[0]._.mapped_skills)
    assert docs[1]._.skill_spans
    assert docs[2]._.skill_spans == []
    assert docs[2]._.mapped_skills == []
    assert extractor([]) == []
