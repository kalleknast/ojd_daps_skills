"""
Utility functions and a configuration manager for
extracting skills from job descriptions.
"""

from typing import List, Optional

import joblib
import spacy
from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from spacy.language import Language

from ojd_daps_skills import PUBLIC_MODEL_FOLDER_PATH


### Define Multiskill Transformer
class MultiSkillTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting necessary, just return self
        return self

    def transform(self, X):
        """Apply the transform_skill function to each element in X.

        Args:
            X (iterable of str): The data to transform.

        Returns:
            List[List[int]]: Transformed data, where each item is the output of transform_skill.
        """
        return [self.transform_skill(skill) for skill in X]

    @staticmethod
    def transform_skill(skill: str) -> List[int]:
        """Transform skill into a list of features. The features are:
            - length of skill span;
            - presence of " and " in skill span;
            - presence of "," in skill span.

        Args:
            skill (str): skill span.

        Returns:
            List[int]: list of integers.
        """
        return [len(skill), int(" and " in skill), int("," in skill)]


class ExtractConfig(BaseModel):
    """
    Configuration manager for EXTRACTING skills using specific NLP models.

    Attributes:
        ner_model_name (str): The name of the Named Entity Recognition model to
            use from HuggingFace Hub. Current configuration supports "nestauk/en_skillner".
            You can use your own NER model if you have a custom NER model to extract skills.
        ms_model_name (str): The name of the Multi-Skill model to use. Current configurations
            supports "nestauk/multiskill-classifier".
        nlp (Optional[Language]): The NLP model to use for Named Entity Recognition. This
            is set during creation.
        ms_model (Optional[Pipeline]): The SVM model to use for Multi-Skill classification.
            This is set during creation.
    """

    ner_model_name: str = "nestauk/en_skillner"
    ms_model_name: str = "nestauk/multiskill-classifier"
    nlp: Optional[Language] = None  # Optional, since it's set during creation
    ms_model: Optional[Pipeline] = None  # Optional for the same reason

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(
        cls,
        ner_model_name: str = "nestauk/en_skillner",
        ms_model_name: str = "nestauk/multiskill-classifier",
    ) -> "ExtractConfig":
        """
        Creates an instance of ExtractConfig by loading configurations.

        Parameters:
            ner_model_name (Optional[str]): The name of the NER model to use. Defaults
                to "nestauk/en_skillner".
            ms_model_name (Optional[str]): The name of the Multi-Skill model to use.
                Defaults to "nestauk/multiskill-classifier".

        Returns:
            ExtractConfig: An initialized instance of this configuration class.

        Raises:
            OSError: A local spaCy model cannot be loaded.
            RuntimeError: A remote NER model cannot be downloaded or loaded.
            ValueError: A model identifier is invalid.
        """
        if not isinstance(ner_model_name, str) or not ner_model_name.strip():
            raise ValueError(
                "ner_model_name must be a spaCy model, path, or Hub repository ID."
            )
        if not isinstance(ms_model_name, str) or "/" not in ms_model_name:
            raise ValueError(
                "ms_model_name must be a Hugging Face namespace/repository ID."
            )

        # Use an installed model or local directory if supplied. Never run pip
        # inside a library call: load the published spaCy files directly instead.
        installed_name = (
            "en_skillner" if ner_model_name == "nestauk/en_skillner" else ner_model_name
        )
        try:
            nlp = spacy.load(installed_name)
        except OSError as error:
            if "/" not in ner_model_name or ner_model_name.startswith(("/", ".")):
                raise OSError(
                    f"spaCy model or directory not found: {ner_model_name}"
                ) from error
            try:
                model_path = snapshot_download(
                    repo_id=ner_model_name,
                    cache_dir=PUBLIC_MODEL_FOLDER_PATH / "hub",
                    allow_patterns=[
                        "config.cfg",
                        "meta.json",
                        "tokenizer",
                        "attribute_ruler/*",
                        "lemmatizer/*",
                        "ner/*",
                        "parser/*",
                        "senter/*",
                        "tagger/*",
                        "tok2vec/*",
                        "vocab/*",
                    ],
                )
                nlp = spacy.load(model_path)
            except (OSError, HfHubHTTPError) as error:
                raise RuntimeError(
                    f"Could not load NER model {ner_model_name!r}. The first use requires "
                    "access to huggingface.co and its download hosts; subsequent offline "
                    "runs can use the cache with HF_HUB_OFFLINE=1."
                ) from error

        # joblib deserializes a trusted Nesta classifier. Custom model repositories
        # must also be trusted. Cache by repo/revision so models cannot collide.
        model_file = hf_hub_download(
            repo_id=ms_model_name,
            filename="multiskill-classifiert4_v38_0.pkl",
            cache_dir=PUBLIC_MODEL_FOLDER_PATH / "hub",
        )
        clf = joblib.load(model_file)

        ms_model = Pipeline(
            [("transformer", MultiSkillTransformer()), ("classifier", clf)]
        )

        return cls(
            ner_model_name=ner_model_name,
            ms_model_name=ms_model_name,
            nlp=nlp,
            ms_model=ms_model,
        )
