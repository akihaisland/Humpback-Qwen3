import json
import datasets

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
A high-quality dataset for efficient instruction tuning.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "other"

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
}


class LimaConfig(datasets.BuilderConfig):
    """BuilderConfig"""

    def __init__(self, **kwargs):
        """BuilderConfig
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(LimaConfig, self).__init__(**kwargs)


class Lima(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        LimaConfig(
            name="plain_text",
            version=datasets.Version("0.0.1", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "conversations": datasets.features.Sequence(datasets.Value("string")),
                    "source": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": dl_manager.download("train.jsonl")}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath":dl_manager.download("test.jsonl")})
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        key = 0
        with open(filepath) as f:
            for line in f.readlines():
                instance = json.loads(line)
                yield key, instance
                key += 1