import tasknet as tn
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset

class IMDBAuthorshipClassification(tn.Classification):
    """
    A classification task for authorship attribution using the IMDb-62 dataset.

    Inherits from tasknet.Classification and automatically processes the dataset
    to select authors with sufficient documents for classification.
    """

    def __init__(self, n_authors: int = 10, min_docs_per_author: int = 1000,
                 random_seed: int = 0, **kwargs):
        """
        Initialize the authorship classification task.

        Args:
            n_authors: Number of authors to include in the classification task
            min_docs_per_author: Minimum number of documents required per author
            random_seed: Random seed for reproducible author selection
            **kwargs: Additional arguments passed to parent Classification class
        """
        self.n_authors = n_authors
        self.min_docs_per_author = min_docs_per_author
        self.random_seed = random_seed

        # Load and process the dataset
        processed_data = self._load_and_process_dataset()

        # Initialize parent class with processed data
        super().__init__(
            dataset=tn.utils.train_validation_test_split(processed_data),
            s1="text",
            y="labels",
            **kwargs
        )

        # Set task name
        self.name = f"imdb_authorship_{n_authors}_authors"

    def _load_and_process_dataset(self) -> Dataset:
        """
        Load IMDb-62 dataset and process it for authorship classification.

        Returns:
            Dataset: Processed dataset with selected authors and formatted columns
        """
        # Load the IMDb-62 dataset
        dataset = load_dataset("tasksource/imdb62")
        df = pd.DataFrame(dataset["train"])

        # Get author document counts
        author_counts = df["userId"].value_counts()

        # Select authors with sufficient documents
        eligible_authors = author_counts[author_counts >= self.min_docs_per_author].index

        if len(eligible_authors) < self.n_authors:
            raise ValueError(
                f"Only {len(eligible_authors)} authors have >= {self.min_docs_per_author} "
                f"documents, but {self.n_authors} authors requested."
            )

        # Randomly select n_authors from eligible authors
        rng = np.random.default_rng(self.random_seed)
        selected_authors = rng.choice(
            np.sort(eligible_authors),
            size=self.n_authors,
            replace=False
        )

        # Filter dataset to selected authors
        filtered_df = df[df["userId"].isin(selected_authors)].dropna()

        # Format for classification task
        processed_df = pd.DataFrame({
            "labels": filtered_df["userId"].values,
            "text": filtered_df["content"].values,
        })

        # Convert to Dataset and encode labels
        data = Dataset.from_dict(processed_df)
        data = data.class_encode_column("labels")

        return data

    @property
    def author_info(self) -> dict:
        """Get information about selected authors and their document counts."""
        # Reconstruct the original data to get author info
        dataset = load_dataset("tasksource/imdb62")
        df = pd.DataFrame(dataset["train"])
        author_counts = df["userId"].value_counts()

        rng = np.random.default_rng(self.random_seed)
        eligible_authors = author_counts[author_counts >= self.min_docs_per_author].index
        selected_authors = rng.choice(np.sort(eligible_authors), size=self.n_authors, replace=False)

        return {
            "selected_authors": selected_authors.tolist(),
            "author_doc_counts": {author: author_counts[author] for author in selected_authors},
            "total_documents": sum(author_counts[author] for author in selected_authors)
        }
