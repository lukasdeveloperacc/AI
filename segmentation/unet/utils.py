from typing import Tuple
import pandas as pd
import logging


class DatasetUtil:
    def __init__(self):
        pass

    @staticmethod
    def train_test_split_from_df(
        df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        shuffled_df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        end = int(len(shuffled_df) * (1 - test_size))
        train_df = shuffled_df.iloc[:end]
        test_df = shuffled_df.iloc[end:]

        logging.info(f"Train-test split: {len(train_df)} train samples, {len(test_df)} test samples")

        return train_df, test_df
