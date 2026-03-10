from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class InteractionTableContract:
    """
    Defines the expected structure of the Interaction DuckDB
    brought in from the DataIntake repository.
    """

    TABLE_NAME: str = "interaction_table"

    # Column Names
    HOST: str = "host"
    PHAGE: str = "phage"
    INTERACTION: str = "interaction"
    SOURCE: str = "source"

    # Standard Values
    VALUE_POSITIVE: int = 1
    VALUE_NEGATIVE: int = 0
    VALUE_UNKNOWN: int = -1

    @property
    def required_columns(self) -> List[str]:
        return [self.HOST, self.PHAGE, self.INTERACTION, self.SOURCE]
