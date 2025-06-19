from faker import Faker
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine, OperatorConfig
from presidio_anonymizer.operators import Operator, OperatorType

from .anonymizer import Anonymizer


# Original code from https://microsoft.github.io/presidio/samples/python/pseudonymization/
class InstanceCounterAnonymizer(Operator):
    """
    Anonymizer which replaces the entity value
    with an instance counter per entity.
    """

    REPLACING_FORMAT = "<{entity_type}_{index}>"

    def operate(self, text: str, params: dict = None) -> str:
        """Anonymize the input text."""

        entity_type: str = params["entity_type"]

        # entity_mapping is a dict of dicts containing mappings per entity type
        entity_mapping = params["entity_mapping"]

        entity_mapping_for_type = entity_mapping.get(entity_type)
        if not entity_mapping_for_type:
            new_text = self.REPLACING_FORMAT.format(entity_type=entity_type, index=0)
            entity_mapping[entity_type] = {}

        else:
            if text in entity_mapping_for_type:
                return entity_mapping_for_type[text]

            previous_index = self._get_last_index(entity_mapping_for_type)
            new_text = self.REPLACING_FORMAT.format(entity_type=entity_type, index=previous_index + 1)

        entity_mapping[entity_type][text] = new_text
        return new_text

    @staticmethod
    def _get_last_index(entity_mapping_for_type: dict) -> int:
        """Get the last index for a given entity type."""

        def get_index(value: str) -> int:
            return int(value.split("_")[-1][:-1])

        indices = [get_index(v) for v in entity_mapping_for_type.values()]
        return max(indices)

    def validate(self, params: dict = None) -> None:
        """Validate operator parameters."""

        if "entity_mapping" not in params:
            raise ValueError("An input Dict called `entity_mapping` is required.")
        if "entity_type" not in params:
            raise ValueError("An entity_type param is required.")

    def operator_name(self) -> str:
        return "entity_counter"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize


class UniquePlaceholderPerEntity(Anonymizer):
    def __init__(self):
        # Create Anonymizer engine and add the custom anonymizer
        self.analyzer = AnalyzerEngine()
        self.anonymizer_engine = AnonymizerEngine()
        self.anonymizer_engine.add_anonymizer(InstanceCounterAnonymizer)

    def anonymize(self, text) -> str:
        entity_mapping = {}
        analyzer_results = self.analyzer.analyze(text=text, language="en")
        anonymized_result = self.anonymizer_engine.anonymize(
            text,
            analyzer_results,
            {"DEFAULT": OperatorConfig("entity_counter", {"entity_mapping": entity_mapping})},
        )
        return anonymized_result.text


class DeletionAnonymizer(Operator):
    """
    Anonymizer which deletes entities.
    """

    REPLACING_FORMAT = ""

    def operate(self, text: str, params: dict = None) -> str:
        """Anonymize the input text."""
        new_text = self.REPLACING_FORMAT
        return new_text

    @staticmethod
    def _get_last_index(entity_mapping_for_type: dict) -> int:
        """Get the last index for a given entity type."""

        def get_index(value: str) -> int:
            return int(value.split("_")[-1][:-1])

        indices = [get_index(v) for v in entity_mapping_for_type.values()]
        return max(indices)

    def validate(self, params: dict = None) -> None:
        """Validate operator parameters."""
        pass

    def operator_name(self) -> str:
        return "entity_remover"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize


class EntityDeletion(Anonymizer):
    def __init__(self):
        # Create Anonymizer engine and add the custom anonymizer
        self.analyzer = AnalyzerEngine()
        self.anonymizer_engine = AnonymizerEngine()
        self.anonymizer_engine.add_anonymizer(DeletionAnonymizer)

    def anonymize(self, text) -> str:
        analyzer_results = self.analyzer.analyze(text=text, language="en")
        anonymized_result = self.anonymizer_engine.anonymize(
            text,
            analyzer_results,
            {"DEFAULT": OperatorConfig("entity_remover")},
        )
        return anonymized_result.text


class PlaceholderAnonymizer(Operator):
    """
    Anonymizer which replaces the entity value
    with a <ENTITY> placeholder.
    """

    REPLACING_FORMAT = "<ENTITY>"

    def operate(self, text: str, params: dict = None) -> str:
        """Anonymize the input text."""
        new_text = self.REPLACING_FORMAT
        return new_text

    @staticmethod
    def _get_last_index(entity_mapping_for_type: dict) -> int:
        """Get the last index for a given entity type."""

        def get_index(value: str) -> int:
            return int(value.split("_")[-1][:-1])

        indices = [get_index(v) for v in entity_mapping_for_type.values()]
        return max(indices)

    def validate(self, params: dict = None) -> None:
        """Validate operator parameters."""
        pass

    def operator_name(self) -> str:
        return "entity_placeholder"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize


class UniformPlaceholder(Anonymizer):
    def __init__(self):
        # Create Anonymizer engine and add the custom anonymizer
        self.analyzer = AnalyzerEngine()
        self.anonymizer_engine = AnonymizerEngine()
        self.anonymizer_engine.add_anonymizer(PlaceholderAnonymizer)

    def anonymize(self, text) -> str:
        analyzer_results = self.analyzer.analyze(text=text, language="en")
        anonymized_result = self.anonymizer_engine.anonymize(
            text,
            analyzer_results,
            {"DEFAULT": OperatorConfig("entity_placeholder")},
        )
        return anonymized_result.text


class CategoryAnonymizer(Operator):
    """
    Anonymizer which replaces the entity value
    with the associated category.
    """

    REPLACING_FORMAT = "<{entity_type}>"

    def operate(self, text: str, params: dict = None) -> str:
        """Anonymize the input text."""

        entity_type: str = params["entity_type"]

        # entity_mapping is a dict of dicts containing mappings per entity type
        entity_mapping = params["entity_mapping"]

        entity_mapping_for_type = entity_mapping.get(entity_type)
        if not entity_mapping_for_type:
            new_text = self.REPLACING_FORMAT.format(entity_type=entity_type)
            entity_mapping[entity_type] = {}

        else:
            if text in entity_mapping_for_type:
                return entity_mapping_for_type[text]

            new_text = self.REPLACING_FORMAT.format(entity_type=entity_type)

        entity_mapping[entity_type][text] = new_text
        return new_text

    def validate(self, params: dict = None) -> None:
        """Validate operator parameters."""

        if "entity_mapping" not in params:
            raise ValueError("An input Dict called `entity_mapping` is required.")
        if "entity_type" not in params:
            raise ValueError("An entity_type param is required.")

    def operator_name(self) -> str:
        return "entity_category"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize


class CategoryPlaceholder(Anonymizer):
    def __init__(self):
        # Create Anonymizer engine and add the custom anonymizer
        self.analyzer = AnalyzerEngine()
        self.anonymizer_engine = AnonymizerEngine()
        self.anonymizer_engine.add_anonymizer(CategoryAnonymizer)

    def anonymize(self, text) -> str:
        entity_mapping = {}
        analyzer_results = self.analyzer.analyze(text=text, language="en")
        anonymized_result = self.anonymizer_engine.anonymize(
            text,
            analyzer_results,
            {"DEFAULT": OperatorConfig("entity_category", {"entity_mapping": entity_mapping})},
        )
        return anonymized_result.text


class FakerAnonymizer(Operator):
    """
    Anonymizer which replaces the entity value
    with a Faker-generated fake entity value.
    """

    def __init__(self):
        self.faker = Faker()

    def operate(self, text: str, params: dict = None) -> str:
        """Anonymize the input text with Faker values."""
        entity_type: str = params["entity_type"]
        # entity_mapping is a dict of dicts containing mappings per entity type
        entity_mapping = params["entity_mapping"]
        entity_mapping_for_type = entity_mapping.get(entity_type)

        if not entity_mapping_for_type:
            entity_mapping[entity_type] = {}
            entity_mapping_for_type = entity_mapping[entity_type]

        if text in entity_mapping_for_type:
            return entity_mapping_for_type[text]

        # Generate appropriate fake data based on entity type
        fake_value = self._generate_fake_value(entity_type)
        entity_mapping[entity_type][text] = fake_value
        return fake_value

    def _generate_fake_value(self, entity_type: str) -> str:
        """Generate appropriate fake value based on entity type."""
        entity_type = entity_type.lower()

        if "person" in entity_type or "name" in entity_type:
            return self.faker.name()
        elif "phone" in entity_type:
            return self.faker.phone_number()
        elif "email" in entity_type:
            return self.faker.email()
        elif "address" in entity_type:
            return self.faker.address().replace("\n", ", ")
        elif "credit" in entity_type or "card" in entity_type:
            return self.faker.credit_card_number()
        elif "ssn" in entity_type or "social" in entity_type:
            return self.faker.ssn()
        elif "date" in entity_type or "birth" in entity_type:
            return self.faker.date()
        elif "ip" in entity_type:
            return self.faker.ipv4()
        elif "url" in entity_type:
            return self.faker.url()
        elif "company" in entity_type or "org" in entity_type:
            return self.faker.company()
        elif "location" in entity_type or "city" in entity_type:
            return self.faker.city()
        elif "country" in entity_type:
            return self.faker.country()
        elif "iban" in entity_type:
            return self.faker.iban()
        elif "passport" in entity_type:
            return self.faker.passport_number()
        else:
            # Default for unknown entity types
            return self.faker.word()

    def validate(self, params: dict = None) -> None:
        """Validate operator parameters."""
        if "entity_mapping" not in params:
            raise ValueError("An input Dict called `entity_mapping` is required.")
        if "entity_type" not in params:
            raise ValueError("An entity_type param is required.")

    def operator_name(self) -> str:
        return "faker_anonymizer"

    def operator_type(self) -> OperatorType:
        return OperatorType.Anonymize

class FakerPlaceholder(Anonymizer):
    def __init__(self):
        # Create Anonymizer engine and add the custom anonymizer
        self.analyzer = AnalyzerEngine()
        self.anonymizer_engine = AnonymizerEngine()
        self.anonymizer_engine.add_anonymizer(FakerAnonymizer)

    def anonymize(self, text) -> str:
        entity_mapping = {}
        analyzer_results = self.analyzer.analyze(text=text, language="en")
        anonymized_result = self.anonymizer_engine.anonymize(
            text,
            analyzer_results,
            {"DEFAULT": OperatorConfig("faker_anonymizer", {"entity_mapping": entity_mapping})},
        )
        return anonymized_result.text
