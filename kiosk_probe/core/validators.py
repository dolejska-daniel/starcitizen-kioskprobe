from InquirerPy.validator import NumberValidator
from prompt_toolkit.validation import ValidationError


class FloatValidator(NumberValidator):
    def __init__(self, min_value: float = float("-inf"), max_value: float = float("inf")):
        super().__init__(float_allowed=True)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, document):
        super().validate(document)
        value = float(document.text)
        if not value >= self.min_value:
            raise ValidationError(
                message=f"Value must be >= {self.min_value}",
                cursor_position=document.cursor_position
            )

        if not value <= self.max_value:
            raise ValidationError(
                message=f"Value must be <= {self.min_value}",
                cursor_position=document.cursor_position
            )
