class TextAreaInput:
    """
    Simple input node that provides a multi-line text area and outputs the entered text.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "在这里输入多行文本",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_text"
    CATEGORY = "text"

    def get_text(self, text: str):
        return (text,)


__all__ = ["TextAreaInput"]
