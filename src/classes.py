from pydantic import BaseModel


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class Prompt(BaseModel):
    prompt: str
