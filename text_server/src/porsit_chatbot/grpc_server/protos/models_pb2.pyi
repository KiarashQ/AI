from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class IntentResponse(_message.Message):
    __slots__ = ("intent", "confidence")
    INTENT_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    intent: str
    confidence: float
    def __init__(self, intent: _Optional[str] = ..., confidence: _Optional[float] = ...) -> None: ...

class Entity(_message.Message):
    __slots__ = ("word", "entity", "start", "end")
    WORD_FIELD_NUMBER: _ClassVar[int]
    ENTITY_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    word: str
    entity: str
    start: int
    end: int
    def __init__(self, word: _Optional[str] = ..., entity: _Optional[str] = ..., start: _Optional[int] = ..., end: _Optional[int] = ...) -> None: ...

class EntityResponse(_message.Message):
    __slots__ = ("entities",)
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    entities: _containers.RepeatedCompositeFieldContainer[Entity]
    def __init__(self, entities: _Optional[_Iterable[_Union[Entity, _Mapping]]] = ...) -> None: ...

class GenerateRequest(_message.Message):
    __slots__ = ("context", "prompt", "model_choice", "client_name")
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    MODEL_CHOICE_FIELD_NUMBER: _ClassVar[int]
    CLIENT_NAME_FIELD_NUMBER: _ClassVar[int]
    context: str
    prompt: str
    model_choice: str
    client_name: str
    def __init__(self, context: _Optional[str] = ..., prompt: _Optional[str] = ..., model_choice: _Optional[str] = ..., client_name: _Optional[str] = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("response",)
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    response: str
    def __init__(self, response: _Optional[str] = ...) -> None: ...

class SummarizeRequest(_message.Message):
    __slots__ = ("conversation_history", "client_name")
    CONVERSATION_HISTORY_FIELD_NUMBER: _ClassVar[int]
    CLIENT_NAME_FIELD_NUMBER: _ClassVar[int]
    conversation_history: str
    client_name: str
    def __init__(self, conversation_history: _Optional[str] = ..., client_name: _Optional[str] = ...) -> None: ...

class ClassifyAllRequest(_message.Message):
    __slots__ = ("text",)
    TEXT_FIELD_NUMBER: _ClassVar[int]
    text: str
    def __init__(self, text: _Optional[str] = ...) -> None: ...

class ClassifyRequestLC(_message.Message):
    __slots__ = ("text", "conversation_history", "client_name")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_HISTORY_FIELD_NUMBER: _ClassVar[int]
    CLIENT_NAME_FIELD_NUMBER: _ClassVar[int]
    text: str
    conversation_history: str
    client_name: str
    def __init__(self, text: _Optional[str] = ..., conversation_history: _Optional[str] = ..., client_name: _Optional[str] = ...) -> None: ...

class CombinedResponse(_message.Message):
    __slots__ = ("intent_response", "entity_response")
    INTENT_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    intent_response: IntentResponse
    entity_response: EntityResponse
    def __init__(self, intent_response: _Optional[_Union[IntentResponse, _Mapping]] = ..., entity_response: _Optional[_Union[EntityResponse, _Mapping]] = ...) -> None: ...
