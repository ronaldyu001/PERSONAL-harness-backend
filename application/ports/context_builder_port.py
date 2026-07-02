from typing import Protocol



class ContextBuilderPort(Protocol):
    async def build(self, )->Context: 
        ...