from langchain_core.tracers.langchain import LangChainTracer


class RootOnlyTracer(LangChainTracer):
    """Emit the first chain span, then silence its children."""

    # _root_seen: bool = False

    # read‑only property → override to change behaviour dynamically
    @property
    def ignore_chain(self) -> bool:  # type: ignore[override]
        return True
        # return self._root_seen  # False for the first run, True afterwards

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return False

    @property
    def ignore_retry(self) -> bool:
        """Whether to ignore retry callbacks."""
        return False

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return False

    @property
    def ignore_retriever(self) -> bool:
        """Whether to ignore retriever callbacks."""
        return False

    @property
    def ignore_chat_model(self) -> bool:
        """Whether to ignore chat model callbacks."""
        return False

    @property
    def ignore_custom_event(self) -> bool:
        """Ignore custom event."""
        return False

    # mark that we have emitted the root span
    # def on_chain_start(self, *args, **kwargs):
    #     super().on_chain_start(*args, **kwargs)
    #     self._root_seen = True
