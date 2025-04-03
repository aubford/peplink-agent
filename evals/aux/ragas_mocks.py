from typing import Any
from ragas.prompt import PydanticPrompt
from ragas.testset.synthesizers.multi_hop.prompts import ConceptCombinations, GeneratedQueryAnswer
from ragas.testset.synthesizers.prompts import PersonaThemesMapping
from ragas.testset.synthesizers import MultiHopAbstractQuerySynthesizer


class MockConceptCombinationPrompt(PydanticPrompt):
    async def generate(self, data: Any, llm: Any, callbacks=None):
        # Return a predetermined ConceptCombinations object
        return ConceptCombinations(
            combinations=[
                ["Network Configuration", "Satellite Internet"],
                ["VPN Setup", "Firmware Updates"],
                ["Load Balancing", "Bandwidth Management"]
            ]
        )


class MockThemePersonaMatchingPrompt(PydanticPrompt):
    async def generate(self, data: Any, llm: Any, callbacks=None):
        # Return a predetermined PersonaThemesMapping object
        return PersonaThemesMapping(
            mapping={
                "Technical Network Engineer": [
                    "Network Configuration", 
                    "VPN Setup",
                    "Load Balancing", 
                    "Satellite Internet",
                    "Firmware Updates",
                    "Bandwidth Management"
                ]
            }
        )


class MockQueryAnswerGenerationPrompt(PydanticPrompt):
    async def generate(self, data: Any, llm: Any, callbacks=None):
        # Return a predetermined GeneratedQueryAnswer object
        return GeneratedQueryAnswer(
            query="How does Pepwave's firmware update process affect VPN configurations on devices connected to Starlink?",
            answer="When updating Pepwave firmware, VPN configurations may require reconfiguration. The process involves backing up current settings, applying the firmware update, and then verifying that the VPN configurations still function correctly with Starlink connections. Some newer firmware versions include automatic compatibility checks for satellite internet connections like Starlink."
        )


class MockMultiHopAbstractQuerySynthesizer(MultiHopAbstractQuerySynthesizer):
    """
    A mock version of MultiHopAbstractQuerySynthesizer that replaces the prompts with
    mocks that return predetermined values instead of calling an LLM.
    """
    def __init__(self, llm):
        super().__init__(llm=llm)
        self.concept_combination_prompt = MockConceptCombinationPrompt()
        self.theme_persona_matching_prompt = MockThemePersonaMatchingPrompt()
        self.generate_query_reference_prompt = MockQueryAnswerGenerationPrompt() 
