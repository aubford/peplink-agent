from abc import ABC, abstractmethod
from pydantic import BaseModel
from langchain_core.documents import Document
from load.batch_manager import BatchManager
from openai.lib._parsing._completions import type_to_response_format_param
from pathlib import Path
import tiktoken
import json


class ModelResponse(BaseModel):
    technical_summary: str  # summary of post information
    is_useful: bool  # if false we will filter it out
    themes: list[str]  # themes of the post


class SyntheticDataLoader(ABC):
    batch_manager: BatchManager = NotImplemented

    @abstractmethod
    def _create_system_prompt(self) -> str:
        """Creates the system prompt that instructs the model on its task."""
        pass

    @abstractmethod
    def create_prompt(self, primary_content: str, lead_content: str) -> str:
        """Creates a prompt that includes content."""
        pass

    @abstractmethod
    def _get_examples(self) -> list[dict]:
        """
        Provides examples of content and ideal responses.

        Returns:
            List of example dictionaries.
        """
        pass

    def create_system_prompt_with_examples(self) -> str:
        """
        Creates a system prompt that includes both instructions and examples.

        Returns:
            Complete system prompt with instructions and examples
        """
        base_prompt = self._create_system_prompt()
        examples = self._get_examples()

        examples_text = "\n\nHere are some examples and expected analyses:\n\n"

        for i, example in enumerate(examples, 1):
            # Format the example conversation
            example_prompt = self.create_prompt(
                primary_content=example["primary_content"],
                lead_content=example.get("lead_content", ""),
            )

            # Format the expected output
            output = example["expected_output"]
            themes_str = ", ".join([f'"{theme}"' for theme in output["themes"]])

            expected_output = (
                f'Expected output:\n'
                f'{{\n'
                f'  "themes": [{themes_str}],\n'
                f'  "technical_summary": "{output["technical_summary"]}",\n'
                f'  "is_useful": {output["is_useful"]}\n'
                f'}}\n'
            )

            examples_text += (
                f"EXAMPLE {i}:\n\n{example_prompt}\n\n{expected_output}\n{'=' * 40}\n\n"
            )

        return base_prompt + examples_text

    # max_tokens counts towards the total tokens used for the batch job even if the response is shorter
    def create_batch_job(self, documents: list[Document], max_tokens: int = 500):
        """Create a batch job for processing documents."""
        if self.batch_manager is None:
            raise ValueError("batch_manager must be set before creating a batch job")

        # Create batch items from documents
        batch_items = []
        for doc in documents:
            if not doc.id:
                raise ValueError("Document ID is required for batch processing")
            lead_content = doc.metadata.get("lead_content", "")
            primary_content = doc.metadata.get("primary_content", "")

            batch_items.append(
                {
                    "id": doc.id,
                    "prompt": self.create_prompt(primary_content, lead_content),
                }
            )

        # If we have batch items, create and run a batch job
        if batch_items:
            system_prompt = self.create_system_prompt_with_examples()
            self.batch_manager.create_batch_tasks(
                items=batch_items,
                schema=ModelResponse,
                system_prompt=system_prompt,
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=max_tokens,
            )
        self.batch_manager.test_batchfile()
        self.batch_manager.create_batch_job()

    def create_capped_batchfiles(
        self, documents: list[Document], max_tokens: int = 500
    ) -> list[str]:
        """
        Create batch files ensuring no file exceeds 18M tokens. Returns paths to created batch files.

        Args:
            documents: List of documents to process
            max_tokens: Maximum tokens for model response (default: 300)

        Returns:
            List of paths to created batch files
        """
        if self.batch_manager is None:
            raise ValueError("batch_manager must be set before creating batch files")

        # Create batch items from documents
        batch_items = []
        for doc in documents:
            if not doc.id:
                raise ValueError("Document ID is required for batch processing")
            lead_content = doc.metadata.get("lead_content", "")
            primary_content = doc.metadata.get("primary_content", "")

            batch_items.append(
                {
                    "id": doc.id,
                    "prompt": self.create_prompt(primary_content, lead_content),
                }
            )

        if not batch_items:
            return []

        # Get system prompt
        system_prompt = self.create_system_prompt_with_examples()

        # Create tasks with full configuration
        tasks = []
        for item in batch_items:
            task = {
                "custom_id": item["id"],
                "method": "POST",
                "url": self.batch_manager.endpoint,
                "body": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": item["prompt"]},
                    ],
                    "response_format": type_to_response_format_param(ModelResponse),
                },
            }
            tasks.append(task)

        enc = tiktoken.get_encoding("cl100k_base")

        # Split tasks into chunks that don't exceed 8M tokens
        MAX_TOKENS = 8_000_000
        current_chunk = []
        current_tokens = 0
        chunks = []

        for task in tasks:
            # Count tokens in this task
            task_tokens = (
                len(enc.encode(task["body"]["messages"][0]["content"]))  # System prompt
                + len(enc.encode(task["body"]["messages"][1]["content"]))  # User prompt
                + task["body"]["max_tokens"]  # Max response tokens
            )

            # If adding this task would exceed limit, save current chunk and start new one
            if current_tokens + task_tokens > MAX_TOKENS:
                if current_chunk:  # Only append if there are tasks in the chunk
                    chunks.append(current_chunk)
                current_chunk = [task]
                current_tokens = task_tokens
            else:
                current_chunk.append(task)
                current_tokens += task_tokens

        # Add the last chunk if it has any tasks
        if current_chunk:
            chunks.append(current_chunk)

        batch_files = []
        batch_path = Path(self.batch_manager.file_name).parent

        for i, chunk in enumerate(chunks):
            file_path = batch_path / f"batchfile_{i}.jsonl"
            with open(file_path, "w") as f:
                for task in chunk:
                    f.write(json.dumps(task) + "\n")
            batch_files.append(str(file_path))

        return batch_files


class ForumSyntheticDataLoader(SyntheticDataLoader):
    """
    Loader class to generate prompts for OpenAI API to extract structured data from forum posts.
    The prompt facilitates transformation of a document's primary_content and lead_content
    into a structured format for analysis.
    """

    def _create_system_prompt(self) -> str:
        """Creates the system prompt that instructs the model on its task."""
        return """You are an expert technical content analyzer specializing in IT networking and Pepwave products.
Your task is to analyze forum conversations and extract key information.

You will be provided with a forum conversation consisting of:
1. The original forum post/question.
2. A response to the original post.

Together, these form a single conversation turn between two forum users.

Analyze this conversation and provide a structured output containing:
1. "themes": A list of technical themes discussed.
2. "technical_summary": A summary of the most useful technical information that can be gleaned from the conversation. Report only the technical facts presented in the conversation, do not discuss the conversation itself. Statements made as part of an inquiry should not be considered technical facts. Provide this in the form of a paragraph. No lists or special formatting.
3. "is_useful": An assessment of whether there is useful technical information related to Pepwave products or IT networking.

Important guidelines:
- Focus only on technical content and information in your analysis.
- "Useful technical information" means factual statements, not questions or inquiries.
- Be specific and precise in identifying themes.
- Base your analysis only on the provided content, do not make assumptions.

REMEMBER, DO NOT REFER TO THE CONVERSATION ITSELF IN YOUR SUMMARY OR THEMES, ONLY THE TECHINCAL FACTS!
"""

    def create_prompt(self, primary_content: str, lead_content: str) -> str:
        """Creates a prompt that includes the forum post content."""

        return f"""# Forum Post (Original Post/Question):
{lead_content}

# Response to the Original Post:
{primary_content}

Analyze this conversation according to the guidelines provided.
"""

    def _get_examples(self) -> list[dict]:
        """
        Provides examples of conversations and ideal responses.

        Returns:
            List of example dictionaries.
        """
        return [
            {
                "lead_content": "I just installed a Pepwave MAX Transit Duo-CAT12 in my RV, but I'm having trouble with the cellular connection. The signal strength is showing only 2 bars even though my phone gets 4 bars in the same location. Does anyone know why this might be happening?",
                "primary_content": "Hi Don,\n\nCheck your antenna connections first. The MAX Transit Duo requires proper external antennas to get the best signal. Make sure you're using the right cellular antennas and they're properly connected to the correct ports (they're labeled CELL on the router). Also, try changing the SIM priority in the admin panel - go to Network > Mobile > Settings, and you can change which SIM card is used or enable band locking for better performance on specific carriers. If you're in a fringe area, enabling band locking to the lower frequencies (like Band 12, 13, or 71 depending on your carrier) might help with penetration and range.",
                "expected_output": {
                    "themes": [
                        "cellular signal strength",
                        "antenna configuration",
                        "Pepwave MAX Transit Duo",
                        "SIM card settings",
                        "band locking",
                        "RV networking",
                    ],
                    "technical_summary": "To troubleshoot poor cellular signal on a Pepwave MAX Transit Duo-CAT12 when another device in the same location has good signal, it is recommended to check antenna connections to the correct ports, adjust SIM priority in the admin panel, and enable band locking for specific carriers, particularly lower frequency bands for better range in fringe areas.",
                    "is_useful": True,
                },
            },
            {
                "lead_content": "Anyone have recommendations for a good backup internet solution? I work from home and need something reliable when my main fiber connection goes down.",
                "primary_content": "I've been there! After trying several options, I settled on a Peplink Balance 20X with a 5G capable modem. The SpeedFusion technology in the Peplink devices is amazing for combining connections. I use it with both my fixed connection and a cellular backup, and the handover between them is completely seamless. I can be on a Zoom call and if my main connection fails, the call doesn't drop at all because of the Hot Failover feature. It's not cheap but worth every penny for reliability.",
                "expected_output": {
                    "themes": [
                        "backup internet solutions",
                        "Peplink Balance 20X",
                        "SpeedFusion technology",
                        "combining connections",
                        "Hot Failover",
                        "Work from home setup",
                        "videoconferencing",
                    ],
                    "technical_summary": "When in need of a backup internet solution for a work from home setup, a Peplink Balance 20X with a 5G capable modem is a good option. Its SpeedFusion technology is amazing for combining connections. The Hot Failover feature provides seamless transition between primary and backup connections, maintaining continuity for applications like video calls.",
                    "is_useful": True,
                },
            },
            {
                "lead_content": "Does anyone know if Peplink routers work with AT&T FirstNet? I'm looking to set up a mobile command center for our emergency response team.",
                "primary_content": "No idea, I've never used FirstNet. Have you tried contacting Peplink support directly? They might have better information about carrier compatibility.",
                "expected_output": {
                    "themes": [
                        "AT&T FirstNet compatibility",
                        "Peplink routers",
                        "emergency response equipment",
                    ],
                    "technical_summary": "To find out if Peplink routers work with AT&T FirstNet, one should contact Peplink support directly. The have better information about carrier compatibility.",
                    "is_useful": False,
                },
            },
        ]


class YouTubeSyntheticDataLoader(SyntheticDataLoader):
    """
    Loader class to generate prompts for OpenAI API to extract structured data from YouTube videos.
    The prompt facilitates transformation of a document's primary_content (transcript)
    into a structured format for analysis.
    """

    def _create_system_prompt(self) -> str:
        """Creates the system prompt that instructs the model on its task."""
        return """You are an expert technical content analyzer specializing in IT networking and Pepwave products.
Your task is to analyze a YouTube video transcript and extract key information.

You will be provided with an excerpt of a transcript of a YouTube video.

Analyze this content and provide a structured output containing:
1. "themes": A list of technical themes discussed in the video.
2. "technical_summary": A concise summary of the most useful technical information that can be gleaned from the video transcript. Report only the technical facts presented in the video, do not discuss the video itself. Provide this in the form of a paragraph. No lists or special formatting.
3. "is_useful": An assessment of whether there is enough useful technical information related to Pepwave products or IT networking to be worth watching the video.

Important guidelines:
- Focus only on technical content and information in your analysis.
- "Useful technical information" means factual statements, demonstrations, or tutorials and not statements made as part of a question or opinion.
- Be specific and precise in identifying themes.
- Provide a concise, factual summary that captures the key technical points.
- Base your analysis only on the provided content, do not make assumptions.
- YouTube transcripts are often imperfect, so do your best to extract meaning despite potential transcription errors.

REMEMBER, DO NOT REFER TO THE VIDEO ITSELF IN YOUR SUMMARY OR THEMES, ONLY THE TECHINCAL FACTS!
"""

    def create_prompt(self, primary_content: str, lead_content: str = "") -> str:
        """Creates a prompt that includes the YouTube video transcript."""

        return f"""# Video Transcript:
    
{primary_content}

Analyze this content according to the guidelines provided.
"""

    def _get_examples(self) -> list[dict]:
        """
        Provides examples of YouTube content and ideal responses.
        Note: For YouTube, we only use the primary_content field (video transcript)
        and ignore the lead_content field.

        Returns:
            List of example dictionaries.
        """
        return [
            {
                "primary_content": "but this is the puma 221 right here you can get up to 7 db gain increase with this unit and when i say up to seven it it it increases the gain different on different frequencies across the cellular bands and for wi-fi and with the br1 because i don't think i mentioned this and i should have it's a single band wi-fi which means you're going to have access to to the 2.4 band which is a great band it's a band that that operates great at further distances away but this is not a dual band wi-fi now we do have dual band wi-fi where you get access to the 2.4 and the five gigahertz wi-fi bands but again this is a single band here anyway the puma 221 this would be something that you would mount on the roof of your rv and it's called a five in one because there's basically five cables into one so you're gonna have two that are going to be your cellular connection so you would basically just unscrew these two paddle antennas and screw these in you're going to have two wi-fi's but because this is a single router then one of these is actually not going to be used you'll just use one of these and then you're going to have your your gps connection right here now if you if you up front did not activate the failover license to get wi-fi then both of these wi-fi connections would not be connected to this unit but we sell it with them because we feel like that even if you did not get the wi-fi immediately that down the road you might want it so if you had put this antenna up you just want the availability of it so that's the the puma 221 and it is a a great option the other thing that you'll see that we have as a recommendation is we do have the pointing antenna and this is a directional antenna and this is a two in one that that you can see here and this is for cellular only so this would be something that while the puma is omnidirectional it sees in all directions the the pointing x-pole is directional so you would need to point this into the direction of where the cell tower is and oftentimes a directional antenna can be great because it it basically eliminates other noise that might be coming from the 360 degrees and it just focuses in in one area",
                "expected_output": {
                    "themes": [
                        "Antenna Types and Directionality",
                        "Router Compatibility",
                        "Antenna Gain and Performance",
                        "Router Connectivity Options",
                        "Poynting XPOL",
                        "Upgrade and Flexibility Considerations",
                        "Puma 221",
                    ],
                    "technical_summary": "The Puma 221 is a roof-mounted, omnidirectional 5-in-1 antenna designed for mobile applications such as RVs, offering up to 7 dB of signal gain across cellular and Wi-Fi frequencies. It includes two cellular connectors, two Wi-Fi connectors (with only one used on single-band routers like the Peplink BR1, which supports 2.4 GHz Wi-Fi only), and one GPS connector. The included Wi-Fi connectors are present even if the failover Wi-Fi license is not activated, allowing for future upgrade flexibility. In contrast, the Poynting XPOL antenna is a 2-in-1, directional, cellular-only antenna that must be aimed toward a nearby cell tower. This directional design enhances signal quality by focusing reception and minimizing interference from other directions.",
                    "is_useful": True,
                },
            },
            {
                "primary_content": "Getting reliable internet connectivity at sea has a number of specific challenges particular to this difficult and ever-changing environment. So, what are your challenges? Network availability and speed of connections vary based on the vessel’s location. Peplink combines any connection to ensure connectivity is always available. Incumbent satellite solutions can no longer accommodate the rising bandwidth requirements for streaming IPTV, access to email and company servers as well as cloud services. To avoid data cost spiraling out of control, Peplink can automatically prioritize data usage to lower cost connections, such as port WiFi before LTE, or satellite. Here is an example. This is a 55 meter motor yacht. We installed a Pepwave HD4 MBX directly beneath where the antennas are. The MBX is capable of combining the bandwidth of up to four cellular links into an unbreakable, high-speed SD-WAN connection. It supports interchangeable LTE cellular modules which can be simply upgraded to 5G or any future mobile technologies. It can receive Cellular offshore to minimize satellite bandwidth cost and also supports WiFi near marina facilities. The HD4 MBX also comes with 8x PoE outputs to power your IP phones, cameras, and access points. To reduce cable needs on the yacht, we installed a Peplink SD-Switch to provide additional ports for your extra devices. The whole system is fully manageable with InControl2 to remotely monitor, troubleshoot and configure your network. To avoid changing SIM cards up the mast we added an SIM Injector, which adds up to 150 meters of flexibility between the router and the SIM cards, enabling you to place the cellular router in the best location for cellular reception. The SIM Injector adds up to 8 SIM cards. If you cross international boundaries, you can also use it to switch between multiple SIM cards, preventing roaming charges and maintaining unbreakable connectivity. Reliable Internet connectivity can provide enhanced enjoyment onboard and increased navigational capabilities by providing reliable access to online resources such as nautical charts, weather reports and information about the nearest marina, additionally connectivity for the guests and crew is becoming something that is expected onboard. Internet Radio, TV, news and social media as well as secure corporate communications are essential at sea. Peplink provides unbreakable maritime connectivity.",
                "expected_output": {
                    "themes": [
                        "Maritime Connectivity",
                        "SD-WAN Solutions",
                        "High-Speed Internet",
                        "Marine Networking",
                        "Peplink Products",
                        "Maritime Applications",
                    ],
                    "technical_summary": "To confront the challenge of getting reliable internet at sea, Pepwave combines multiple cellular connections to provide a reliability and speed and can automatically prioritize lower cost connections to reduce costs. The Pepwave HD4 MBX is a high-performance SD-WAN router with a modular architecture that supports Wifi and comes with 8x PoE outputs. The system is fully manageable with InControl2, allowing for remote monitoring, troubleshooting, and configuration. Adding a SIM injector adds up to 8 SIM cards that can be switched on demand when crossing international boundaries or to prevent roaming charges while at sea.",
                    "is_useful": True,
                },
            },
            {
                "primary_content": "Hey what's up everyone so today I tried to set up my new router and it was a complete disaster first of all the box was really hard to open I don't know why companies make packaging so difficult to get into anyway I finally got it open and I was excited to try this new expensive router I heard about so I plug everything in and guess what nothing happened absolutely nothing the lights didn't even come on so I checked all the connections tried a different outlet all that stuff still nothing so I called customer service and was on hold for like 45 minutes which was super annoying the guy on the phone was asking me a bunch of questions that I already tried and then he tells me oh it sounds like you got a defective unit I was like yeah no kidding so now I have to pack it all up send it back and wait for a replacement which is going to take another week at least just wanted to vent about this whole experience because I was planning to use this weekend to set up my new home network and now that's not happening anyway that's my story of tech fails for today hope your day is going better than mine",
                "expected_output": {
                    "themes": [
                        "router setup",
                        "defective hardware",
                        "customer service experience",
                    ],
                    "technical_summary": "There is no useful technical information in this content.",
                    "is_useful": False,
                },
            },
        ]
