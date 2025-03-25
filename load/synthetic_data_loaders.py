from abc import ABC, abstractmethod
from pydantic import BaseModel
from langchain.document_loaders import Document
from load.batch_manager import BatchManager


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
                lead_content=example["lead_content"],
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

    def create_batch_job(self, documents: list[Document]):
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
                max_tokens=2040,
            )
        self.batch_manager.test_batchfile()
        # todo: undo
        # self.batch_manager.create_batch_job()


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
1. A list of technical themes discussed.
2. A summary of the technical facts that can be gleaned from the conversation. Provide this in the form of sentences, not lists or other special formatting.
3. An assessment of whether there is useful technical information related to Pepwave products or IT networking.

Important guidelines:
- Focus only on technical content and information in your analysis.
- "Useful technical information" means factual statements, not questions asking for information.
- Be specific and precise in identifying themes.
- Provide a concise, factual summary that captures the key technical points.
- Base your analysis only on the provided content, do not make assumptions.
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
Your task is to analyze YouTube video content and extract key information.

You will be provided with an excerpt of a transcript of a YouTube video.

Analyze this content and provide a structured output containing:
1. A list of technical themes discussed in the video.
2. A concise summary of the most useful technical facts that can be gleaned from the video transcript. Provide this in the form of sentences, not lists or other special formatting.
3. An assessment of whether there is enough useful technical information related to Pepwave products or IT networking to be worth watching.

Important guidelines:
- Focus only on technical content and information in your analysis.
- "Useful technical information" means factual statements, demonstrations, or tutorials and not statements made as part of a question or opinion.
- Be specific and precise in identifying themes.
- Provide a concise, factual summary that captures the key technical points.
- Base your analysis only on the provided content, do not make assumptions.
- YouTube transcripts are often imperfect, so do your best to extract meaning despite potential transcription errors.
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

        Returns:
            List of example dictionaries.
        """
        return [
            {
                "lead_content": "In this video, I do a deep dive into the Pepwave MAX Transit Duo CAT12 router. I'll show you the unboxing, setup process, and how to configure it for optimal performance in an RV or mobile setup. We'll look at antenna connection, cellular settings, and Wi-Fi configuration.",
                "primary_content": "Hello everyone welcome to my channel today I'm going to be walking you through the Pepwave MAX Transit Duo CAT12 router which is a great solution for mobile internet especially for RVs boats and other mobile applications so let's get started first let's look at what comes in the box you get the router itself two cellular SIM card slots power adapter and mounting brackets I've already mounted mine in my RV's technology cabinet now let's talk about the antennas this unit requires external antennas for best performance you'll need to connect them to the ports labeled cell on the back of the router I'm using MIMO antennas which really help boost performance in fringe areas now let's go through the configuration process I've connected to the router using the default IP address 192.168.50.1 and the default password is admin the first thing I recommend is going to the SIM card settings under Network Mobile settings here you can set up your carrier details APN settings and also enable band locking which can really help improve your connection in certain areas for example if you're in a rural location you might want to lock to bands 12 13 or 71 depending on your carrier as these lower frequencies provide better range and building penetration another useful feature is the Wi-Fi as WAN capability which allows you to connect to campground Wi-Fi and use that as your primary connection while keeping cellular as backup you configure this under Network Wi-Fi WAN and just scan for available networks the SpeedFusion technology in these devices is really amazing for combining connections I use it to bond my cellular connection with the campground Wi-Fi for increased reliability especially when I'm on video calls for work if you're having issues with your connection make sure to check signal strength under the Status page this will help you determine if repositioning your antennas might help that's it for the basic setup and configuration of the Pepwave MAX Transit Duo I hope this helps you get the most out of your mobile internet setup if you have any questions drop them in the comments below",
                "expected_output": {
                    "themes": [
                        "Pepwave MAX Transit Duo CAT12",
                        "RV/mobile internet setup",
                        "antenna configuration",
                        "cellular settings",
                        "band locking",
                        "Wi-Fi as WAN",
                        "SpeedFusion technology",
                        "signal optimization",
                    ],
                    "technical_summary": "The Pepwave MAX Transit Duo CAT12 router is designed for mobile internet applications like RVs and boats. For optimal performance, it requires external MIMO antennas connected to the ports labeled 'cell'. The router can be configured through its web interface at 192.168.50.1. Important settings include SIM card configuration under Network > Mobile settings, where users can set carrier details, APN settings, and enable band locking for better performance in rural areas by selecting lower frequency bands (12, 13, or 71). The device supports Wi-Fi as WAN capability, allowing users to connect to external Wi-Fi networks as a primary connection while keeping cellular as backup. Its SpeedFusion technology allows bonding cellular and Wi-Fi connections for increased reliability during video calls.",
                    "is_useful": True,
                },
            },
            {
                "lead_content": "I created this quick video to show how I set up my mobile internet solution in my van. Using a cellular router for remote work.",
                "primary_content": "Hey guys I wanted to show you my internet setup for my van so I can work remotely from anywhere first thing I want to mention is I got this little router it's a Peplink Balance One works really nicely but kind of expensive I paid about 600 bucks for it but totally worth it for reliable internet I've mounted it here near my electrical system it's connected to a small inverter which powers it from my solar setup I've got two SIM cards in it one from Verizon and one from AT&T so if one carrier doesn't have good coverage in an area I can switch to the other one thing that's been super helpful is I added these cell antennas on the roof of the van they connect directly to the router and boost my signal a lot especially when I'm in remote areas to set everything up I just went into the admin page and created two profiles one for each carrier the great thing is I can set it to automatically switch between carriers based on signal strength no manual switching needed when I get to a coffee shop or somewhere with WiFi I can set the router to use that as the primary connection and keep the cellular as backup the battery usage isn't too bad it draws about 10 watts so with my solar system I can easily keep it running all day while working it's been super reliable for zoom calls and uploading large files which I need for my job that's my quick tour of my mobile internet setup let me know if you have any questions in the comments",
                "expected_output": {
                    "themes": [
                        "mobile internet solutions",
                        "Peplink Balance One",
                        "van life networking",
                        "dual SIM configuration",
                        "cellular antennas",
                        "automatic carrier switching",
                        "power consumption",
                        "remote work setup",
                    ],
                    "technical_summary": "This video demonstrates a mobile internet setup using a Peplink Balance One router in a van. The router is connected to a small inverter powered by a solar system and contains two SIM cards (Verizon and AT&T) for redundancy. Roof-mounted cellular antennas are connected directly to the router to boost signal in remote areas. The router is configured with two profiles (one for each carrier) and can automatically switch between carriers based on signal strength. When Wi-Fi is available, the router can use it as the primary connection with cellular as backup. The router draws approximately 10 watts of power, making it sustainable for all-day use with a solar power system.",
                    "is_useful": True,
                },
            },
            {
                "lead_content": "Technology Fail! Watch what happened when I tried to set up my new router",
                "primary_content": "Hey what's up everyone so today I tried to set up my new router and it was a complete disaster first of all the box was really hard to open I don't know why companies make packaging so difficult to get into anyway I finally got it open and I was excited to try this new expensive router I heard about so I plug everything in and guess what nothing happened absolutely nothing the lights didn't even come on so I checked all the connections tried a different outlet all that stuff still nothing so I called customer service and was on hold for like 45 minutes which was super annoying the guy on the phone was asking me a bunch of questions that I already tried and then he tells me oh it sounds like you got a defective unit I was like yeah no kidding so now I have to pack it all up send it back and wait for a replacement which is going to take another week at least just wanted to vent about this whole experience because I was planning to use this weekend to set up my new home network and now that's not happening anyway that's my story of tech fails for today hope your day is going better than mine",
                "expected_output": {
                    "themes": [
                        "router setup",
                        "defective hardware",
                        "customer service experience",
                    ],
                    "technical_summary": "The video recounts an experience with a defective router that wouldn't power on despite trying different outlets and checking connections. After a long customer service call, it was determined to be a defective unit that needed to be returned and replaced.",
                    "is_useful": False,
                },
            },
        ]
