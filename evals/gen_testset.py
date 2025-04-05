from ragas.testset.graph import KnowledgeGraph
from ragas.testset.persona import Persona
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

network_persona = Persona(
    name="Technical Network Engineer",
    role_description="A professional specializing in network configurations and troubleshooting, particularly with experience in WAN setups and integrating satellite internet systems like Starlink with networking hardware such as Peplink Balance and Juniper SRX. This individual actively participates in technical forums to share knowledge, solve problems, and seek advice on complex network-related issues.",
)
full_time_rv_persona = Persona(
    name="Full-Time RV Owner",
    role_description="A full-time RV owner who is interested in using Pepwave devices to connect to the internet while traveling.",
)


LLM_MODEL = "gpt-4o"
TESTSET_SIZE = 50
output_dir = Path(__file__).parent / "output"
kg_path = output_dir / "kg_output__LATEST.json"

kg = KnowledgeGraph.load(kg_path)

# df = dataset.to_pandas()
# current_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
# to_serialized_parquet(
#     df, Path.joinpath(evals_dir, f"ragas_testset_{current_date}.parquet")
# )
# dataset.upload()
