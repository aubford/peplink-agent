from pathlib import Path
from generate_testset_class import GenerateTestSet
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the output directory
output_dir = Path(__file__).parent / "output"

# Create an instance of GenerateTestSet
generator = GenerateTestSet(
    output_dir=output_dir,
    testset_size=50,
)

# Generate the test set
generator.get_clusters()
