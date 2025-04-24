# %%
from evals.generate_testset import TransformTestset

transform_testset = TransformTestset("testset-200_max_context_25-04-23")
transform_testset.create_query_set_text_file()
transform_testset = TransformTestset("testset-200_main_testset_25-04-23")
transform_testset.create_query_set_text_file()
transform_testset = TransformTestset("testset-200_reduced_context_25-04-23")
transform_testset.create_query_set_text_file()

# %%
from evals.generate_testset import GenerateTestSet, GPT_4_1_MODEL

generate_testset = GenerateTestSet(
    testset_name="ludicrous_context",
    testset_size=15,
    max_context_token_count=100_000,
    temperature=0.6,
    non_sibling_target_cluster_size=200,
    min_cluster_size=75,
    llm_model=GPT_4_1_MODEL,
    doc_text_column="technical_summary",
)
