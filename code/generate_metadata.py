from analysis_pipeline_utils.metadata import get_codeocean_process_metadata
from aind_data_schema.core.metadata import Metadata, Processing
import os
from pathlib import Path

computation_id = os.getenv("CO_COMPUTATION_ID")
capsule_id = os.getenv("CO_CAPSULE_ID")
os.environ["CODEOCEAN_EMAIL"] = "michael.xie@alleninstitute.org"
os.environ["CODEOCEAN_DOMAIN"] = "codeocean.allenneuraldynamics.org"

process = get_codeocean_process_metadata(
    capsule_id=capsule_id,
    computation_id=computation_id
)
# read metadata from json in attached assets
# (alternatively could query via aind-data-access-api with names stored in process)
combined_data_path = Path("/data/zstacks")
input_md_paths = combined_data_path.glob("./*/metadata.nd.json")
input_md = [Metadata.model_validate_json(path.read_text()) for path in input_md_paths]

data_description_args = dict(data_summary="...")
md = Metadata.from_metadata(
    input_md,
    process_name="simulated",
    new_processing=Processing(data_processes=[process]),
    **data_description_args
)
md.write_standard_files("/results")