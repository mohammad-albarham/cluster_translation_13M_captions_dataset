from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="/home/malbarham/Arabic-Image-Captioning-latest/ccs_synthetic_ar_8000000_10000000_translated_v2.csv",
    path_in_repo="ccs_synthetic_ar_8000000_10000000_translated_v2.csv",
    repo_id="Arabic-Image-Captioning-latest/13M_dataset_CC_facebook_nllb-200-distilled-1.3B",
    repo_type="dataset",

)
