# ancp-bids-apps
A collection of standard apps/pipelines that can cope with BIDS compliant datasets using the ancpbids package.


## Nilearn First Level Analysis
This is app is a demo on how to interact with the ancpBIDS API.
It is a replication of `nilearn.glm.first_level.first_level.first_level_from_bids()`. 

Example call:

    python app.py -a nilearnfirstlevelapp --args \
        -dp ~/datasets/fMRI-language-localizer-demo-dataset \
        -dvf derivatives/fmriprep \
        -tl languagelocalizer -c language-string \
        -if desc preproc

Result:
Creates a new derivatives folder 'ancpbidsfla' with a model written to .nii.gz for each subject.
Additionally, a default plot is generated as a PNG file.