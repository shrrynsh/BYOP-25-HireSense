# BYOP-25-HireSense
My Project contain 3 pyhton notebooks namely: 1) jdnertrainnotebooknew
                                              2)resumenertrainnotebooknew
                                              3)finalbyopnew

The 1st file contains the training code of the Job Description Parser Model,The 2nd file contains the training code for the Resume Parser Model.
In The third file file I have loaded these two models to return the parse Resusmes and and Job Descriptions in JSON format of which the embeddings will be made by the T5 model.In this notebook I have calculated the similarity of an Electrical Engineer JD with the resumes of Electrical Engineer,Animator and  a Bar Tender.

You have to upload the resumes and JDs of which embeddings has to be calculated in the third notebook.

For Resumes use this format:

text=text_preprocess(pdf_load("path of the resume"))
ensemble = EnsembleNERResume(
        pretrained_model_path="/kaggle/input/vvvvvvvvvvvvv/resume_ner_model_pickle.pkl",
        device="cuda" if torch.cuda.is_available() else "cpu")

  
predictions = ensemble.predict(text1)
parsed_resume=group_entities_unique_resume(predictions1)

Here parsed_resume contains the parsed resume in JSON Format.

For JDs use this Format:

jd=text_preprocess("type your jd here")
ensemble = EnsembleNERJd(
        pretrained_model_path="/kaggle/input/vvvvvvvvvvvvv/jd_ner_model_pickle.pkl",
        device="cuda" if torch.cuda.is_available() else "cpu")

  
predictionsjd = ensemble.predict(jd)
parsed_jd=group_entities_unique_jd(predictionsjd)

Here parsed_jd contains the parsed JD in JSON Format.

Then upload parsed_resume and parsed_jd in the eval() function to calculate similarity score.


I have also provided Training Data as well as Training Data.
The Training Data contains some resumes and JDs to test.The Resumes are present in pdf format whereas the JDs are present in txt format.
